"""Dev mock of the gpupool coordinator API for working on the UI without a cluster.

    uv run python scripts/ui_mock_server.py   ->  http://127.0.0.1:8090   (admin key: dev)

Everything is in memory and driven by wall-clock time: GPU utilization fluctuates, a download
progresses, models walk stopped -> starting -> running. About 30 s after start, server
CTG-Server-2 "dies" (events node_offline, realloc_started, realloc_done) and at ~50 s one GPU of
CTG-Server-1 vanishes (gpu_missing). POST /api/_mock/kill/{node_id} triggers the outage by hand.
Setting ctx_size above 32768 makes /plan answer 409 (does not fit). POST /api/recommend answers with
ranked options normally and with `not_possible` when the estimated need exceeds the biggest server
(e.g. ctx_size 131072 on the 8B file, about 3 MB per context token). GET /api/capacity mirrors the live GPU numbers.
Autoscaling: the model "chat-demand" starts in state `idle` (unloaded; POST /start loads it), "chat-auto" is an
autoscaled running model whose busy ratio oscillates; GET /api/models/{name}/scaling returns plausible data and a few
scaled_up / cold_start / unloaded_idle events are seeded. PUT /api/models/{name} validates the scaling fields (422).
Preemption: specs carry `preemptible` (default true; "chat-demand" is false). POST /api/recommend adds an option with
fits_now false + requires_preemption when the request priority is 80 or more. POST /api/simulate is a pure dry run:
- an add / change that cannot fit at all (huge ctx) -> `unplaced`;
- priority above a running preemptible model's, or 3+ replicas -> `preempt` (+ `start`); priority 90 with 4+ replicas
  also shows `stop` (an autoscaled model shrinks) and `unplaced` (not enough GPUs);
- an edit that changes nothing, or only lowers replicas, -> empty / `stop`.
A `preempted` warning event is seeded.
Model files: GET /api/library/browse lists /models (host folder /srv/gguf) with an in-library file, a split part and a broken
link; POST /api/library {path} accepts a listed file by its /models or /srv/gguf path and answers 400 with the long
"No such file inside the coordinator ..." message for anything else.
KV cache / speculative decoding: PUT /api/models/{name} takes kv_cache_type (f16|q8_0|q4_0), speculative (none|ngram|draft|mtp),
draft_file (ready library file, required for "draft") and draft_n_max (1-16, default 4); 422 for a missing/unready draft, a draft equal to
the model file, or a tokenizer mismatch (names starting with "llama" vs the others). Specs carry kv_cache_type, speculative, draft
("coordinator://<file>" or null) and draft_n_max; placements of draft models carry draft_est_mb. "chat-auto" is seeded with q8_0 + n-gram.
POST /api/recommend and /api/simulate accept the same optional fields (q8_0 / q4_0 shrink the KV estimate).
Rebalancing: POST /api/rebalance {dry_run} lists one qualifying move (a "chat-auto" replica onto CTG-Server-1/CUDA0). With
dry_run false it starts the move: a replacement replica appears (state starting), GET /api/state carries
`rebalance.in_progress`, and ~20 s later the old replica is dropped (events rebalance_started, rebalanced). Afterwards the
dry run answers with no moves ("all replicas are well placed"). `rebalance.next_run_ts` is 10 min after start.
Conversion (Hugging Face / server folder -> GGUF): every route of the real API under /api/convert*. Inspectable sources:
HuggingFaceTB/SmolLM2-135M-Instruct (small, supported), Qwen/Qwen2.5-7B-Instruct (job ends in needs_review: one tokenizer mismatch),
Qwen/Qwen2.5-72B-Instruct (recommendation depends on the live GPUs), Qwen/Qwen2.5-7B-Instruct-AWQ (pre-quantized, unsupported, base_model set),
acme/NovelNet-7B (unsupported architecture), meta-llama/Llama-3.1-8B-Instruct (gated: the job fails at the download),
acme/Quirky-3B (ships remote code: the job fails unless allow_remote_code) and the server folders /models/hf/smollm2-135m and
/models/hf/broken-model (fails while converting). Those repos have no .gguf files in GET /api/hf/files. Jobs walk queued -> downloading
-> converting -> quantizing -> validating in about 20 s of wall-clock time. Three jobs are seeded at start: done (in the library),
failed (Quirky-3B) and a running Qwen2.5-7B job. POST /api/_mock/convert_available {"available": false} simulates a missing toolchain; {"imatrix_available": false} a coordinator
without llama-imatrix (importance-matrix types are refused, options.imatrix_available false); {"disk_free_gb": 100} a small disk
(acme/Huge-400B, the 72B model, ... then answer 507 at submit). Types that need an importance matrix (IQ1/IQ2/IQ3_XXS/IQ3_XS) and
every type under 4 bits (auto) walk through a "calibrating" stage; imatrix off for those types is 422. A calibration path ending in
corrupt.txt fails in calibrating. Failed and cancelled jobs carry failed_stage; inspect carries name_stem.
"""
from __future__ import annotations

import math
import re
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.staticfiles import StaticFiles

ADMIN_KEY = "dev"
UI_DIR = Path(__file__).resolve().parent.parent / "src" / "gpupool" / "ui"

T0 = time.time()
LAST_TICK = [T0]
EVENTS: list[dict] = []
SCHEDULED: list[tuple[float, object]] = []  # (absolute time, callable)
REBAL: dict = {"in_progress": None, "done": False, "next_run_ts": None}
REBAL_SECONDS = 20.0


COMPUTE_CAP = {"NVIDIA H100 80GB": "9.0", "NVIDIA RTX 4090": "8.9", "NVIDIA A100 40GB": "8.0", "NVIDIA RTX 3090": "8.6"}
BANDWIDTH_GBPS = {"NVIDIA H100 80GB": 3350.0, "NVIDIA RTX 4090": 1008.0, "NVIDIA A100 40GB": 1555.0, "NVIDIA RTX 3090": 936.2}


def _gpu(i: int, name: str, total_mb: int, phase: float, driver="535.154.05", cuda="12.2") -> dict:
    return {"device_id": f"CUDA{i}", "kind": "cuda", "name": name, "total_mb": total_mb, "free_mb": total_mb,
            "usable_mb": total_mb, "util_pct": 0, "temp_c": 40, "power_w": 60, "processes": [],
            "driver": driver, "cuda": cuda, "bandwidth_gbps": BANDWIDTH_GBPS.get(name),
            "compute_cap": COMPUTE_CAP.get(name), "kernels_ok": True, "_phase": phase, "_base": 20 + 12 * i}


def _server(node_id: str, ip: str, gpus: list[dict], alive: bool, ram_total: int) -> dict:
    return {"node_id": node_id, "agent_url": f"http://{ip}:7070", "added_at": T0 - 3600, "alive": alive,
            "last_seen": time.time(), "host": ip, "gpus": gpus, "gpu_enabled": {g["device_id"]: True for g in gpus},
            "cpu_pct": 30.0, "ram_used_mb": int(ram_total * 0.4), "ram_total_mb": ram_total}


SERVERS: dict[str, dict] = {}
MODELS: dict[str, dict] = {}
LIBRARY: dict[str, dict] = {}
SETTINGS = {"public_url": "http://127.0.0.1:8090", "cluster_token": "tok_mock_123", "api_keys_set": True}


def reset() -> None:
    global T0
    T0 = time.time()
    EVENTS.clear()
    SCHEDULED.clear()
    REBAL.update(in_progress=None, done=False, next_run_ts=T0 + 600)
    SERVERS.clear()
    MODELS.clear()
    LIBRARY.clear()
    SERVERS["CTG-Server-1"] = _server("CTG-Server-1", "10.0.0.5", [
        _gpu(0, "NVIDIA H100 80GB", 81920, 0.0), _gpu(1, "NVIDIA H100 80GB", 81920, 1.3),
        _gpu(2, "NVIDIA H100 80GB", 81920, 2.1), _gpu(3, "NVIDIA H100 80GB", 81920, 3.4)], True, 262144)
    SERVERS["CTG-Server-2"] = _server("CTG-Server-2", "10.0.0.6", [
        _gpu(0, "NVIDIA RTX 4090", 24576, 0.7, "550.54", "12.4"), _gpu(1, "NVIDIA RTX 4090", 24576, 2.5, "550.54", "12.4"),
        _gpu(2, "NVIDIA RTX 4090", 24576, 4.0, "550.54", "12.4")], True, 131072)
    off = _server("CTG-Server-3", "10.0.0.7", [_gpu(0, "NVIDIA A100 40GB", 40960, 1.0)], False, 65536)
    off["last_seen"] = T0 - 600
    SERVERS["CTG-Server-3"] = off
    LIBRARY["qwen2.5-3b-q4.gguf"] = {"name": "qwen2.5-3b-q4.gguf", "path": "/data/models/qwen2.5-3b-q4.gguf", "source": "hf",
        "hf_repo": "Qwen/Qwen2.5-3B-Instruct-GGUF", "hf_file": "qwen2.5-3b-q4.gguf", "bytes": 2_000_000_000,
        "downloaded": 2_000_000_000, "status": "ready", "error": None, "created_at": T0 - 1000}
    LIBRARY["qwen2.5-0.5b-q8.gguf"] = {"name": "qwen2.5-0.5b-q8.gguf", "path": "/data/models/qwen2.5-0.5b-q8.gguf", "source": "hf",
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "hf_file": "qwen2.5-0.5b-q8.gguf", "bytes": 530_000_000,
        "downloaded": 530_000_000, "status": "ready", "error": None, "created_at": T0 - 900}
    LIBRARY["llama-3.2-1b-q8.gguf"] = {"name": "llama-3.2-1b-q8.gguf", "path": "/data/models/llama-3.2-1b-q8.gguf", "source": "hf",
        "hf_repo": "bartowski/Llama-3.2-1B-GGUF", "hf_file": "llama-3.2-1b-q8.gguf", "bytes": 1_320_000_000,
        "downloaded": 1_320_000_000, "status": "ready", "error": None, "created_at": T0 - 800}
    LIBRARY["llama-8b.gguf"] = {"name": "llama-8b.gguf", "path": "/data/models/llama-8b.gguf", "source": "hf",
        "hf_repo": "bartowski/Llama-3.1-8B-GGUF", "hf_file": "llama-8b.gguf", "bytes": 5_000_000_000,
        "downloaded": 500_000_000, "status": "downloading", "error": None, "created_at": T0, "_t": T0}
    MODELS["qwen3b"] = {"spec": {"name": "qwen3b", "source": "coordinator://qwen2.5-3b-q4.gguf", "ctx_size": 4096,
        "parallel": 1, "replicas": 0, "pin_devices": [], "priority": 50, "preemptible": True, "spread": "gpu"}, "file": "qwen2.5-3b-q4.gguf", "state": "stopped", "error": None,
        "replicas": [], "_t": 0.0}
    MODELS["chat-auto"] = {"spec": {"name": "chat-auto", "source": "coordinator://qwen2.5-3b-q4.gguf", "ctx_size": 8192, "parallel": 4,
        "replicas": 2, "pin_devices": [], "priority": 70, "preemptible": True, "spread": "gpu", "min_replicas": 1, "max_replicas": 4,
        "autoscale": {"target_busy": 0.7, "up_after_s": 30, "down_after_s": 300}, "idle_unload_s": None,
        "kv_cache_type": "q8_0", "speculative": "ngram", "draft": None, "draft_n_max": 4},
        "file": "qwen2.5-3b-q4.gguf", "state": "running", "error": None, "replicas": [], "_t": 0.0}
    _place(MODELS["chat-auto"], 2)
    MODELS["chat-demand"] = {"spec": {"name": "chat-demand", "source": "coordinator://qwen2.5-3b-q4.gguf", "ctx_size": 4096, "parallel": 2,
        "replicas": 1, "pin_devices": [], "priority": 30, "preemptible": False, "spread": "gpu", "min_replicas": 0, "max_replicas": 2,
        "autoscale": {"target_busy": 0.7, "up_after_s": 30, "down_after_s": 300}, "idle_unload_s": 600},
        "file": "qwen2.5-3b-q4.gguf", "state": "idle", "error": None, "replicas": [], "_t": 0.0}
    add_event("info", "unloaded_idle", "chat-demand unloaded after 10 min without requests", None, "chat-demand")
    add_event("info", "cold_start", "chat-demand was unloaded; a request loaded it (cold start)", None, "chat-demand")
    add_event("info", "scaled_up", "chat-auto scaled up to 2 replicas (busy 0.82 above target 0.70 for 30 s)", None, "chat-auto")
    add_event("warning", "preempted", "chat-batch-1a2b3c stopped to make room for chat-auto (priority 20 < 70)", None, "chat-batch")
    add_event("warning", "rebalance_failed", "Rebalance of chat-batch aborted: replacement replica did not become ready in time", None, "chat-batch")
    SCHEDULED.append((T0 + 30, lambda: kill("CTG-Server-2")))
    SCHEDULED.append((T0 + 50, lambda: vanish_gpu("CTG-Server-1", "CUDA3")))
    if "seed_conversions" in globals():  # defined further down; at import time the first reset() runs before it exists
        seed_conversions()


def add_event(level: str, kind: str, message: str, node_id=None, model=None) -> None:
    EVENTS.insert(0, {"id": (EVENTS[0]["id"] + 1) if EVENTS else 1, "ts": time.time(), "level": level, "kind": kind,
                      "message": message, "node_id": node_id, "model": model, "read": False})
    del EVENTS[200:]


def kill(node_id: str) -> None:
    s = SERVERS.get(node_id)
    if not s or not s["alive"]:
        return
    s["alive"] = False
    hit = [m for m in MODELS.values() if any(a["node_id"] == node_id for r in m["replicas"] for a in r["placement"]["assignments"])]
    add_event("error", "node_offline", f"Server {node_id} went offline (no report for 10 s); {len(hit)} model affected", node_id)
    for m in hit:
        add_event("warning", "realloc_started", f"Re-allocating {m['spec']['name']} away from {node_id}", node_id, m["spec"]["name"])
        m["state"], m["replicas"], m["_t"], m["_manual"] = "starting", [], time.time(), False
        SCHEDULED.append((time.time() + 3, lambda m=m: finish_realloc(m, node_id)))


def finish_realloc(m: dict, node_id: str) -> None:
    if m["state"] != "starting":
        return
    if m["spec"]["ctx_size"] > 32768:
        m["state"], m["error"] = "failed", "Not enough free VRAM to re-allocate: needs 9.2 GB, pool has 6.1 GB"
        add_event("error", "realloc_failed", m["error"], node_id, m["spec"]["name"])
        return
    _place(m)
    add_event("info", "realloc_done", f"{m['spec']['name']} re-allocated and running again", node_id, m["spec"]["name"])


def vanish_gpu(node_id: str, device_id: str) -> None:
    s = SERVERS.get(node_id)
    if s:
        s["gpus"] = [g for g in s["gpus"] if g["device_id"] != device_id]
        add_event("warning", "gpu_missing", f"GPU {device_id} disappeared from {node_id}", node_id)


def _pinned(pins: list, node: str, device: str) -> bool:
    """pin_devices is an allowed set: "<node>/<device>" or "<node>/*" (every device of that server, also GPUs added later)."""
    return not pins or f"{node}/{device}" in pins or f"{node}/*" in pins


def _check_pins(pins: object) -> list:
    """Validate pin_devices like the real API: a list of "<node>/<device>" or "<node>/*" strings naming known servers."""
    if not isinstance(pins, list) or not all(isinstance(p, str) and p.count("/") == 1 and all(p.split("/")) for p in pins):
        raise HTTPException(422, 'pin_devices: must be a list of "<node>/<device>" or "<node>/*" strings')
    for p in pins:
        if p.split("/")[0] not in SERVERS:
            raise HTTPException(422, f"pin_devices entry names an unregistered server {p.split('/')[0]!r}")
    return list(pins)


def _pin_devices(pins: list) -> list[tuple[str, str]]:
    """Expand pins to (node, device) pairs, "<node>/*" to every device the server has now."""
    out: list[tuple[str, str]] = []
    for p in pins:
        node, dev = p.split("/", 1)
        if dev == "*":
            out += [(node, g["device_id"]) for g in SERVERS[node]["gpus"]] if node in SERVERS else []
        else:
            out.append((node, dev))
    return out


def _place(m: dict, replicas: int = 1) -> None:
    """Fake scheduler: first enabled GPUs of the first alive server (or the pinned ones)."""
    pins = m["spec"]["pin_devices"]
    cands = [(s["node_id"], g["device_id"]) for s in SERVERS.values() if s["alive"] for g in s["gpus"]
             if s["gpu_enabled"].get(g["device_id"], True) and _pinned(pins, s["node_id"], g["device_id"])]
    if not cands:
        raise HTTPException(409, "No enabled GPU is available")
    cands.sort(key=lambda c: c[0] != "CTG-Server-2")  # prefer Server-2 so the outage demo has something to move
    chosen = cands[:2]
    m["replicas"] = [{"replica_id": f"{m['spec']['name']}-{i}a2b3c", "model": m["spec"]["name"], "state": "ready",
        "error": None, "outstanding": 0, "created_at": time.time(), "updated_at": time.time(),
        "placement": _placement(m["spec"]["name"], chosen, m["spec"].get("draft"))} for i in range(replicas)]
    m["state"], m["error"] = "running", None


def _placement(name: str, chosen: list[tuple[str, str]], draft: str | None = None) -> dict:
    return {"model": name, "replica_id": f"{name}-plan", "tier": "multi_node" if len({c[0] for c in chosen}) > 1 else "single_node",
            "head_node": chosen[0][0], "head_port": 9000, "tensor_split": [1.0] * len(chosen), "est_total_mb": 5000 * len(chosen),
            "score": 82.5, "est_decode_tps": 96.4, "draft_est_mb": _draft_mb(draft),
            "reasons": ["fastest GPUs with room (about 1008 GB/s)", "spread: replicas on different GPUs"],
            "assignments": [{"node_id": n, "device_id": d, "llama_device": d, "rpc_endpoint": None, "layers": 18 // len(chosen),
                             "est_mb": 5000} for n, d in chosen]}


def _draft_mb(draft: str | None) -> int | None:
    """Estimated VRAM of the draft model (its file size plus a little context), None without a draft."""
    item = LIBRARY.get(str(draft or "").replace("coordinator://", ""))
    return int(item["bytes"] / 1048576 * 1.1 + 120) if item else None


def _rebalance_moves() -> list[dict]:
    """The one qualifying move of the mock: a chat-auto replica that would run better on CTG-Server-1/CUDA0."""
    m = MODELS.get("chat-auto")
    if REBAL["done"] or REBAL["in_progress"] or not m:
        return []
    ready = [r for r in m["replicas"] if r["state"] == "ready"]
    if not ready:
        return []
    r = ready[-1]
    frm = [{"node_id": a["node_id"], "device_id": a["device_id"]} for a in r["placement"]["assignments"]]
    to = [{"node_id": "CTG-Server-1", "device_id": "CUDA0"}]
    if frm == to:
        return []
    return [{"replica_id": r["replica_id"], "model": m["spec"]["name"], "from": frm, "to": to, "current_score": 48.0,
             "new_score": 91.5, "gain": 43.5, "reasons": ["fits on 1 GPU instead of %d" % len(frm), "fastest GPU with room (about 3350 GB/s)"]}]


def _finish_rebalance(model: str, old: str, new: str) -> None:
    REBAL["in_progress"] = None
    m = MODELS.get(model)
    reps = {r["replica_id"]: r for r in (m["replicas"] if m else [])}
    if old not in reps or new not in reps:
        add_event("warning", "rebalance_failed", f"Rebalance of {model} aborted: replica {old if old not in reps else new} is gone", None, model)
        return
    reps[new]["state"] = "ready"
    m["replicas"] = [r for r in m["replicas"] if r["replica_id"] != old]
    REBAL["done"] = True
    add_event("info", "rebalanced", f"{model}: replica {old} replaced by {new} on better GPUs", None, model)


def tick() -> None:
    now = time.time()
    for item in list(SCHEDULED):
        if item[0] <= now:
            SCHEDULED.remove(item)
            item[1]()
    for it in LIBRARY.values():
        if it["status"] == "downloading":
            it["downloaded"] = min(it["bytes"], int(500_000_000 + (now - it["_t"]) * 150_000_000))
            if it["downloaded"] >= it["bytes"]:
                it["status"] = "ready"
    for m in MODELS.values():
        if m["state"] == "starting" and m.get("_manual") and now - m["_t"] > 4 and not m["replicas"]:
            _place(m, max(1, int(m["spec"].get("replicas") or 1)))
            add_event("info", "model_started", f"Model {m['spec']['name']} is running", None, m["spec"]["name"])
        if m["state"] == "stopping" and now - m["_t"] > 2:
            m["state"], m["replicas"] = "stopped", []
    # telemetry
    for s in SERVERS.values():
        if not s["alive"]:
            continue
        s["last_seen"] = now
        s["cpu_pct"] = round(35 + 25 * math.sin(now / 5 + len(s["node_id"])), 1)
        for g in s["gpus"]:
            util = max(0, min(100, int(g["_base"] + 45 * math.sin(now / 3 + g["_phase"]))))
            used = int(g["total_mb"] * (0.15 + 0.7 * util / 100))
            g.update(util_pct=util, free_mb=g["total_mb"] - used, usable_mb=max(0, g["total_mb"] - used - 1024),
                     temp_c=45 + util // 3, power_w=70 + util * 4,
                     processes=[{"pid": 18231 + hash(g["device_id"]) % 100, "name": "python", "used_mb": used}] if util > 25 else [])


def _device(g: dict) -> dict:
    return {k: v for k, v in g.items() if not k.startswith("_")}


def _server_json(s: dict) -> dict:
    devices = [_device(g) for g in s["gpus"]]
    devices.append({"device_id": "CPU", "kind": "cpu", "name": "CPU", "total_mb": s["ram_total_mb"],
                    "free_mb": s["ram_total_mb"] - s["ram_used_mb"], "usable_mb": 0, "util_pct": None, "temp_c": None,
                    "power_w": None, "processes": [], "driver": None, "cuda": None})
    report = {"node_id": s["node_id"], "agent_url": s["agent_url"], "host": s["host"], "devices": devices, "engines": [],
              "llama_version": "b4000", "models": [], "ts": s["last_seen"], "cpu_pct": s["cpu_pct"],
              "ram_used_mb": s["ram_used_mb"], "ram_total_mb": s["ram_total_mb"]}
    return {"node_id": s["node_id"], "agent_url": s["agent_url"], "added_at": s["added_at"], "alive": s["alive"],
            "last_seen": s["last_seen"], "report": report, "gpu_enabled": dict(s["gpu_enabled"])}


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def _busy(m: dict) -> float | None:
    """Fake average busy ratio: oscillates for running models, None when nothing runs."""
    if not m["replicas"]:
        return None
    return round(0.5 + 0.4 * math.sin(time.time() / 7 + len(m["spec"]["name"])), 2)


PERF_DEFAULTS = {"kv_cache_type": "f16", "speculative": "none", "draft": None, "draft_n_max": 4}
KV_FACTOR = {"f16": 1.0, "q8_0": 0.53, "q4_0": 0.28}


def _tok_family(file: str) -> str:
    return "llama" if str(file).startswith("llama") else "other"


def _perf_fields(body: dict, file: str) -> dict:
    """Validate kv_cache_type / speculative / draft_file / draft_n_max (all optional) like the real API; returns the spec fields."""
    kv = _choice(body.get("kv_cache_type", "f16"), tuple(KV_FACTOR), "kv_cache_type")
    spec = _choice(body.get("speculative", "none"), ("none", "ngram", "draft", "mtp"), "speculative")
    n = _int_in(body.get("draft_n_max", 4), 1, 16, "draft_n_max")
    draft = None
    if spec == "draft":
        df = body.get("draft_file")
        if not df or df not in LIBRARY or LIBRARY[df]["status"] != "ready":
            raise HTTPException(422, f"speculative 'draft' needs a ready draft_file from the library (got {df!r})")
        if df == file:
            raise HTTPException(422, "draft_file must differ from the model file")
        if _tok_family(df) != _tok_family(file):
            raise HTTPException(422, f"draft model {df} does not share the tokenizer of {file}")
        draft = f"coordinator://{df}"
    fa = _choice(body.get("flash_attn", "auto"), ("auto", "on", "off"), "flash_attn")
    ub = _int_in(body.get("ubatch", 512), 32, 8192, "ubatch")
    b = max(_int_in(body.get("batch", 2048), 32, 16384, "batch"), ub)
    if kv != "f16" and fa == "off":
        raise HTTPException(422, f"KV cache {kv} needs flash attention (auto or on)")
    return {"kv_cache_type": kv, "speculative": spec, "draft": draft, "draft_n_max": n,
            "flash_attn": fa, "ubatch": ub, "batch": b, "kv_unified": bool(body.get("kv_unified", False))}


def _mock_tips(body: dict, perf: dict, ctx: int, parallel: int, need: int, biggest_gpu: int) -> list[dict]:
    """A plausible subset of the real tuning suggestions (coordinator/tuning.py)."""
    tips = []

    def tip(tid, kind, title, detail, apply, tier="single_gpu"):
        tips.append({"id": tid, "kind": kind, "title": title, "detail": detail, "apply": apply, "tier": tier,
                     "est_decode_tps": None, "est_total_mb": None})
    if need > biggest_gpu and perf["kv_cache_type"] == "f16":
        tip("kv_cache", "fix", "Quantize the KV cache to q8_0 to fit on a single GPU",
            "Saves about half of the KV memory with negligible quality loss.",
            {"kv_cache_type": "q8_0", **({"flash_attn": "auto"} if perf["flash_attn"] == "off" else {})})
    if need <= biggest_gpu:
        if parallel == 1:
            tip("parallel", "throughput", "Serve 4 requests at once (4 slots)",
                f"Each slot keeps {ctx} tokens of context (total {ctx * 4}).", {"parallel": 4, "ctx_size": ctx * 4})
        if perf["speculative"] == "none":
            tip("ngram", "speed", "Speculative decoding with n-gram (no extra memory)",
                "Helps when outputs repeat the input (code edits, RAG, extraction).", {"speculative": "ngram"})
        if perf["ubatch"] == 512:
            tip("ubatch", "speed", "Micro-batch 2048 for faster prompt processing",
                "Long prompts are read faster on NVIDIA H100 80GB (Hopper, cc 9.0); about 300 MB more compute memory per GPU.",
                {"ubatch": 2048, "batch": max(perf["batch"], 2048)})
    return tips


def _scaling(m: dict) -> dict:
    sp = m["spec"]
    mn, mx = sp.get("min_replicas"), sp.get("max_replicas")
    n = max(1, int(sp.get("replicas") or 1))
    lo, hi = (n, n) if mn is None or mx is None else (mn, mx)
    desired = len(m["replicas"]) if m["replicas"] else (0 if m["state"] == "idle" else lo)
    return {"min": lo, "max": hi, "desired": desired, "avg_busy": _busy(m), "unloaded": m["state"] == "idle"}


def _state_model(m: dict) -> dict:
    return {**_clean(m), "spec": {**PERF_DEFAULTS, **m["spec"]}, "scaling": _scaling(m)}


def auth(authorization: str = Header(default="")) -> None:
    if authorization != f"Bearer {ADMIN_KEY}":
        raise HTTPException(401, "Invalid admin key")


app = FastAPI(title="gpupool UI mock")
reset()
api = Depends(auth)


@app.get("/api/state", dependencies=[api])
def state() -> dict:
    tick()
    servers = [_server_json(s) for s in SERVERS.values()]
    gpus = [g for s in SERVERS.values() for g in s["gpus"]]
    alive = [s for s in SERVERS.values() if s["alive"]]
    usable = sum(g["usable_mb"] for s in alive for g in s["gpus"] if s["gpu_enabled"].get(g["device_id"], True))
    enabled = sum(1 for s in SERVERS.values() for g in s["gpus"] if s["gpu_enabled"].get(g["device_id"], True))
    return {
        "summary": {"servers_total": len(SERVERS), "servers_online": len(alive), "gpus_total": len(gpus), "gpus_enabled": enabled,
                    "pool_total_mb": sum(g["total_mb"] for s in alive for g in s["gpus"]), "pool_usable_mb": usable,
                    "models_running": sum(1 for m in MODELS.values() if m["state"] == "running")},
        "servers": servers,
        "models": [_state_model(m) for m in MODELS.values()],
        "library": [_clean(i) for i in LIBRARY.values()],
        "settings": dict(SETTINGS),
        "rebalance": {"in_progress": dict(REBAL["in_progress"]) if REBAL["in_progress"] else None, "next_run_ts": REBAL["next_run_ts"]},
        "events": EVENTS[:50],
        "unread_events": sum(1 for e in EVENTS if not e["read"]),
    }


@app.post("/api/servers", dependencies=[api])
def add_server(body: dict) -> dict:
    url = str(body.get("agent_url", "")).strip()
    if not url.startswith("http"):
        raise HTTPException(400, "agent_url must start with http:// or https://")
    host = url.split("//", 1)[1].split(":")[0].split("/")[0]
    node_id = f"node-{host}"
    if node_id in SERVERS:
        raise HTTPException(409, f"node {node_id} already exists")
    SERVERS[node_id] = _server(node_id, host, [_gpu(0, "NVIDIA RTX 3090", 24576, 0.3)], True, 65536)
    add_event("info", "server_added", f"Server {node_id} added", node_id)
    return {"node_id": node_id, "agent_url": url}


@app.delete("/api/servers/{node_id}", dependencies=[api])
def del_server(node_id: str) -> dict:
    if node_id not in SERVERS:
        raise HTTPException(404, "unknown server")
    for m in MODELS.values():
        if any(a["node_id"] == node_id for r in m["replicas"] for a in r["placement"]["assignments"]):
            m["state"], m["replicas"], m["_t"] = "stopped", [], time.time()
    del SERVERS[node_id]
    add_event("info", "server_removed", f"Server {node_id} removed", node_id)
    return {"ok": True}


@app.put("/api/servers/{node_id}/gpus/{device_id}", dependencies=[api])
def set_gpu(node_id: str, device_id: str, body: dict) -> dict:
    s = SERVERS.get(node_id)
    if not s or not any(g["device_id"] == device_id for g in s["gpus"]):
        raise HTTPException(404, "unknown GPU")
    if device_id == "CUDA2" and node_id == "CTG-Server-1" and not body.get("enabled"):
        raise HTTPException(409, "Mock: CUDA2 on CTG-Server-1 cannot be disabled (tests the optimistic revert)")
    s["gpu_enabled"][device_id] = bool(body.get("enabled"))
    return {"enabled": s["gpu_enabled"][device_id]}


@app.get("/api/hf/files", dependencies=[api])
def hf_files(repo: str = Query(...)) -> list[dict]:
    if "/" not in repo:
        raise HTTPException(404, f"Repository {repo} not found")
    if repo.lower() in REPOS:  # a safetensors-only repo: no .gguf files, so the UI offers "Convert to GGUF"
        return []
    base = repo.split("/")[1].replace("-GGUF", "").lower()
    return [{"file": f"{base}-q4_k_m.gguf", "bytes": 2_300_000_000}, {"file": f"{base}-q8_0.gguf", "bytes": 3_900_000_000}]


MODEL_ROOT, HOST_ROOT = "/models", "/srv/gguf"
# (relative name, bytes, split_part, broken_link). `in_library` is computed from LIBRARY.
BROWSE_FILES = [("qwen2.5-3b-q4.gguf", 2_000_000_000, False, False), ("llama-3.1-8b-instruct-q4_k_m.gguf", 4_920_000_000, False, False),
                ("mistral-7b/mistral-7b-instruct-v0.3-q8_0.gguf", 7_700_000_000, False, False),
                ("deepseek-70b/deepseek-r1-70b-q4_k_m-00001-of-00003.gguf", 14_300_000_000, True, False),
                ("old/gemma-2-9b-it.gguf", 0, False, True)]


@app.get("/api/library/browse", dependencies=[api])
def browse_library() -> dict:
    """Model files the coordinator can see. In Docker that is only the mounted folder (/models <- /srv/gguf on the host)."""
    files = [{"path": f"{MODEL_ROOT}/{rel}", "name": rel.split("/")[-1], "bytes": size, "in_library": rel.split("/")[-1] in LIBRARY,
              "split_part": split, "broken_link": broken, "host_path": f"{HOST_ROOT}/{rel}"} for rel, size, split, broken in BROWSE_FILES]
    return {"roots": [{"path": MODEL_ROOT, "exists": True, "host_path": HOST_ROOT}], "files": files, "truncated": False}


def _resolve_model_path(p: str) -> dict:
    """Translate a host path to the mounted one and require the file to be in the browse list (mimics the real coordinator)."""
    if p.startswith(HOST_ROOT + "/"):
        p = MODEL_ROOT + p[len(HOST_ROOT):]
    hit = next((f for f in browse_library()["files"] if f["path"] == p), None)
    if hit is None or hit["broken_link"]:
        raise HTTPException(400, f"No such file inside the coordinator: {p}. The coordinator runs in Docker and only sees its mounted "
                                 f"folders: {MODEL_ROOT} (host folder {HOST_ROOT}). Mount the folder that holds the file with "
                                 f"-v /path/on/host:{MODEL_ROOT}, then paste a path under {MODEL_ROOT} or pick the file from the list.")
    if hit["split_part"]:
        raise HTTPException(400, f"{hit['name']} is one part of a split GGUF; split models are not supported. Merge the parts into one file first.")
    return hit


@app.post("/api/library", dependencies=[api])
def add_library(body: dict) -> dict:
    if body.get("hf_repo"):
        name = str(body.get("hf_file", "")).split("/")[-1]
        item = {"name": name, "path": f"/data/models/{name}", "source": "hf", "hf_repo": body["hf_repo"], "hf_file": body.get("hf_file"),
                "bytes": 3_000_000_000, "downloaded": 0, "status": "downloading", "error": None, "created_at": time.time(), "_t": time.time()}
        item["downloaded"] = 0
    elif body.get("path"):
        p = str(body["path"])
        if not p.startswith("/"):
            raise HTTPException(400, "path must be absolute")
        hit = _resolve_model_path(p)
        name, p = hit["name"], hit["path"]
        item = {"name": name, "path": p, "source": "path", "hf_repo": None, "hf_file": None, "bytes": hit["bytes"],
                "downloaded": 0, "status": "ready", "error": None, "created_at": time.time()}
    else:
        raise HTTPException(422, "hf_repo+hf_file or path required")
    if item["name"] in LIBRARY:
        raise HTTPException(409, f"{item['name']} is already in the library")
    LIBRARY[item["name"]] = item
    return _clean(item)


@app.delete("/api/library/{name}", dependencies=[api])
def del_library(name: str) -> dict:
    if name not in LIBRARY:
        raise HTTPException(404, "unknown library item")
    if any(m["file"] == name for m in MODELS.values()):
        raise HTTPException(409, "a model uses this file")
    del LIBRARY[name]
    return {"ok": True}


@app.put("/api/models/{name}", dependencies=[api])
def put_model(name: str, body: dict) -> dict:
    file = body.get("file")
    if file not in LIBRARY:
        raise HTTPException(404, f"{file} is not in the library")
    perf = _perf_fields(body, file)  # validate before touching any state
    pins_in = _check_pins(body.get("pin_devices") or [])
    m = MODELS.setdefault(name, {"spec": {"name": name, "replicas": 0}, "state": "stopped", "error": None, "replicas": [], "_t": 0.0})
    m["file"] = file
    m["spec"].update(source=f"coordinator://{file}", ctx_size=int(body.get("ctx_size", 4096)), parallel=int(body.get("parallel", 1)),
                     pin_devices=pins_in, priority=_int_in(body.get("priority", 50), 0, 100, "priority"),
                     preemptible=_bool(body.get("preemptible", True), "preemptible"),
                     spread=_choice(body.get("spread", "gpu"), ("gpu", "node", "none"), "spread"))
    m["spec"].update(_scaling_fields(body))
    m["spec"].update(perf)
    return m["spec"]


def _scaling_fields(body: dict) -> dict:
    mn, mx, idle, auto = body.get("min_replicas"), body.get("max_replicas"), body.get("idle_unload_s"), body.get("autoscale")
    if mn is not None:
        mn = _int_in(mn, 0, 64, "min_replicas")
    if mx is not None:
        mx = _int_in(mx, 1, 64, "max_replicas")
    if mn is not None and mx is not None and mn > mx:
        raise HTTPException(422, "min_replicas must not exceed max_replicas")
    if idle is not None:
        idle = _int_in(idle, 1, 10_000_000, "idle_unload_s")
        if mn != 0:
            raise HTTPException(422, "idle_unload_s requires min_replicas 0")
    if auto is not None:
        if not isinstance(auto, dict):
            raise HTTPException(422, "autoscale must be an object")
        tb = auto.get("target_busy", 0.7)
        if not isinstance(tb, (int, float)) or not 0 < tb <= 1:
            raise HTTPException(422, "autoscale.target_busy must be between 0 and 1")
        auto = {"target_busy": tb, "up_after_s": _int_in(auto.get("up_after_s", 30), 0, 10_000_000, "autoscale.up_after_s"),
                "down_after_s": _int_in(auto.get("down_after_s", 300), 0, 10_000_000, "autoscale.down_after_s")}
    return {"min_replicas": mn, "max_replicas": mx, "autoscale": auto, "idle_unload_s": idle}


def _int_in(v, lo: int, hi: int, field: str) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise HTTPException(422, f"{field} must be an integer")
    if not lo <= n <= hi:
        raise HTTPException(422, f"{field} must be between {lo} and {hi}")
    return n


def _bool(v, field: str) -> bool:
    if not isinstance(v, bool):
        raise HTTPException(422, f"{field} must be a boolean")
    return v


def _choice(v, allowed: tuple, field: str) -> str:
    if v not in allowed:
        raise HTTPException(422, f"{field} must be one of {', '.join(allowed)}")
    return v


def _model(name: str) -> dict:
    if name not in MODELS:
        raise HTTPException(404, "unknown model")
    return MODELS[name]


@app.get("/api/models/{name}/scaling", dependencies=[api])
def model_scaling(name: str) -> dict:
    tick()
    m = _model(name)
    sc = _scaling(m)
    reps = m["replicas"]
    avg = sc["avg_busy"]
    state = {"stopped": "stopped", "failed": "stopped", "idle": "unloaded"}.get(m["state"], "fixed" if sc["min"] == sc["max"] else "steady")
    if state == "steady" and avg is not None and avg > (m["spec"].get("autoscale") or {}).get("target_busy", 0.7):
        state = "scaling_up"
    rows = [{"replica_id": r["replica_id"], "busy": round(min(1.0, max(0.0, (avg or 0) + 0.08 * (i - 0.5))), 2),
             "requests_processing": 1 + i, "requests_deferred": 0 if (avg or 0) < 0.7 else 2, "measured_decode_tps": round(88.0 - 6 * i, 1),
             "est_decode_tps": r["placement"]["est_decode_tps"], "metrics_ok": not (i == 1 and name == "chat-auto" and int(time.time()) % 20 < 3)}
            for i, r in enumerate(reps)]
    decision = None
    if name == "chat-auto":
        decision = {"ts": time.time() - 42, "action": "scale_up", "reason": "avg busy 0.82 > target 0.70 for 30 s"}
    elif name == "chat-demand":
        decision = {"ts": time.time() - 600, "action": "unload", "reason": "no requests for 600 s"}
    return {"model": name, "min": sc["min"], "max": sc["max"], "desired": sc["desired"], "ready": len(reps),
            "launching": 1 if m["state"] == "starting" else 0, "avg_busy": avg, "queued": 0,
            "idle_s": 0.0 if reps else (4200.0 if m["state"] == "idle" else None), "state": state,
            "last_decision": decision, "replicas": rows}


@app.post("/api/models/{name}/start", dependencies=[api])
def start_model(name: str, body: dict | None = None) -> dict:
    m = _model(name)
    m["spec"]["replicas"] = int((body or {}).get("replicas", 1))
    m["state"], m["error"], m["_t"], m["_manual"] = "starting", None, time.time(), True
    return m["spec"]


@app.post("/api/models/{name}/stop", dependencies=[api])
def stop_model(name: str) -> dict:
    m = _model(name)
    m["spec"]["replicas"] = 0
    m["state"], m["_t"], m["_manual"] = "stopping", time.time(), False
    return m["spec"]


@app.delete("/api/models/{name}", dependencies=[api])
def del_model(name: str) -> dict:
    _model(name)
    del MODELS[name]
    return {"ok": True}


@app.post("/api/models/{name}/plan", dependencies=[api])
def plan_model(name: str) -> dict:
    m = _model(name)
    if m["spec"]["ctx_size"] > 32768:
        raise HTTPException(409, "Not enough free VRAM: needs 41.0 GB, pool has 120.5 GB usable but no single placement fits")
    pins = m["spec"]["pin_devices"]
    chosen = _pin_devices(pins)[:3] or [("CTG-Server-1", "CUDA0"), ("CTG-Server-2", "CUDA0")]
    return _placement(name, chosen)


def _need_mb(file_bytes: int, ctx: int, parallel: int, kv: str = "f16") -> int:
    return int(file_bytes / 1048576 * 1.1 + ctx * parallel * 3.0 * KV_FACTOR[kv] + 300)


@app.post("/api/recommend", dependencies=[api])
def recommend(body: dict) -> dict:
    item = LIBRARY.get(body.get("file"))
    if item is None:
        raise HTTPException(404, f"{body.get('file')} is not in the library")
    ctx = _int_in(body.get("ctx_size", 4096), 256, 10_000_000, "ctx_size")
    parallel = _int_in(body.get("parallel", 1), 1, 64, "parallel")
    limit = max(1, min(10, int(body.get("limit", 3))))
    perf = _perf_fields(body, body["file"])
    kv_f = KV_FACTOR[perf["kv_cache_type"]]
    draft_mb = _draft_mb(perf["draft"])
    pins = _check_pins(body.get("pin_devices") or [])
    alive = [s for s in SERVERS.values() if s["alive"]]
    gpus = [(s["node_id"], g) for s in alive for g in s["gpus"]
            if s["gpu_enabled"].get(g["device_id"], True) and _pinned(pins, s["node_id"], g["device_id"])]
    need = _need_mb(item["bytes"], ctx, parallel, perf["kv_cache_type"]) + (draft_mb or 0)
    fixed = _need_mb(item["bytes"], 0, 1) + (draft_mb or 0)
    biggest_gpu = max((g["usable_mb"] for _, g in gpus), default=0)
    per_node: dict[str, int] = {}
    for n, g in gpus:
        per_node[n] = per_node.get(n, 0) + g["usable_mb"]
    biggest_node = max(per_node.values(), default=0)
    max_ctx_single = max(0, int((biggest_gpu - fixed) / (3.0 * kv_f * parallel)) // 256 * 256)
    out = {"need_mb": need, "options": [], "max_ctx_single_gpu": max_ctx_single or None, "not_possible": None,
           "tips": _mock_tips(body, perf, ctx, parallel, need, biggest_gpu)}
    if need > biggest_node:
        fits = max(0, int((biggest_node - fixed) / (3.0 * kv_f * parallel)) // 256 * 256)
        out["not_possible"] = {"need_mb": need, "largest_single_gpu_mb": biggest_gpu, "largest_single_node_mb": biggest_node,
                               "max_ctx_that_fits": fits or None}
        return out
    ranked = sorted(gpus, key=lambda t: -(t[1]["bandwidth_gbps"] or 0))
    singles = [(n, g) for n, g in ranked if g["usable_mb"] >= need]
    opts = []
    for n, g in singles[:limit]:
        tps = round((g["bandwidth_gbps"] or 100) * 1024 / max(need, 1) * 0.5, 1)
        opts.append({"score": round(60 + tps / 10, 1), "tier": "single_gpu", "fits_now": True,
                     "assignments": [{"node_id": n, "device_id": g["device_id"], "layers": 33, "est_mb": need}],
                     "est_decode_tps": tps, "est_total_mb": need, "draft_est_mb": draft_mb,
                     "reasons": [f"{g['name']}: {g['usable_mb'] // 1024} GB free", f"about {round(g['bandwidth_gbps'] or 0)} GB/s memory bandwidth"]})
    if len(opts) < limit:
        multi = [(n, g) for n, g in ranked if (n, g) not in singles][:2]
        if len(multi) >= 2:
            half = need // 2
            tps = round(sum(g["bandwidth_gbps"] or 100 for _, g in multi) / 2 * 1024 / max(need, 1) * 0.4, 1)
            opts.append({"score": round(40 + tps / 10, 1), "tier": "multi_gpu" if multi[0][0] == multi[1][0] else "multi_node",
                         "fits_now": sum(g["usable_mb"] for _, g in multi) >= need,
                         "assignments": [{"node_id": n, "device_id": g["device_id"], "layers": 17, "est_mb": half} for n, g in multi],
                         "est_decode_tps": tps, "est_total_mb": need,
                         "reasons": ["needs more than one GPU", "split layers evenly across the pair"]})
    if _int_in(body.get("priority", 50), 0, 100, "priority") >= 80 and ranked:
        victim = _victim(100)
        if victim:
            n, g = ranked[0]
            opts.append({"score": 55.0, "tier": "single_gpu", "fits_now": False,
                         "assignments": [{"node_id": n, "device_id": g["device_id"], "layers": 33, "est_mb": need}],
                         "est_decode_tps": round((g["bandwidth_gbps"] or 100) * 1024 / max(need, 1) * 0.5, 1), "est_total_mb": need,
                         "requires_preemption": [{"replica_id": victim[1]["replica_id"], "model": victim[0]["spec"]["name"],
                                                  "priority": victim[0]["spec"]["priority"]}],
                         "reasons": ["the fastest GPU is full; a lower-priority replica would have to stop"]})
    for i, o in enumerate(sorted(opts, key=lambda o: -o["score"]), 1):
        out["options"].append({"rank": i, **o})
    return out


def _victim(priority: int, skip: str | None = None):
    """First replica of a running, preemptible model with a lower priority than `priority`."""
    for m in MODELS.values():
        sp = m["spec"]
        if m["replicas"] and sp["name"] != skip and sp.get("preemptible", True) and sp["priority"] < priority:
            return m, m["replicas"][0]
    return None


def _sim_start(model: str, n: int, ctx: int, parallel: int, size_mb: int) -> list[dict]:
    gpus = sorted(((s["node_id"], g) for s in SERVERS.values() if s["alive"] for g in s["gpus"]
                   if s["gpu_enabled"].get(g["device_id"], True)), key=lambda t: -(t[1]["bandwidth_gbps"] or 0))
    out = []
    for i in range(min(n, len(gpus))):
        node, g = gpus[i]
        out.append({"model": model, "tier": "single_gpu", "est_decode_tps": round((g["bandwidth_gbps"] or 100) * 1024 / max(size_mb, 1) * 0.5, 1),
                    "assignments": [{"node_id": node, "device_id": g["device_id"], "layers": 33, "est_mb": size_mb}]})
    return out


@app.post("/api/simulate", dependencies=[api])
def simulate(body: dict) -> dict:
    """Pure dry run with canned rules (see the module docstring); changes nothing."""
    out: dict = {"start": [], "stop": [], "preempt": [], "unplaced": []}
    biggest_node = max((sum(g["usable_mb"] for g in s["gpus"]) for s in SERVERS.values() if s["alive"]), default=0)
    items = [(c.get("model"), c, MODELS.get(c.get("model"))) for c in body.get("changes") or []]
    items += [(a.get("name"), a, None) for a in body.get("add") or []]
    if not items:
        raise HTTPException(422, "changes or add required")
    for name, req, cur in items:
        if cur is None and "changes" in body and any(c is req for c in body["changes"]):
            raise HTTPException(404, f"unknown model {name}")
        file = req.get("file") or (cur or {}).get("file")
        if file not in LIBRARY:
            raise HTTPException(404, f"{file} is not in the library")
        sp = (cur or {}).get("spec", {})
        ctx = _int_in(req.get("ctx_size", sp.get("ctx_size", 4096)), 256, 10_000_000, "ctx_size")
        parallel = _int_in(req.get("parallel", sp.get("parallel", 1)), 1, 64, "parallel")
        prio = _int_in(req.get("priority", sp.get("priority", 50)), 0, 100, "priority")
        want = int(req.get("replicas") or req.get("min_replicas") or sp.get("replicas") or 1)
        running = len((cur or {}).get("replicas", []))
        perf = _perf_fields({**{k: v for k, v in sp.items() if k in PERF_DEFAULTS and k != "draft"},
                             **({"draft_file": str(sp["draft"]).replace("coordinator://", "")} if sp.get("draft") else {}), **req}, file)
        need = _need_mb(LIBRARY[file]["bytes"], ctx, parallel, perf["kv_cache_type"]) + (_draft_mb(perf["draft"]) or 0)
        if need > biggest_node:
            out["unplaced"].append({"model": name, "missing": max(1, want - running),
                                    "why": f"needs about {need // 1024} GB on one server; the largest has {biggest_node // 1024} GB usable"})
            continue
        extra = want - running
        if extra < 0 and req.get("replicas") is None:  # autoscaled models are not shrunk by a new minimum
            continue
        if extra < 0:
            out["stop"] += [{"replica_id": r["replica_id"], "model": name, "reason": "replicas lowered"}
                            for r in cur["replicas"][want:]]
            continue
        if cur is not None and extra == 0 and prio == sp.get("priority") and ctx == sp.get("ctx_size") and parallel == sp.get("parallel"):
            continue
        extra = max(extra, 1) if cur is None or running == 0 else extra
        if extra <= 0:
            continue
        victim = _victim(prio, skip=name) if (prio > 50 or want >= 3) else None
        if victim:
            m, r = victim
            out["preempt"].append({"replica_id": r["replica_id"], "model": m["spec"]["name"], "priority": m["spec"]["priority"], "for_model": name})
            if prio >= 90 and want >= 4 and m["spec"].get("min_replicas") is not None and len(m["replicas"]) > 1:
                out["stop"].append({"replica_id": m["replicas"][1]["replica_id"], "model": m["spec"]["name"],
                                    "reason": "autoscaled model shrinks to its minimum to free GPUs"})
        starts = _sim_start(name, extra, ctx, parallel, need)
        out["start"] += starts
        if len(starts) < extra:
            out["unplaced"].append({"model": name, "missing": extra - len(starts), "why": "no more GPUs with room, even after preemption"})
        elif prio >= 90 and want >= 4:
            out["unplaced"].append({"model": name, "missing": 1, "why": "spread=gpu needs a different GPU per replica and only the remaining ones are protected"})
    return out


@app.get("/api/capacity", dependencies=[api])
def capacity() -> dict:
    tick()
    rows = []
    for s in SERVERS.values():
        for g in s["gpus"]:
            reps = [{"replica_id": r["replica_id"], "model": m["spec"]["name"], "est_mb": a["est_mb"], "busy": 0.0}
                    for m in MODELS.values() for r in m["replicas"] for a in r["placement"]["assignments"]
                    if a["node_id"] == s["node_id"] and a["device_id"] == g["device_id"]]
            reserved = sum(r["est_mb"] for r in reps)
            on = s["gpu_enabled"].get(g["device_id"], True)
            rows.append({"node_id": s["node_id"], "device_id": g["device_id"], "uuid": f"GPU-mock-{s['node_id']}-{g['device_id']}",
                         "name": g["name"], "kind": "cuda", "enabled": on, "alive": s["alive"], "total_mb": g["total_mb"],
                         "usable_mb": g["usable_mb"], "free_for_new_mb": max(0, g["usable_mb"] - reserved) if on and s["alive"] else 0,
                         "reserved_mb": reserved, "bandwidth_gbps": g["bandwidth_gbps"], "busy": round(g["util_pct"] / 100, 2),
                         "replicas": reps})
    usable = [r for r in rows if r["enabled"] and r["alive"]]
    per_node: dict[str, int] = {}
    for r in usable:
        per_node[r["node_id"]] = per_node.get(r["node_id"], 0) + r["free_for_new_mb"]
    return {"gpus": rows, "summary": {"gpus": len(rows), "free_for_new_mb": sum(r["free_for_new_mb"] for r in usable),
                                      "largest_single_gpu_mb": max((r["free_for_new_mb"] for r in usable), default=0),
                                      "largest_single_node_mb": max(per_node.values(), default=0)}}


@app.get("/api/events", dependencies=[api])
def list_events(limit: int = 200, after_id: int = 0) -> list[dict]:
    return [e for e in EVENTS if e["id"] > after_id][:limit]


@app.post("/api/events/read", dependencies=[api])
def read_events(body: dict) -> dict:
    up = int(body.get("up_to_id", 0))
    for e in EVENTS:
        if e["id"] <= up:
            e["read"] = True
    return {"unread": sum(1 for e in EVENTS if not e["read"])}


@app.post("/api/rebalance", dependencies=[api])
def rebalance(body: dict) -> dict:
    tick()
    dry = _bool(body.get("dry_run", True), "dry_run")
    moves = _rebalance_moves()
    started = None
    if not dry and moves:
        mv = moves[0]
        m = MODELS[mv["model"]]
        now = time.time()
        new_id = f"{mv['model']}-{int(now) % 100000:05d}"
        m["replicas"].append({"replica_id": new_id, "model": mv["model"], "state": "starting", "error": None, "outstanding": 0,
                              "created_at": now, "updated_at": now,
                              "placement": _placement(mv["model"], [(t["node_id"], t["device_id"]) for t in mv["to"]])})
        REBAL["in_progress"] = {"model": mv["model"], "old": mv["replica_id"], "new": new_id, "since": now}
        started = {"replica_id": mv["replica_id"], "model": mv["model"]}
        add_event("info", "rebalance_started", f"Moving {mv['replica_id']} of {mv['model']}: starting {new_id} on better GPUs", None, mv["model"])
        SCHEDULED.append((now + REBAL_SECONDS, lambda: _finish_rebalance(mv["model"], mv["replica_id"], new_id)))
    ip = REBAL["in_progress"]
    return {"moves": moves, "started": started, "in_progress": dict(ip) if ip else None}


@app.post("/api/_mock/kill/{node_id}")
def mock_kill(node_id: str) -> dict:
    kill(node_id)
    return {"ok": True}


# ---------------------------------------------------------------------------------------------------------------------
# Model conversion (Hugging Face / folder -> GGUF): mirrors the routes and JSON shapes of the real coordinator API.
# ---------------------------------------------------------------------------------------------------------------------
# (type, bits per weight, tier, via, quality note, needs_imatrix), best quality first. Same order as the real option list.
QUANTS = [
    ("F16", 16.0, "lossless", "convert", "16-bit floats, no quantization loss (~14.0 GB for a 7B model)", False),
    ("BF16", 16.0, "lossless", "convert", "16-bit brain floats, same size as F16 and a wider value range", False),
    ("Q8_0", 8.5, "near_lossless", "convert", "+0.0026 ppl @ Llama-3-8B, practically indistinguishable from F16", False),
    ("Q6_K", 6.5625, "near_lossless", "quantize", "+0.0217 ppl @ Llama-3-8B", False),
    ("Q5_K_M", 5.69, "balanced", "quantize", "+0.0569 ppl @ Llama-3-8B, very good quality", False),
    ("Q5_K_S", 5.54, "balanced", "quantize", "+0.1049 ppl @ Llama-3-8B", False),
    ("Q4_K_M", 4.89, "balanced", "quantize", "+0.1754 ppl @ Llama-3-8B, the usual sweet spot of size and quality", False),
    ("Q4_K_S", 4.58, "small", "quantize", "+0.2689 ppl @ Llama-3-8B", False),
    ("IQ4_XS", 4.25, "small", "quantize", "4.25 bits per weight, close to Q4_K_S at a smaller size", False),
    ("Q4_0", 4.55, "small", "quantize", "+0.4685 ppl @ Llama-3-8B, legacy format", False),
    ("Q3_K_L", 4.3, "small", "quantize", "+0.5562 ppl @ Llama-3-8B, noticeable quality loss", False),
    ("Q3_K_M", 3.91, "tiny", "quantize", "+0.6569 ppl @ Llama-3-8B", False),
    ("IQ3_M", 3.66, "tiny", "quantize", "3.66 bits per weight, better than Q3_K_M at a similar size", False),
    ("IQ3_S", 3.44, "tiny", "quantize", "3.44 bits per weight, better than Q3_K_S at a similar size", False),
    ("Q3_K_S", 3.5, "tiny", "quantize", "+1.6321 ppl @ Llama-3-8B, clear quality loss", False),
    ("IQ3_XS", 3.3, "tiny", "quantize", "3.3 bits per weight, needs an importance matrix", True),
    ("IQ3_XXS", 3.06, "tiny", "quantize", "3.06 bits per weight, needs an importance matrix", True),
    ("Q2_K", 3.35, "tiny", "quantize", "+3.5199 ppl @ Llama-3-8B, last resort", False),
    ("IQ2_M", 2.7, "tiny", "quantize", "2.7 bits per weight, strong quality loss, needs an importance matrix", True),
    ("IQ2_S", 2.5, "tiny", "quantize", "2.5 bits per weight, strong quality loss, needs an importance matrix", True),
    ("IQ2_XS", 2.31, "tiny", "quantize", "2.31 bits per weight, strong quality loss, needs an importance matrix", True),
    ("IQ2_XXS", 2.06, "tiny", "quantize", "2.06 bits per weight, severe quality loss, needs an importance matrix", True),
    ("IQ1_M", 1.75, "tiny", "quantize", "1.75 bits per weight, severe quality loss, needs an importance matrix", True),
    ("IQ1_S", 1.56, "tiny", "quantize", "1.56 bits per weight, extreme quality loss, needs an importance matrix", True),
]
QUANT_BY_TYPE = {q[0]: q for q in QUANTS}
ACTIVE_JOB_STATES = ("queued", "downloading", "converting", "calibrating", "quantizing", "validating")
CONVERT: dict = {"problem": None, "imatrix": True, "disk_free_gb": 500.0}  # imatrix: llama-imatrix installed; disk_free_gb: free space for the 507 check; a string makes the toolchain "missing": options.available false, POST /api/convert 503
JOBS: dict[str, dict] = {}
INSPECT_DELAY = [0.0]  # seconds a real run adds to inspect so the spinner is visible (set in __main__)
STAGE_SECONDS = {"queued": 1.5, "downloading": 6.0, "converting": 5.0, "calibrating": 6.0, "quantizing": 5.0, "validating": 3.0}

# Fake Hugging Face repos (all lower-cased keys). `files` are (name, bytes); the rest of the fields is what inspect reports.
_ST = "safetensors"
REPOS: dict[str, dict] = {
    "huggingfacetb/smollm2-135m-instruct": dict(
        repo="HuggingFaceTB/SmolLM2-135M-Instruct", arch="LlamaForCausalLM", model_type="llama", params=134_515_008, layers=30, ctx=8192,
        files=[("config.json", 861), ("tokenizer.json", 2_104_556), ("tokenizer_config.json", 3_764), ("model.safetensors", 269_060_552)],
        skipped=["README.md", ".gitattributes", "onnx/model.onnx"],
        alts=["bartowski/SmolLM2-135M-Instruct-GGUF", "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF"]),
    "qwen/qwen2.5-7b-instruct": dict(
        repo="Qwen/Qwen2.5-7B-Instruct", arch="Qwen2ForCausalLM", model_type="qwen2", params=7_615_616_512, layers=28, ctx=32768,
        files=[("config.json", 663), ("tokenizer.json", 7_031_645), ("tokenizer_config.json", 7_305), ("vocab.json", 2_776_833),
               ("merges.txt", 1_671_839)] + [(f"model-0000{i}-of-00004.safetensors", b) for i, b in
                                              enumerate((3_945_000_000, 3_864_000_000, 3_864_000_000, 3_563_000_000), 1)],
        skipped=["README.md", "LICENSE", ".gitattributes"], alts=["Qwen/Qwen2.5-7B-Instruct-GGUF", "bartowski/Qwen2.5-7B-Instruct-GGUF"]),
    "qwen/qwen2.5-72b-instruct": dict(
        repo="Qwen/Qwen2.5-72B-Instruct", arch="Qwen2ForCausalLM", model_type="qwen2", params=72_706_203_648, layers=80, ctx=32768,
        files=[("config.json", 664), ("tokenizer.json", 7_031_645)] + [(f"model-{i:05d}-of-00037.safetensors", 3_900_000_000) for i in range(1, 38)],
        skipped=["README.md", "LICENSE"], alts=["Qwen/Qwen2.5-72B-Instruct-GGUF"]),
    "qwen/qwen2.5-7b-instruct-awq": dict(
        repo="Qwen/Qwen2.5-7B-Instruct-AWQ", arch="Qwen2ForCausalLM", model_type="qwen2", params=7_615_616_512, layers=28, ctx=32768,
        files=[("config.json", 1_196), ("tokenizer.json", 7_031_645), ("model-00001-of-00002.safetensors", 3_990_000_000),
               ("model-00002-of-00002.safetensors", 1_590_000_000)],
        skipped=["README.md"], prequantized="awq", prequant_supported=False, base="Qwen/Qwen2.5-7B-Instruct",
        alts=["Qwen/Qwen2.5-7B-Instruct-GGUF"],
        warnings=["Quant method awq is not yet supported by the converter: the weights are already 4-bit AWQ and cannot be turned back into floats."]),
    "acme/novelnet-7b": dict(
        repo="acme/NovelNet-7B", arch="NovelNetForCausalLM", model_type="novelnet", params=7_000_000_000, layers=32, ctx=4096,
        files=[("config.json", 700), ("tokenizer.model", 500_000), ("model.safetensors", 14_000_000_000)], skipped=[], supported=False,
        warnings=["Architecture NovelNetForCausalLM is not known to the pinned llama.cpp converter (b11413)."]),
    "meta-llama/llama-3.1-8b-instruct": dict(
        repo="meta-llama/Llama-3.1-8B-Instruct", arch="LlamaForCausalLM", model_type="llama", params=8_030_261_248, layers=32, ctx=131072,
        files=[("config.json", 855), ("tokenizer.json", 9_085_657)] + [(f"model-0000{i}-of-00004.safetensors", b) for i, b in
                                                                       enumerate((4_976_698_672, 4_999_802_720, 4_915_916_176, 1_168_138_808), 1)],
        skipped=["README.md", "LICENSE"], gated=True, alts=["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"],
        warnings=["Gated repository: the download needs a Hugging Face token that has been granted access (HF_TOKEN)."]),
    "acme/huge-400b": dict(
        repo="acme/Huge-400B", arch="LlamaForCausalLM", model_type="llama", params=405_000_000_000, layers=126, ctx=131072,
        files=[("config.json", 900), ("tokenizer.json", 17_000_000)] + [(f"model-{i:05d}-of-00191.safetensors", 4_250_000_000) for i in range(1, 192)],
        skipped=["README.md"], warnings=["This is a very large model: converting it needs several hundred GB of free disk."]),
    "acme/quirky-3b": dict(
        repo="acme/Quirky-3B", arch="QuirkyForCausalLM", model_type="llama", params=3_212_749_824, layers=28, ctx=8192,
        files=[("config.json", 900), ("tokenizer_config.json", 4_000), ("model.safetensors", 6_425_499_648)], skipped=["tokenization_quirky.py", "modeling_quirky.py"],
        remote_code=True, warnings=["The repository ships custom Python code (tokenization_quirky.py). It is skipped unless you allow remote code."]),
}
# Fake server folders: path -> same fields as a repo (no gated / alts).
FOLDERS: dict[str, dict] = {
    "/models/hf/smollm2-135m": {**REPOS["huggingfacetb/smollm2-135m-instruct"], "alts": [], "skipped": ["README.md"]},
    "/models/hf/broken-model": dict(arch="LlamaForCausalLM", model_type="llama", params=1_100_000_000, layers=22, ctx=2048,
                                    files=[("config.json", 700), ("pytorch_model.bin", 2_200_000_000)], skipped=[], fmt="pytorch_bin"),
}


def _cluster() -> dict:
    """Stable usable-VRAM numbers of the live pool (alive + enabled GPUs, 2 GB reserved each) for fit checks."""
    mbs = [max(0, g["total_mb"] - 2048) for s in SERVERS.values() if s["alive"] for g in s["gpus"] if s["gpu_enabled"].get(g["device_id"], True)]
    return {"largest_gpu_mb": max(mbs, default=0), "pool_mb": sum(mbs)}


def _options(params: int, layers: int, cluster: dict) -> list[dict]:
    out = []
    for t, bpw, tier, via, note, need in QUANTS:
        est = int(params * bpw / 8 * 1.01)
        vram = int(est / 1048576 + layers * 8.4 + 500)  # file + KV cache at ctx 4096 + runtime overhead
        out.append({"type": t, "bpw": bpw, "tier": tier, "note": note, "via": via, "needs_imatrix": need, "est_bytes": est, "est_vram_mb": vram,
                    "fits_single_gpu": vram <= cluster["largest_gpu_mb"], "fits_pool": vram <= cluster["pool_mb"], "recommended": False})
    return out


def _recommend(opts: list[dict], cluster: dict) -> tuple[str, list[str]]:
    """Best non-lossless type that leaves 30 % headroom on the largest GPU, else the best one that fits at all."""
    big = cluster["largest_gpu_mb"]
    lossy = [o for o in opts if o["type"] not in ("F16", "BF16") and o["bpw"] >= 3.3 and not o["needs_imatrix"]]
    roomy = next((o for o in lossy if o["est_vram_mb"] <= big * 0.7), None)
    if roomy:
        return roomy["type"], [f"Best quality that leaves 30% of the largest GPU ({big // 1024} GB) free for context and batching",
                               f"{roomy['type']} needs about {roomy['est_vram_mb'] / 1024:.1f} GB of VRAM",
                               "F16/BF16 are skipped: twice the size of Q8_0 for no audible difference"]
    single = next((o for o in lossy if o["fits_single_gpu"]), None)
    if single:
        return single["type"], [f"The largest type that still fits one GPU ({big // 1024} GB)", "Little headroom is left for a long context"]
    pool = next((o for o in lossy if o["fits_pool"]), None)
    if pool:
        return pool["type"], ["No single GPU can hold it: this is the best type that fits the whole pool",
                              "It will run split over several GPUs (slower, over the network)"]
    return "Q2_K", ["Nothing fits the cluster at the moment; Q2_K is the smallest type that needs no importance matrix"]


def _translate_folder(p: str) -> str:
    return MODEL_ROOT + p[len(HOST_ROOT):] if p == HOST_ROOT or p.startswith(HOST_ROOT + "/") else p


def _inspect(spec: dict) -> dict:
    """Mock of POST /api/convert/inspect for a SourceSpec dict."""
    repo, path = spec.get("hf_repo"), spec.get("path")
    if (repo is None) == (path is None):
        raise HTTPException(422, "give exactly one of hf_repo or path")
    rev = spec.get("revision") or "main"
    if repo is not None:
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise HTTPException(400, f"{repo!r} is not a Hugging Face repository id: use owner/name")
        d = REPOS.get(repo.lower())
        if d is None:
            raise HTTPException(404, f"Repository {repo} was not found on Hugging Face (or it is private: set HF_TOKEN)")
        source = {"hf_repo": d["repo"], "revision": rev, "path": None}
    else:
        if not path.startswith("/"):
            raise HTTPException(400, "path must be absolute")
        p = _translate_folder(path).rstrip("/")
        d = FOLDERS.get(p)
        if d is None:
            raise HTTPException(400, f"No such folder inside the coordinator: {p}. The coordinator runs in Docker and only sees its mounted "
                                     f"folders: {MODEL_ROOT} (host folder {HOST_ROOT}). Try {MODEL_ROOT}/hf/smollm2-135m.")
        source = {"hf_repo": None, "revision": "main", "path": p}
    if INSPECT_DELAY[0]:
        time.sleep(INSPECT_DELAY[0])
    cl = _cluster()
    opts = _options(d["params"], d["layers"], cl)
    rec, reasons = _recommend(opts, cl)
    next(o for o in opts if o["type"] == rec)["recommended"] = True
    files = [{"name": n, "bytes": b} for n, b in d["files"]]
    return {"source": source, "architecture": d["arch"], "model_type": d["model_type"], "supported": d.get("supported", True), "params": d["params"],
            "n_layers": d["layers"], "context_length": d["ctx"], "weight_format": d.get("fmt", _ST), "prequantized": d.get("prequantized"),
            "prequant_supported": d.get("prequant_supported") if d.get("prequantized") else None, "source_bytes": sum(f["bytes"] for f in files),
            "files": files, "skipped": list(d.get("skipped", [])), "remote_code": bool(d.get("remote_code")), "gated": bool(d.get("gated")),
            "base_model": d.get("base"), "gguf_alternatives": list(d.get("alts", [])), "options": opts, "recommended": rec,
            "name_stem": _model_base(source), "recommend_reasons": reasons, "warnings": list(d.get("warnings", []))}


def _model_base(spec: dict) -> str:
    src = (spec.get("hf_repo") or spec.get("path") or "model").rstrip("/")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", src.split("/")[-1]).lstrip(".-") or "model"


def _check_output_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.gguf", name) or re.search(r"-\d{5}-of-\d{5}\.gguf$", name):
        raise HTTPException(400, f"{name!r} is not a valid output name: use letters, digits, dot, dash and underscore, ending in .gguf "
                                 "(not a split part such as -00001-of-00002.gguf)")
    return name


def _normalize_request(body: dict) -> dict:
    """Validate a ConvertRequest body like pydantic would (422) and fill the defaults."""
    src = body.get("source")
    if not isinstance(src, dict):
        raise HTTPException(422, "source: field required")
    quant = body.get("quant", "Q4_K_M")
    if quant not in QUANT_BY_TYPE:
        raise HTTPException(422, f"quant: must be one of {', '.join(QUANT_BY_TYPE)}")
    adv = body.get("advanced") or {}
    inter = adv.get("intermediate", "auto")
    if inter not in ("auto", "f16", "bf16", "f32"):
        raise HTTPException(422, "advanced.intermediate: must be one of auto, f16, bf16, f32")
    threads = adv.get("threads", 0)
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 0:
        raise HTTPException(422, "advanced.threads: must be an integer >= 0")
    for k in ("leave_output_tensor", "pure", "allow_remote_code", "validate_generation"):
        if k in adv and not isinstance(adv[k], bool):
            raise HTTPException(422, f"advanced.{k}: must be a boolean")
    imatrix = adv.get("imatrix", "auto")
    if imatrix not in ("auto", "on", "off"):
        raise HTTPException(422, "advanced.imatrix: must be one of auto, on, off")
    chunks = adv.get("imatrix_chunks", 0)
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 0:
        raise HTTPException(422, "advanced.imatrix_chunks: must be an integer >= 0")
    calib = adv.get("calibration_path") or None
    if calib is not None and (not isinstance(calib, str) or not calib.startswith("/")):
        raise HTTPException(422, "advanced.calibration_path: must be an absolute path on the server")
    if calib is not None and not calib.lower().endswith(".txt"):
        raise HTTPException(422, "advanced.calibration_path: the calibration text must be a .txt file")
    advanced = {"intermediate": inter, "imatrix": imatrix, "calibration_path": calib, "imatrix_chunks": chunks, "output_tensor_type": adv.get("output_tensor_type") or None,
                "token_embedding_type": adv.get("token_embedding_type") or None,
                "leave_output_tensor": bool(adv.get("leave_output_tensor", False)), "pure": bool(adv.get("pure", False)),
                "allow_remote_code": bool(adv.get("allow_remote_code", False)),
                "validate_generation": bool(adv.get("validate_generation", True)), "threads": threads}
    source = {"hf_repo": src.get("hf_repo"), "revision": src.get("revision") or "main", "path": src.get("path")}
    return {"source": source, "quant": quant, "name": body.get("name") or None, "keep_source": bool(body.get("keep_source", False)), "advanced": advanced}


def _validation(ok: bool, generation: bool) -> dict:
    cases = [("Hello, world!", [9906, 11, 1917, 0]), ("The quick brown fox", [791, 4062, 14198, 39935]),
             ("def f(x):\n    return x  # tab\there", [755, 282, 2120, 1680, 220, 220, 220, 471, 865, 220, 674, 1587, 3984, 1618]),
             ("  leading spaces and 日本語", [220, 6522, 12908, 323, 220, 101, 102, 103])]
    out = [{"text": t, "hf": ids, "gguf": list(ids), "match": True} for t, ids in cases]
    warnings: list[str] = []
    if not ok:
        out[2]["gguf"] = out[2]["gguf"][:4] + [256] + out[2]["gguf"][4:]  # one extra token: whitespace handling differs
        out[2]["match"] = False
        warnings.append("The GGUF tokenizer split 1 of 4 test strings differently from the original; whitespace handling may differ.")
    return {"header_ok": True, "architecture": "llama", "n_layers": 30, "vocab_size": 49152, "chat_template": True, "tokenizer_ok": ok,
            "tokenizer_cases": out, "generation_ok": True if generation else None,
            "generation_sample": "The capital of France is Paris. It is known for the Eiffel Tower and its museums." if generation else None,
            "warnings": warnings, "errors": []}


def _conv_fail_point(j: dict) -> tuple[str, float, str] | None:
    """(stage, fraction at which it fails, error) for the jobs that are meant to fail, else None."""
    s, adv = j["request"]["source"], j["request"]["advanced"]
    repo = (s.get("hf_repo") or "").lower()
    if repo == "meta-llama/llama-3.1-8b-instruct":
        return "downloading", 0.0, ("Hugging Face answered 403 for meta-llama/Llama-3.1-8B-Instruct: the repository is gated or private. "
                                    "Set HF_TOKEN on the coordinator to a token that has been granted access.")
    if repo == "acme/quirky-3b" and not adv["allow_remote_code"]:
        return "converting", 0.4, ("convert_hf_to_gguf.py exited with code 1: ValueError: Loading acme/Quirky-3B requires executing custom code "
                                  "(tokenization_quirky.py). Enable 'Allow remote code' if you trust the repository.")
    if adv["calibration_path"] and adv["calibration_path"].lower().endswith("corrupt.txt") and j["imatrix_used"]:
        return "calibrating", 0.5, ("llama-imatrix exited with code 1: the calibration text is not valid UTF-8 "
                                   f"({adv['calibration_path']})")
    if s.get("path") == "/models/hf/broken-model":
        return "converting", 0.6, ("convert_hf_to_gguf.py exited with code 1: KeyError: 'rope_theta' while reading config.json "
                                  "(the folder looks truncated)")
    return None


def _conv_outcome(j: dict) -> str:
    return "needs_review" if (j["request"]["source"].get("hf_repo") or "").lower() == "qwen/qwen2.5-7b-instruct" else "done"


def _conv_logs(state: str, frac: float, j: dict) -> list[str]:
    n = max(1, int(frac * 339))
    if state == "downloading":
        return [f"GET {f['name']}" for f in _inspect_files(j)[: 1 + int(frac * 3)]] + [f"{int(frac * 100):3d}% of {j['bytes_total']} bytes"]
    if state == "converting":
        return ["INFO:hf-to-gguf:Loading model: " + (j["request"]["source"].get("hf_repo") or j["request"]["source"].get("path")), "INFO:hf-to-gguf:gguf: loading model weight map",
                f"INFO:hf-to-gguf:blk.{int(frac * 27)}.attn_q.weight, torch.bfloat16 --> F16, shape = {{3584, 3584}}",
                f"Writing: {int(frac * 100):3d}%|{'#' * int(frac * 20):<20}| {int(frac * 15):d}.2G/15.2G"]
    if state == "calibrating":
        return ["compute_imatrix: tokenizing the input ..", "compute_imatrix: tokenization took 41.2 ms",
                f"compute_imatrix: computing over {j['request']['advanced']['imatrix_chunks'] or 100} chunks, n_ctx=512, batch_size=512",
                f"[{max(1, int(frac * 100))}]5.8142,{int(frac * 100)} chunks processed"]
    if state == "quantizing":
        return ["main: build = 11342", f"[{n:4d}/ 339] blk.{int(frac * 27)}.ffn_up.weight - [3584, 18944, 1, 1], type = f16, converting to q4_K .. size = 129.50 MiB -> 36.51 MiB"]
    return ["tokenizer check: 4 test strings", "generation check: 8 tokens on the CPU"]


def _inspect_files(j: dict) -> list[dict]:
    d = REPOS.get((j["request"]["source"].get("hf_repo") or "").lower()) or {"files": []}
    return [{"name": n, "bytes": b} for n, b in d["files"]]


def _conv_stages(j: dict) -> list[tuple[str, float]]:
    out = [("queued", STAGE_SECONDS["queued"])]
    if j["request"]["source"].get("hf_repo"):
        out.append(("downloading", STAGE_SECONDS["downloading"]))
    out.append(("converting", STAGE_SECONDS["converting"]))
    if j["imatrix_used"]:
        out.append(("calibrating", STAGE_SECONDS["calibrating"]))
    if QUANT_BY_TYPE[j["request"]["quant"]][3] == "quantize":
        out.append(("quantizing", STAGE_SECONDS["quantizing"]))
    out.append(("validating", STAGE_SECONDS["validating"]))
    return out


def _conv_finish(j: dict, state: str, error: str | None = None) -> None:
    if state == "failed":
        j["failed_stage"] = j["state"]
    j["state"], j["error"], j["finished_at"] = state, error, time.time()
    if state == "failed":
        return
    j["stage_progress"] = 1.0
    j["output_bytes"] = int(j["est_output_bytes"] * 0.995)
    adv = j["request"]["advanced"]
    j["validation"] = _validation(state == "done", adv["validate_generation"])
    if state == "done":
        _conv_to_library(j)


def _conv_to_library(j: dict) -> None:
    s = j["request"]["source"]
    LIBRARY[j["output_name"]] = {"name": j["output_name"], "path": f"/data/models/{j['output_name']}", "source": "convert",
                                 "hf_repo": s.get("hf_repo"), "hf_file": None, "bytes": j["output_bytes"], "downloaded": j["output_bytes"],
                                 "status": "ready", "error": None, "created_at": time.time()}


def _conv_advance(j: dict) -> None:
    """Move an active job along its wall-clock timeline (queued, download, convert, quantize, validate, outcome)."""
    if j["state"] not in ACTIVE_JOB_STATES:
        return
    el, t, fail = time.time() - j["_t0"], 0.0, _conv_fail_point(j)
    for state, dur in _conv_stages(j):
        frac = min(1.0, max(0.0, (el - t) / dur))
        if fail and fail[0] == state and el >= t + dur * fail[1]:
            j["state"], j["stage_progress"] = state, fail[1]
            j["started_at"] = j["started_at"] or j["_t0"] + STAGE_SECONDS["queued"]
            j["log_tail"] = _conv_logs(state, fail[1], j) + ["ERROR: " + fail[2].split(": ", 1)[-1]]
            _conv_finish(j, "failed", fail[2])
            return
        if el < t + dur:
            j["state"] = state
            if state != "queued":
                j["started_at"] = j["started_at"] or j["_t0"] + STAGE_SECONDS["queued"]
            j["stage_progress"] = None if state == "queued" else round(frac, 3)
            if state == "downloading":
                j["bytes_total"], j["bytes_done"] = j["_src_bytes"], int(frac * j["_src_bytes"])
            j["log_tail"] = [] if state == "queued" else _conv_logs(state, frac, j)
            return
        t += dur
    if j["request"]["source"].get("hf_repo"):
        j["bytes_done"] = j["bytes_total"] = j["_src_bytes"]
    j["log_tail"] = ["validation finished"]
    _conv_finish(j, _conv_outcome(j))


def _imatrix_used(req: dict) -> bool:
    """auto = on when the type needs an importance matrix or is under 4 bits; the direct types never run one."""
    q = QUANT_BY_TYPE[req["quant"]]
    mode = req["advanced"]["imatrix"]
    return q[3] == "quantize" and (mode == "on" or (mode == "auto" and (q[5] or q[1] < 4.0)))


def _conv_new(request: dict, name: str, insp: dict, created: float) -> dict:
    opt = next(o for o in insp["options"] if o["type"] == request["quant"])
    j = {"id": uuid.uuid4().hex[:12], "request": request, "state": "queued", "stage_progress": None, "bytes_done": 0, "bytes_total": None,
         "output_name": name, "failed_stage": None, "imatrix_used": _imatrix_used(request), "output_bytes": None, "est_output_bytes": opt["est_bytes"], "validation": None, "error": None, "log_tail": [],
         "created_at": created, "started_at": None, "finished_at": None, "_t0": created, "_src_bytes": insp["source_bytes"]}
    JOBS[j["id"]] = j
    return j


def _conv_json(j: dict) -> dict:
    _conv_advance(j)
    return _clean(j)


def _conv_job(job_id: str) -> dict:
    j = JOBS.get(job_id)
    if j is None:
        raise HTTPException(404, f"No conversion job {job_id}")
    _conv_advance(j)
    return j


def seed_conversions() -> None:
    """Three jobs of the demo: one finished (and in the library), one failed, one running that ends in needs_review."""
    JOBS.clear()
    CONVERT.update(problem=None, imatrix=True, disk_free_gb=500.0)
    done_req = _normalize_request({"source": {"hf_repo": "HuggingFaceTB/SmolLM2-135M-Instruct"}, "quant": "Q6_K",
                                   "name": "SmolLM2-135M-Instruct-Q6_K.gguf"})
    j = _conv_new(done_req, "SmolLM2-135M-Instruct-Q6_K.gguf", _inspect(done_req["source"]), T0 - 3600)
    j.update(started_at=T0 - 3598, bytes_done=j["_src_bytes"], bytes_total=j["_src_bytes"], state="done", stage_progress=1.0,
             output_bytes=int(j["est_output_bytes"] * 0.995), finished_at=T0 - 3540, log_tail=["validation finished"],
             validation=_validation(True, True))
    LIBRARY[j["output_name"]] = {"name": j["output_name"], "path": f"/data/models/{j['output_name']}", "source": "convert",
                                 "hf_repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "hf_file": None, "bytes": j["output_bytes"],
                                 "downloaded": j["output_bytes"], "status": "ready", "error": None, "created_at": T0 - 3540}
    bad_req = _normalize_request({"source": {"hf_repo": "acme/Quirky-3B"}, "quant": "Q4_K_M", "name": "Quirky-3B-Q4_K_M.gguf"})
    j = _conv_new(bad_req, "Quirky-3B-Q4_K_M.gguf", _inspect(bad_req["source"]), T0 - 1800)
    fp = _conv_fail_point(j)
    j.update(started_at=T0 - 1798, bytes_done=j["_src_bytes"], bytes_total=j["_src_bytes"], state="failed", failed_stage="converting", stage_progress=fp[1], error=fp[2],
             finished_at=T0 - 1780, log_tail=_conv_logs("converting", 0.4, j) + ["ERROR: " + fp[2].split(": ", 1)[-1]])
    run_req = _normalize_request({"source": {"hf_repo": "Qwen/Qwen2.5-7B-Instruct"}, "quant": "Q4_K_M", "name": "Qwen2.5-7B-Instruct-Q4_K_M.gguf"})
    _conv_new(run_req, "Qwen2.5-7B-Instruct-Q4_K_M.gguf", _inspect(run_req["source"]), time.time())


@app.get("/api/convert/options", dependencies=[api])
def convert_options() -> dict:
    return {"available": CONVERT["problem"] is None, "problem": CONVERT["problem"], "imatrix_available": CONVERT["imatrix"],
            "cluster": _cluster(),
            "quant_options": [{"type": t, "bpw": bpw, "tier": tier, "note": note, "via": via, "needs_imatrix": need, "est_bytes": None,
                               "est_vram_mb": None, "fits_single_gpu": None, "fits_pool": None, "recommended": False}
                              for t, bpw, tier, via, note, need in QUANTS]}


@app.post("/api/convert/inspect", dependencies=[api])
def convert_inspect(body: dict) -> dict:
    return _inspect(body)


@app.post("/api/convert", dependencies=[api])
def convert_start(body: dict) -> dict:
    req = _normalize_request(body)
    if CONVERT["problem"] is not None:
        raise HTTPException(503, CONVERT["problem"])
    insp = _inspect(req["source"])
    if insp["supported"] is False:
        raise HTTPException(400, f"Architecture {insp['architecture']} is not supported by the converter")
    if insp["prequantized"] and insp["prequant_supported"] is False:
        raise HTTPException(400, f"Quant method {insp['prequantized']} is not yet supported: convert the original model "
                                 f"({insp['base_model'] or 'unquantized'}) instead")
    if insp["weight_format"] == "none":
        raise HTTPException(400, "No safetensors or PyTorch weights were found in the source")
    name = _check_output_name(req["name"] or f"{_model_base(req['source'])}-{req['quant']}.gguf")
    req["name"] = name
    q = QUANT_BY_TYPE[req["quant"]]
    if req["advanced"]["imatrix"] == "off" and q[5]:
        raise HTTPException(422, f"{req['quant']} needs an importance matrix (llama-quantize refuses it without one): "
                                 "set the importance matrix to Auto or On")
    if _imatrix_used(req) and not CONVERT["imatrix"]:
        raise HTTPException(503, "llama-imatrix is not installed: rebuild the coordinator image (the importance matrix is needed for "
                                 f"{req['quant']})")
    gb = 1024 ** 3
    opt = next(o for o in insp["options"] if o["type"] == req["quant"])
    dl = insp["source_bytes"] if req["source"].get("hf_repo") else 0
    inter = 0 if q[3] == "convert" else int(insp["params"] * 2)
    need = dl + inter + opt["est_bytes"] + gb // 2
    if need > CONVERT["disk_free_gb"] * gb:
        raise HTTPException(507, f"Not enough disk space in {MODEL_ROOT}/.hf: this conversion needs about {need / gb:.0f} GB "
                                 f"(download {dl / gb:.0f} GB + 16-bit intermediate {inter / gb:.0f} GB + output {opt['est_bytes'] / gb:.0f} GB "
                                 f"+ 0.5 GB margin) but only {CONVERT['disk_free_gb']:.0f} GB are free. Free some space or pick a smaller type.")
    if name in LIBRARY or any(j["output_name"] == name and j["state"] not in ("done", "failed", "cancelled") for j in JOBS.values()):
        raise HTTPException(409, f"{name} already exists in the library or is being produced by another job; pick another name")
    return _conv_json(_conv_new(req, name, insp, time.time()))


@app.get("/api/convert", dependencies=[api])
def convert_list() -> list[dict]:
    return [_conv_json(j) for j in sorted(JOBS.values(), key=lambda j: -j["created_at"])]


@app.get("/api/convert/{job_id}", dependencies=[api])
def convert_get(job_id: str) -> dict:
    return _conv_json(_conv_job(job_id))


@app.post("/api/convert/{job_id}/cancel", dependencies=[api])
def convert_cancel(job_id: str) -> dict:
    j = _conv_job(job_id)
    if j["state"] not in ACTIVE_JOB_STATES:
        raise HTTPException(409, f"Job {job_id} is {j['state']}, only an active job can be cancelled")
    j["failed_stage"], j["state"], j["finished_at"], j["error"] = j["state"], "cancelled", time.time(), None
    j["log_tail"] = j["log_tail"] + ["cancelled by the user"]
    return _clean(j)


@app.post("/api/convert/{job_id}/retry", dependencies=[api])
def convert_retry(job_id: str) -> dict:
    j = _conv_job(job_id)
    if j["state"] not in ("failed", "cancelled"):
        raise HTTPException(409, f"Job {job_id} is {j['state']}, only a failed or cancelled job can be retried")
    if CONVERT["problem"] is not None:
        raise HTTPException(503, CONVERT["problem"])
    j.update(state="queued", stage_progress=None, bytes_done=0, bytes_total=None, output_bytes=None, validation=None, error=None, log_tail=[], failed_stage=None,
             imatrix_used=_imatrix_used(j["request"]), started_at=None, finished_at=None, _t0=time.time())
    return _clean(j)


@app.post("/api/convert/{job_id}/accept", dependencies=[api])
def convert_accept(job_id: str) -> dict:
    j = _conv_job(job_id)
    if j["state"] != "needs_review":
        raise HTTPException(409, f"Job {job_id} is {j['state']}, only a job that needs review can be accepted")
    j["state"], j["finished_at"] = "done", time.time()
    _conv_to_library(j)
    add_event("info", "convert_done", f"{j['output_name']} accepted into the library", None, None)
    return _clean(j)


@app.delete("/api/convert/{job_id}", dependencies=[api])
def convert_delete(job_id: str) -> dict:
    j = _conv_job(job_id)
    if j["state"] in ACTIVE_JOB_STATES:
        raise HTTPException(409, f"Job {job_id} is {j['state']}: cancel it first")
    del JOBS[job_id]  # the library file of a finished job is never touched
    return {"ok": True}


@app.post("/api/_mock/convert_available")
def mock_convert_available(body: dict) -> dict:
    """Dev helper: {"available": false} makes the toolchain "missing" (problem text, jobs refuse to start)."""
    if "imatrix_available" in body:
        CONVERT["imatrix"] = bool(body["imatrix_available"])
    if "disk_free_gb" in body:
        CONVERT["disk_free_gb"] = float(body["disk_free_gb"])
    CONVERT["problem"] = None if body.get("available", True) else str(
        body.get("problem") or "The conversion toolchain is not installed: llama-quantize was not found in /opt/llama/bin (set GPUPOOL_LLAMA_TOOLS_DIR).")
    return {"available": CONVERT["problem"] is None, "problem": CONVERT["problem"], "imatrix_available": CONVERT["imatrix"],
            "disk_free_gb": CONVERT["disk_free_gb"]}



seed_conversions()


app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn

    INSPECT_DELAY[0] = 0.5  # visible spinner; tests keep it at 0
    uvicorn.run(app, host="127.0.0.1", port=8090)
