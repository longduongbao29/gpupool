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
Rebalancing: POST /api/rebalance {dry_run} lists one qualifying move (a "chat-auto" replica onto CTG-Server-1/CUDA0). With
dry_run false it starts the move: a replacement replica appears (state starting), GET /api/state carries
`rebalance.in_progress`, and ~20 s later the old replica is dropped (events rebalance_started, rebalanced). Afterwards the
dry run answers with no moves ("all replicas are well placed"). `rebalance.next_run_ts` is 10 min after start.
"""
from __future__ import annotations

import math
import time
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


BANDWIDTH_GBPS = {"NVIDIA H100 80GB": 3350.0, "NVIDIA RTX 4090": 1008.0, "NVIDIA A100 40GB": 1555.0, "NVIDIA RTX 3090": 936.2}


def _gpu(i: int, name: str, total_mb: int, phase: float, driver="535.154.05", cuda="12.2") -> dict:
    return {"device_id": f"CUDA{i}", "kind": "cuda", "name": name, "total_mb": total_mb, "free_mb": total_mb,
            "usable_mb": total_mb, "util_pct": 0, "temp_c": 40, "power_w": 60, "processes": [],
            "driver": driver, "cuda": cuda, "bandwidth_gbps": BANDWIDTH_GBPS.get(name), "_phase": phase, "_base": 20 + 12 * i}


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
    LIBRARY["llama-8b.gguf"] = {"name": "llama-8b.gguf", "path": "/data/models/llama-8b.gguf", "source": "hf",
        "hf_repo": "bartowski/Llama-3.1-8B-GGUF", "hf_file": "llama-8b.gguf", "bytes": 5_000_000_000,
        "downloaded": 500_000_000, "status": "downloading", "error": None, "created_at": T0, "_t": T0}
    MODELS["qwen3b"] = {"spec": {"name": "qwen3b", "source": "coordinator://qwen2.5-3b-q4.gguf", "ctx_size": 4096,
        "parallel": 1, "replicas": 0, "pin_devices": [], "priority": 50, "preemptible": True, "spread": "gpu"}, "file": "qwen2.5-3b-q4.gguf", "state": "stopped", "error": None,
        "replicas": [], "_t": 0.0}
    MODELS["chat-auto"] = {"spec": {"name": "chat-auto", "source": "coordinator://qwen2.5-3b-q4.gguf", "ctx_size": 8192, "parallel": 4,
        "replicas": 2, "pin_devices": [], "priority": 70, "preemptible": True, "spread": "gpu", "min_replicas": 1, "max_replicas": 4,
        "autoscale": {"target_busy": 0.7, "up_after_s": 30, "down_after_s": 300}, "idle_unload_s": None},
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


def _place(m: dict, replicas: int = 1) -> None:
    """Fake scheduler: first enabled GPUs of the first alive server (or the pinned ones)."""
    pins = m["spec"]["pin_devices"]
    cands = [(s["node_id"], g["device_id"]) for s in SERVERS.values() if s["alive"] for g in s["gpus"]
             if s["gpu_enabled"].get(g["device_id"], True) and (not pins or f"{s['node_id']}/{g['device_id']}" in pins)]
    if not cands:
        raise HTTPException(409, "No enabled GPU is available")
    cands.sort(key=lambda c: c[0] != "CTG-Server-2")  # prefer Server-2 so the outage demo has something to move
    chosen = cands[:2]
    m["replicas"] = [{"replica_id": f"{m['spec']['name']}-{i}a2b3c", "model": m["spec"]["name"], "state": "ready",
        "error": None, "outstanding": 0, "created_at": time.time(), "updated_at": time.time(),
        "placement": _placement(m["spec"]["name"], chosen)} for i in range(replicas)]
    m["state"], m["error"] = "running", None


def _placement(name: str, chosen: list[tuple[str, str]]) -> dict:
    return {"model": name, "replica_id": f"{name}-plan", "tier": "multi_node" if len({c[0] for c in chosen}) > 1 else "single_node",
            "head_node": chosen[0][0], "head_port": 9000, "tensor_split": [1.0] * len(chosen), "est_total_mb": 5000 * len(chosen),
            "score": 82.5, "est_decode_tps": 96.4,
            "reasons": ["fastest GPUs with room (about 1008 GB/s)", "spread: replicas on different GPUs"],
            "assignments": [{"node_id": n, "device_id": d, "llama_device": d, "rpc_endpoint": None, "layers": 18 // len(chosen),
                             "est_mb": 5000} for n, d in chosen]}


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


def _scaling(m: dict) -> dict:
    sp = m["spec"]
    mn, mx = sp.get("min_replicas"), sp.get("max_replicas")
    n = max(1, int(sp.get("replicas") or 1))
    lo, hi = (n, n) if mn is None or mx is None else (mn, mx)
    desired = len(m["replicas"]) if m["replicas"] else (0 if m["state"] == "idle" else lo)
    return {"min": lo, "max": hi, "desired": desired, "avg_busy": _busy(m), "unloaded": m["state"] == "idle"}


def _state_model(m: dict) -> dict:
    return {**_clean(m), "scaling": _scaling(m)}


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
    m = MODELS.setdefault(name, {"spec": {"name": name, "replicas": 0}, "state": "stopped", "error": None, "replicas": [], "_t": 0.0})
    m["file"] = file
    m["spec"].update(source=f"coordinator://{file}", ctx_size=int(body.get("ctx_size", 4096)), parallel=int(body.get("parallel", 1)),
                     pin_devices=list(body.get("pin_devices", [])), priority=_int_in(body.get("priority", 50), 0, 100, "priority"),
                     preemptible=_bool(body.get("preemptible", True), "preemptible"),
                     spread=_choice(body.get("spread", "gpu"), ("gpu", "node", "none"), "spread"))
    m["spec"].update(_scaling_fields(body))
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
    chosen = [tuple(p.split("/", 1)) for p in pins][:3] or [("CTG-Server-1", "CUDA0"), ("CTG-Server-2", "CUDA0")]
    return _placement(name, chosen)


def _need_mb(file_bytes: int, ctx: int, parallel: int) -> int:
    return int(file_bytes / 1048576 * 1.1 + ctx * parallel * 3.0 + 300)


@app.post("/api/recommend", dependencies=[api])
def recommend(body: dict) -> dict:
    item = LIBRARY.get(body.get("file"))
    if item is None:
        raise HTTPException(404, f"{body.get('file')} is not in the library")
    ctx = _int_in(body.get("ctx_size", 4096), 256, 10_000_000, "ctx_size")
    parallel = _int_in(body.get("parallel", 1), 1, 64, "parallel")
    limit = max(1, min(10, int(body.get("limit", 3))))
    pins = list(body.get("pin_devices") or [])
    alive = [s for s in SERVERS.values() if s["alive"]]
    gpus = [(s["node_id"], g) for s in alive for g in s["gpus"]
            if s["gpu_enabled"].get(g["device_id"], True) and (not pins or f"{s['node_id']}/{g['device_id']}" in pins)]
    need = _need_mb(item["bytes"], ctx, parallel)
    fixed = _need_mb(item["bytes"], 0, 1)
    biggest_gpu = max((g["usable_mb"] for _, g in gpus), default=0)
    per_node: dict[str, int] = {}
    for n, g in gpus:
        per_node[n] = per_node.get(n, 0) + g["usable_mb"]
    biggest_node = max(per_node.values(), default=0)
    max_ctx_single = max(0, int((biggest_gpu - fixed) / (3.0 * parallel)) // 256 * 256)
    out = {"need_mb": need, "options": [], "max_ctx_single_gpu": max_ctx_single or None, "not_possible": None}
    if need > biggest_node:
        fits = max(0, int((biggest_node - fixed) / (3.0 * parallel)) // 256 * 256)
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
                     "est_decode_tps": tps, "est_total_mb": need,
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
        need = _need_mb(LIBRARY[file]["bytes"], ctx, parallel)
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


app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8090)
