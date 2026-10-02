"""Dev mock of the gpupool coordinator API for working on the UI without a cluster.

    uv run python scripts/ui_mock_server.py   ->  http://127.0.0.1:8090   (admin key: dev)

Everything is in memory and driven by wall-clock time: GPU utilization fluctuates, a download
progresses, models walk stopped -> starting -> running. About 30 s after start, server
CTG-Server-2 "dies" (events node_offline, realloc_started, realloc_done) and at ~50 s one GPU of
CTG-Server-1 vanishes (gpu_missing). POST /api/_mock/kill/{node_id} triggers the outage by hand.
Setting ctx_size above 32768 makes /plan answer 409 (does not fit).
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


def _gpu(i: int, name: str, total_mb: int, phase: float, driver="535.154.05", cuda="12.2") -> dict:
    return {"device_id": f"CUDA{i}", "kind": "cuda", "name": name, "total_mb": total_mb, "free_mb": total_mb,
            "usable_mb": total_mb, "util_pct": 0, "temp_c": 40, "power_w": 60, "processes": [],
            "driver": driver, "cuda": cuda, "_phase": phase, "_base": 20 + 12 * i}


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
        "parallel": 1, "replicas": 0, "pin_devices": []}, "file": "qwen2.5-3b-q4.gguf", "state": "stopped", "error": None,
        "replicas": [], "_t": 0.0}
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
            "assignments": [{"node_id": n, "device_id": d, "llama_device": d, "rpc_endpoint": None, "layers": 18 // len(chosen),
                             "est_mb": 5000} for n, d in chosen]}


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
            _place(m)
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
        "models": [_clean(m) for m in MODELS.values()],
        "library": [_clean(i) for i in LIBRARY.values()],
        "settings": dict(SETTINGS),
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
        name = p.split("/")[-1]
        item = {"name": name, "path": p, "source": "path", "hf_repo": None, "hf_file": None, "bytes": 1_500_000_000,
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
                     pin_devices=list(body.get("pin_devices", [])))
    return m["spec"]


def _model(name: str) -> dict:
    if name not in MODELS:
        raise HTTPException(404, "unknown model")
    return MODELS[name]


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


@app.post("/api/_mock/kill/{node_id}")
def mock_kill(node_id: str) -> dict:
    kill(node_id)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8090)
