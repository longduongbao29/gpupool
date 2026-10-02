"""JSON API behind the web UI: state, servers, GPU switches, models, events. All under /api."""
from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, params
from pydantic import BaseModel, Field, ValidationError

from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import ACTIVE_STATES, LIVE_STATES, ModelSpec
from gpupool.coordinator.agent_client import AgentError
from gpupool.coordinator.store import ServerRecord, gpu_key
from gpupool.scheduler.placement import NoFit

log = logging.getLogger("gpupool.api")

NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
PIN_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
COORD_PREFIX = "coordinator://"


class AddServerBody(BaseModel):
    agent_url: str


class GpuBody(BaseModel):
    enabled: bool


class ModelBody(BaseModel):
    file: str
    ctx_size: int = Field(default=4096, ge=1)
    parallel: int = Field(default=1, ge=1)
    pin_devices: list[str] = Field(default_factory=list)


class StartBody(BaseModel):
    replicas: int = Field(default=1, ge=1)


class ReadBody(BaseModel):
    up_to_id: int


def normalize_agent_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise HTTPException(400, "agent_url must look like http://host:port")
    return url


def spec_or_404(store, name: str) -> ModelSpec:
    spec = store.get_model(name)
    if spec is None:
        raise HTTPException(404, f"unknown model {name}")
    return spec


def plan_http_error(e: Exception) -> HTTPException:
    """Map a planning failure to an HTTP error: no capacity is a conflict, a missing model file
    is not found, anything else a bad request."""
    code = 409 if isinstance(e, NoFit) else 404 if isinstance(e, FileNotFoundError) else 400
    return HTTPException(code, f"{type(e).__name__}: {e}")


async def drain_and_delete_model(store, reconciler, name: str) -> None:
    """Drain the model's replicas, then drop its spec. Shared by /api and /admin so they cannot drift."""
    spec_or_404(store, name)
    for r in store.list_replicas(model=name, states=set(ACTIVE_STATES)):
        await reconciler.drain(r.replica_id)
    store.delete_model(name)


def make_api_router(*, store, reconciler, poller, balancer, library, cfg: CoordinatorConfig,
                    admin_dep, notifier=None) -> APIRouter:
    """`admin_dep` may be a callable or an already-built Depends(...)."""
    dep = admin_dep if isinstance(admin_dep, params.Depends) else Depends(admin_dep)
    router = APIRouter(prefix="/api", dependencies=[dep])
    notifier = notifier or reconciler.notifier

    def now() -> float:
        return reconciler.clock()

    def emit(*a, **kw) -> None:
        try:
            notifier.emit(*a, **kw)
        except Exception:
            log.exception("emitting event failed")

    # ------------------------------------------------------------------ state helpers
    def server_entry(s: ServerRecord, nodes: dict, flags: dict) -> dict:
        n = nodes.get(s.node_id)
        devices = n.report.devices if n else []
        return {
            "node_id": s.node_id, "agent_url": s.agent_url, "added_at": s.added_at,
            "alive": bool(n and n.alive(now(), cfg.heartbeat_timeout_s)),
            "last_seen": n.last_seen if n else 0.0,
            "report": n.report.model_dump(mode="json") if n else None,
            "gpu_enabled": {d.device_id: flags.get((s.node_id, gpu_key(d)), True) for d in devices},
        }

    def model_entry(spec: ModelSpec, reps: list) -> dict:
        """`reps`: this model's replicas, oldest first."""
        active = [r for r in reps if r.state in LIVE_STATES]
        failed = [r for r in reps if r.state == "failed"]
        newest_failed = failed[-1] if failed else None
        ready = any(r.state == "ready" for r in reps)
        launching = any(r.state in ("pending", "launching") for r in reps)
        error = None
        if spec.replicas > 0:
            if ready:
                state = "running"
            elif launching:
                state = "starting"
            elif newest_failed is not None:
                state, error = "failed", newest_failed.error or "replica failed"
            elif (nofit := reconciler.nofit_reason(spec.name)):
                state, error = "failed", nofit
            else:
                state = "starting"  # desired > 0, the next reconcile tick will launch it
        elif any(r.state in ("draining", "ready", "launching", "pending") for r in reps):
            state = "stopping"
        else:
            state = "stopped"
        if error is None and newest_failed is not None and state == "starting":
            error = newest_failed.error
        listed = active + ([newest_failed] if newest_failed is not None else [])
        listed.sort(key=lambda r: (r.created_at, r.replica_id))
        file = spec.source[len(COORD_PREFIX):] if spec.source.startswith(COORD_PREFIX) else None
        return {
            "spec": spec.model_dump(mode="json"), "file": file, "state": state, "error": error,
            "replicas": [{**r.model_dump(mode="json"), "outstanding": balancer.outstanding(r.replica_id)}
                         for r in listed],
        }

    def server_or_404(node_id: str) -> ServerRecord:
        s = store.get_server(node_id)
        if s is None:
            raise HTTPException(404, f"unknown server {node_id}")
        return s

    # ------------------------------------------------------------------ state
    @router.get("/state")
    async def state() -> dict:
        nodes = {n.report.node_id: n for n in store.list_nodes()}
        flags = store.gpu_flags()
        servers = [server_entry(s, nodes, flags) for s in store.list_servers()]
        gpus_total = gpus_enabled = pool_total = pool_usable = 0
        for s in servers:
            if s["report"] is None:
                continue
            for d in s["report"]["devices"]:
                if d["kind"] != "cuda":
                    continue
                on = s["gpu_enabled"].get(d["device_id"], True)
                gpus_total += 1
                if on:
                    gpus_enabled += 1
                    if s["alive"]:
                        pool_total += d["total_mb"]
                        pool_usable += d["usable_mb"]
        by_model: dict[str, list] = {}
        for r in store.list_replicas():  # one query, not one per model; oldest first
            by_model.setdefault(r.model, []).append(r)
        models = [model_entry(m, by_model.get(m.name, [])) for m in store.list_models()]
        return {
            "summary": {
                "servers_total": len(servers), "servers_online": sum(1 for s in servers if s["alive"]),
                "gpus_total": gpus_total, "gpus_enabled": gpus_enabled,
                "pool_total_mb": pool_total, "pool_usable_mb": pool_usable,
                "models_running": sum(1 for m in models if m["state"] == "running"),
            },
            "servers": servers,
            "models": models,
            "library": [i.model_dump(mode="json") for i in library.list()],
            "settings": {"public_url": cfg.public_url, "cluster_token": cfg.cluster_token,
                         "api_keys_set": bool(cfg.api_keys)},
            "events": [e.model_dump(mode="json") for e in store.list_events(limit=50)],
            "unread_events": store.unread_count(),
        }

    # ------------------------------------------------------------------ servers
    @router.post("/servers")
    async def add_server(body: AddServerBody) -> dict:
        url = normalize_agent_url(body.agent_url)
        try:
            report = await poller.probe(url)
        except AgentError as e:
            if e.status == 401:
                raise HTTPException(400, f"agent at {url} rejected the token (401): check the cluster token")
            raise HTTPException(400, f"agent at {url} answered {e.status}: {e.body}")
        except httpx.HTTPError as e:
            raise HTTPException(400, f"cannot reach agent at {url}: {type(e).__name__} {e}".strip())
        except (ValidationError, ValueError) as e:
            raise HTTPException(400, f"{url} did not return a valid agent report: {e}")
        if store.get_server(report.node_id) is not None:
            raise HTTPException(409, f"server {report.node_id} is already registered")
        t = now()
        rec = ServerRecord(node_id=report.node_id, agent_url=url, added_at=t)
        store.add_server(rec)
        store.clear_removed(report.node_id)  # a manual add undoes an earlier removal
        store.upsert_node(report, t)
        emit("info", "server_added", f"Server {rec.node_id} added ({url})", node_id=rec.node_id)
        nodes = {n.report.node_id: n for n in store.list_nodes()}
        return server_entry(rec, nodes, store.gpu_flags())

    @router.delete("/servers/{node_id}")
    async def delete_server(node_id: str) -> dict:
        server_or_404(node_id)
        await reconciler.remove_node(node_id)
        store.mark_removed(node_id, now())  # sticks against the agent's auto-join until re-added here
        emit("info", "server_removed", f"Server {node_id} removed; its replicas were stopped", node_id=node_id)
        return {"ok": True}

    @router.put("/servers/{node_id}/gpus/{device_id}")
    async def set_gpu(node_id: str, device_id: str, body: GpuBody) -> dict:
        server_or_404(node_id)
        # The UI addresses GPUs by device_id; flags are stored by the card's identity.
        node = next((n for n in store.list_nodes() if n.report.node_id == node_id), None)
        dev = next((d for d in node.report.devices if d.device_id == device_id), None) if node else None
        store.set_gpu_enabled(node_id, gpu_key(dev) if dev else device_id, body.enabled)
        reconciler.wake()
        return {"node_id": node_id, "device_id": device_id, "enabled": body.enabled}

    # ------------------------------------------------------------------ models
    def check_name(name: str) -> None:
        if not NAME_RE.match(name):
            raise HTTPException(422, "model name must match [A-Za-z0-9._-]{1,64}")

    @router.put("/models/{name}")
    async def put_model(name: str, body: ModelBody) -> dict:
        check_name(name)
        item = library.get(body.file)
        if item is None or item.status != "ready":
            raise HTTPException(422, f"file {body.file!r} is not a ready library item")
        registered = {s.node_id for s in store.list_servers()}
        for pin in body.pin_devices:
            if not PIN_RE.match(pin):
                raise HTTPException(422, f"pin_devices entry {pin!r} must look like node/device")
            if pin.split("/", 1)[0] not in registered:
                raise HTTPException(422, f"pin_devices entry {pin!r} names an unregistered server")
        existing = store.get_model(name)
        spec = ModelSpec(
            name=name, source=COORD_PREFIX + body.file, ctx_size=body.ctx_size, parallel=body.parallel,
            replicas=existing.replicas if existing else 0, pin_devices=list(dict.fromkeys(body.pin_devices)))
        store.put_model(spec)
        reconciler.wake()
        return spec.model_dump(mode="json")

    @router.post("/models/{name}/start")
    async def start_model(name: str, body: StartBody | None = Body(default=None)) -> dict:
        spec = spec_or_404(store, name)
        spec = spec.model_copy(update={"replicas": (body or StartBody()).replicas})
        store.put_model(spec)
        emit("info", "model_started", f"Model {name} started ({spec.replicas} replica(s) requested)", model=name)
        reconciler.wake()  # not tick(): that waits on the tick lock and agent HTTP calls
        return spec.model_dump(mode="json")

    @router.post("/models/{name}/stop")
    async def stop_model(name: str) -> dict:
        spec = spec_or_404(store, name).model_copy(update={"replicas": 0})
        store.put_model(spec)
        emit("info", "model_stopped", f"Model {name} stopped", model=name)
        reconciler.wake()
        return spec.model_dump(mode="json")

    @router.delete("/models/{name}")
    async def delete_model(name: str) -> dict:
        await drain_and_delete_model(store, reconciler, name)
        emit("info", "model_stopped", f"Model {name} removed", model=name)
        reconciler.wake()
        return {"ok": True}

    @router.post("/models/{name}/plan")
    async def plan_model(name: str) -> Any:
        spec = spec_or_404(store, name)
        try:
            return (await reconciler.plan_for(spec)).model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as e:
            raise plan_http_error(e) from e

    # ------------------------------------------------------------------ events
    @router.get("/events")
    async def events(limit: int = 200, after_id: int | None = None) -> dict:
        limit = max(1, min(limit, 1000))
        return {"events": [e.model_dump(mode="json") for e in store.list_events(limit=limit, after_id=after_id)],
                "unread": store.unread_count()}

    @router.post("/events/read")
    async def mark_read(body: ReadBody) -> dict:
        store.mark_read(body.up_to_id)
        return {"unread": store.unread_count()}

    return router
