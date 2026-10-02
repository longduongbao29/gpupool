"""Coordinator FastAPI app: heartbeat intake, admin API, file server, router, metrics."""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from gpupool.common.auth import require_bearer
from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import ModelMeta, ModelSpec, NodeReport, ReplicaEndpoint
from gpupool.coordinator.agent_client import AgentClient
from gpupool.coordinator.reconciler import Reconciler
from gpupool.coordinator.store import Store

ALL_STATES = ["pending", "launching", "ready", "draining", "stopped", "failed"]


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def make_meta_provider(cfg: CoordinatorConfig) -> Callable[[ModelSpec], Awaitable[ModelMeta]]:
    cache: dict[str, ModelMeta] = {}

    async def meta_for(spec: ModelSpec) -> ModelMeta:
        source = spec.source
        if source in cache:
            return cache[source]
        resolved = source
        if source.startswith("coordinator://"):
            name = source[len("coordinator://"):]
            resolved = str(_safe_file(cfg.models_dir, name))
        from gpupool.scheduler.gguf_meta import read_meta  # lazy: owned by another module

        meta = await asyncio.to_thread(read_meta, resolved)
        cache[source] = meta
        return meta

    return meta_for


def _safe_file(models_dir: Path, name: str) -> Path:
    """Resolve `name` to a regular file directly inside models_dir, else raise HTTPException."""
    if (not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name
            or Path(name).name != name):
        raise HTTPException(status_code=400, detail="invalid file name")
    base = models_dir.resolve()
    p = (base / name).resolve()
    if p.parent != base:
        raise HTTPException(status_code=400, detail="invalid file name")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="no such file")
    return p


def create_app(
    cfg: CoordinatorConfig,
    *,
    store: Store | None = None,
    client: AgentClient | None = None,
    meta_for: Callable[[ModelSpec], Awaitable[ModelMeta]] | None = None,
    start_background: bool = True,
) -> FastAPI:
    from gpupool.router.balancer import Balancer
    from gpupool.router.proxy import RouterMetrics, make_router

    store = store or Store(cfg.db_path)
    client = client or AgentClient(cfg.cluster_token)
    balancer = Balancer()
    metrics = RouterMetrics(balancer)
    reconciler = Reconciler(store, cfg, client, meta_for or make_meta_provider(cfg), balancer.outstanding)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(reconciler.run(), name="reconciler") if start_background else None
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await reconciler.shutdown()
            await client.aclose()
            store.close()

    app = FastAPI(title="gpupool coordinator", lifespan=lifespan)
    app.state.store, app.state.reconciler, app.state.balancer = store, reconciler, balancer
    cluster_auth = Depends(require_bearer(cfg.cluster_token))
    admin_auth = Depends(require_bearer(cfg.admin_key))

    # ---- agents
    @app.post("/internal/heartbeat", dependencies=[cluster_auth])
    def heartbeat(report: NodeReport) -> dict:
        store.upsert_node(report, reconciler.clock())
        return {"ok": True}

    @app.get("/files/{name:path}", dependencies=[cluster_auth])
    def get_file(name: str):
        p = _safe_file(cfg.models_dir, name)
        return FileResponse(p, media_type="application/octet-stream", filename=p.name)

    # ---- admin
    def _spec_or_404(name: str) -> ModelSpec:
        spec = store.get_model(name)
        if spec is None:
            raise HTTPException(404, f"unknown model {name}")
        return spec

    @app.post("/admin/models", dependencies=[admin_auth])
    def put_model(spec: ModelSpec) -> ModelSpec:
        store.put_model(spec)
        return spec

    @app.delete("/admin/models/{name}", dependencies=[admin_auth])
    async def delete_model(name: str) -> dict:
        _spec_or_404(name)
        for r in store.list_replicas(model=name, states={"pending", "launching", "ready"}):
            await reconciler.drain(r.replica_id)
        store.delete_model(name)
        return {"ok": True}

    @app.post("/admin/models/{name}/scale", dependencies=[admin_auth])
    def scale(name: str, replicas: int) -> ModelSpec:
        if replicas < 0:
            raise HTTPException(422, "replicas must be >= 0")
        spec = _spec_or_404(name).model_copy(update={"replicas": replicas})
        store.put_model(spec)
        return spec

    @app.post("/admin/deploy/{model}", dependencies=[admin_auth])
    async def deploy(model: str, dry_run: int = 0):
        spec = _spec_or_404(model)
        if dry_run:
            try:
                return await reconciler.plan_for(spec)
            except Exception as e:
                code = 409 if type(e).__name__ == "NoFit" else 400
                raise HTTPException(code, f"{type(e).__name__}: {e}")
        await reconciler.tick()
        return [r.model_dump() for r in store.list_replicas(model=model)]

    @app.delete("/admin/replicas/{replica_id}", dependencies=[admin_auth])
    async def drain_replica(replica_id: str) -> dict:
        if store.get_replica(replica_id) is None:
            raise HTTPException(404, f"unknown replica {replica_id}")
        await reconciler.drain(replica_id)
        return {"ok": True}

    @app.get("/admin/status", dependencies=[admin_auth])
    def status() -> dict:
        now = reconciler.clock()
        return {
            "nodes": [
                {
                    "node_id": n.report.node_id,
                    "alive": now - n.last_seen <= cfg.heartbeat_timeout_s,
                    "last_seen": n.last_seen,
                    "agent_url": n.report.agent_url,
                    "host": n.report.host,
                    "devices": [d.model_dump() for d in n.report.devices],
                    "engines": [e.model_dump() for e in n.report.engines],
                }
                for n in store.list_nodes()
            ],
            "models": [m.model_dump() for m in store.list_models()],
            "replicas": [
                {**r.model_dump(), "outstanding": balancer.outstanding(r.replica_id)}
                for r in store.list_replicas()
            ],
        }

    # ---- router
    def get_candidates(model: str) -> list[ReplicaEndpoint]:
        now = reconciler.clock()
        nodes = {n.report.node_id: n for n in store.list_nodes()}
        out = []
        for r in store.list_replicas(model=model, states={"ready"}):
            n = nodes.get(r.placement.head_node)
            if n is None or now - n.last_seen > cfg.heartbeat_timeout_s:
                continue
            out.append(ReplicaEndpoint(
                replica_id=r.replica_id, model=model,
                base_url=f"http://{n.report.host}:{r.placement.head_port}"))
        return out

    app.include_router(make_router(
        get_candidates=get_candidates,
        list_models=lambda: [m.name for m in store.list_models()],
        balancer=balancer, metrics=metrics, api_keys=cfg.api_keys,
        on_replica_error=reconciler.note_error,
    ))

    @app.get("/metrics")
    def render_metrics() -> PlainTextResponse:
        now = reconciler.clock()
        lines = [metrics.render().rstrip("\n")]
        lines.append("# TYPE gpupool_device_free_mb gauge")
        free, usable, alive = [], [], []
        for n in store.list_nodes():
            nid = _esc(n.report.node_id)
            alive.append(f'gpupool_node_alive{{node="{nid}"}} '
                         f'{int(now - n.last_seen <= cfg.heartbeat_timeout_s)}')
            for d in n.report.devices:
                lab = f'node="{nid}",device="{_esc(d.device_id)}"'
                free.append(f"gpupool_device_free_mb{{{lab}}} {d.free_mb}")
                usable.append(f"gpupool_device_usable_mb{{{lab}}} {d.usable_mb}")
        lines += free
        lines.append("# TYPE gpupool_device_usable_mb gauge")
        lines += usable
        lines.append("# TYPE gpupool_node_alive gauge")
        lines += alive
        lines.append("# TYPE gpupool_replicas gauge")
        counts: dict[tuple[str, str], int] = {}
        for r in store.list_replicas():
            counts[(r.model, r.state)] = counts.get((r.model, r.state), 0) + 1
        for m in store.list_models():
            for s in ALL_STATES:
                counts.setdefault((m.name, s), 0)
        for (m, s), c in sorted(counts.items()):
            lines.append(f'gpupool_replicas{{model="{_esc(m)}",state="{s}"}} {c}')
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app


def run_coordinator(cfg: CoordinatorConfig) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
