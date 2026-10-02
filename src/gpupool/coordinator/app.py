"""Coordinator FastAPI app: heartbeat intake, admin API, file server, router, metrics."""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from gpupool.common.auth import require_bearer
from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import ModelMeta, ModelSpec, NodeReport, ReplicaEndpoint
from gpupool.coordinator.agent_client import AgentClient
from gpupool.coordinator.api import make_api_router
from gpupool.coordinator.events import Notifier
from gpupool.coordinator.library import Library
from gpupool.coordinator.library_api import make_files_router, make_library_router
from gpupool.coordinator.poller import Poller
from gpupool.coordinator.reconciler import Reconciler
from gpupool.coordinator.store import Store

UI_DIR = Path(__file__).resolve().parents[1] / "ui"

ALL_STATES = ["pending", "launching", "ready", "draining", "stopped", "failed"]


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def make_meta_provider(cfg: CoordinatorConfig, library: Library | None = None
                       ) -> Callable[[ModelSpec], Awaitable[ModelMeta]]:
    cache: dict[str, ModelMeta] = {}

    async def meta_for(spec: ModelSpec) -> ModelMeta:
        source = spec.source
        if source in cache:
            return cache[source]
        resolved = source
        if source.startswith("coordinator://"):
            name = source[len("coordinator://"):]
            found = library.resolve(name) if library is not None else None
            resolved = str(found) if found is not None else str(_safe_file(cfg.models_dir, name))
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
    library: Library | None = None,
) -> FastAPI:
    from gpupool.router.balancer import Balancer
    from gpupool.router.proxy import RouterMetrics, make_router

    store = store or Store(cfg.db_path)
    client = client or AgentClient(cfg.cluster_token)
    if library is None:
        if str(cfg.db_path) != ":memory:":
            Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
        library = Library(cfg.db_path, cfg.models_dir, cfg.hf_token)
    balancer = Balancer()
    metrics = RouterMetrics(balancer)
    notifier = Notifier(store, cfg.webhook_url)
    reconciler = Reconciler(store, cfg, client, meta_for or make_meta_provider(cfg, library),
                            balancer.outstanding, notifier=notifier)
    poller = Poller(store, client, cfg.poll_s)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = []
        if start_background:
            library.resume()  # restart HF downloads interrupted by a previous run
            tasks = [asyncio.create_task(poller.run(), name="poller"),
                     asyncio.create_task(reconciler.run(), name="reconciler")]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await reconciler.shutdown()
            await library.shutdown()
            await notifier.aclose()
            await client.aclose()
            store.close()

    app = FastAPI(title="gpupool coordinator", lifespan=lifespan)
    app.state.store, app.state.reconciler, app.state.balancer = store, reconciler, balancer
    app.state.poller, app.state.library, app.state.notifier = poller, library, notifier
    cluster_auth = Depends(require_bearer(cfg.cluster_token))
    admin_auth = Depends(require_bearer(cfg.admin_key))

    # ---- agents
    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/internal/heartbeat", dependencies=[cluster_auth])
    def heartbeat(report: NodeReport) -> dict:
        # Push heartbeats are accepted from registered servers only: a server removed in the
        # UI must not come back by itself. Registration happens via the UI / POST /api/servers.
        if store.get_server(report.node_id) is None:
            raise HTTPException(403, f"server {report.node_id} is not registered")
        store.upsert_node(report, reconciler.clock())
        return {"ok": True}

    def model_uses_file(name: str) -> bool:
        return any(m.source == f"coordinator://{name}" for m in store.list_models())

    app.include_router(make_files_router(library, require_bearer(cfg.cluster_token)))
    app.include_router(make_library_router(library, require_bearer(cfg.admin_key), model_uses_file))
    app.include_router(make_api_router(store=store, reconciler=reconciler, poller=poller,
                                       balancer=balancer, library=library, cfg=cfg,
                                       admin_dep=admin_auth, notifier=notifier))

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

    # The web UI. Mounted last: a mount at "/" would otherwise shadow the API routes.
    if UI_DIR.is_dir():
        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app


def run_coordinator(cfg: CoordinatorConfig) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
