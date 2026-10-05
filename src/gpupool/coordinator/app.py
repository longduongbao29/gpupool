"""Coordinator FastAPI app: heartbeat intake, admin API, file server, router, metrics."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from gpupool.common.auth import require_bearer
from gpupool.common.config import CoordinatorConfig, detect_local_ip, load_or_create_secrets
from gpupool.common.net import external_client
from gpupool.common.models import ALL_REPLICA_STATES, ModelMeta, ModelSpec, NodeReport, ReplicaEndpoint
from gpupool.converter import source as convert_source
from gpupool.converter.jobs import ConvertManager
from gpupool.converter.models import ClusterVram, InspectResult, SourceSpec
from gpupool.converter.toolchain import Toolchain
from gpupool.coordinator.agent_client import AgentClient, AgentError
from gpupool.coordinator.autoscaler import Autoscaler
from gpupool.coordinator.api import (
    drain_and_delete_model,
    make_api_router,
    normalize_agent_url,
    plan_http_error,
    spec_or_404,
)
from gpupool.coordinator.events import Notifier
from gpupool.coordinator.convert_api import make_convert_router
from gpupool.coordinator.library import Library
from gpupool.coordinator.library_api import make_files_router, make_library_router
from gpupool.coordinator.poller import Poller
from gpupool.coordinator.reconciler import Reconciler
from gpupool.coordinator.store import ServerRecord, Store, gpu_key
from gpupool.router.balancer import Balancer
from gpupool.router.proxy import RouterMetrics, make_router, prom_label_escape
from gpupool.scheduler.gguf_meta import read_meta, read_meta_parts

log = logging.getLogger("gpupool.coordinator")

# Loop-lag watchdog (diagnostic only): a blocked loop starves polls and API calls, and the
# symptom elsewhere (nodes "dying") is hard to trace back without this line.
WATCHDOG_TICK_S = 0.25
WATCHDOG_LAG_S = 1.0
WATCHDOG_LOG_EVERY_S = 10.0


async def loop_lag_watchdog(tick_s: float = WATCHDOG_TICK_S, lag_s: float = WATCHDOG_LAG_S,
                            log_every_s: float = WATCHDOG_LOG_EVERY_S,
                            clock: Callable[[], float] = time.monotonic,
                            sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
    last_logged = float("-inf")
    while True:
        t0 = clock()
        await sleep(tick_s)
        t1 = clock()
        over = t1 - t0 - tick_s
        if over > lag_s and t1 - last_logged >= log_every_s:
            last_logged = t1
            log.warning("event loop blocked for %.1f s", over)

UI_DIR = Path(__file__).resolve().parents[1] / "ui"

class JoinBody(BaseModel):
    agent_url: str


def make_meta_provider(cfg: CoordinatorConfig, library: Library | None = None
                       ) -> Callable[[ModelSpec], Awaitable[ModelMeta]]:
    cache: dict[str, ModelMeta] = {}

    async def meta_for(spec: ModelSpec) -> ModelMeta:
        source = spec.source
        if source in cache:
            return cache[source]
        if source.startswith("coordinator://"):
            name = source[len("coordinator://"):]
            paths = library.part_paths(name) if library is not None else None
            if paths is None:
                paths = [_safe_file(cfg.models_dir, name)]
        else:
            paths = None
        if paths is None:
            meta = await asyncio.to_thread(read_meta, source)
        elif len(paths) == 1:
            meta = await asyncio.to_thread(read_meta, str(paths[0]))
        else:
            # Split GGUF: every part holds some tensors; reading part 1 alone under-counts VRAM.
            meta = await asyncio.to_thread(read_meta_parts, [str(p) for p in paths])
        cache[source] = meta
        return meta

    return meta_for


def _safe_file(models_dir: Path, name: str) -> Path:
    """Resolve `name` to a regular file directly inside models_dir.

    Raises ValueError for an invalid name and FileNotFoundError for a missing file; this runs
    in the reconciler's meta provider, not in a route, so it must not raise HTTP errors.
    """
    if (not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name
            or Path(name).name != name):
        raise ValueError("invalid file name")
    base = models_dir.resolve()
    p = (base / name).resolve()
    if p.parent != base:
        raise ValueError("invalid file name")
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {name}")
    return p


def create_app(
    cfg: CoordinatorConfig,
    *,
    store: Store | None = None,
    client: AgentClient | None = None,
    meta_for: Callable[[ModelSpec], Awaitable[ModelMeta]] | None = None,
    start_background: bool = True,
    library: Library | None = None,
    autoscaler: Autoscaler | None = None,
    convert_manager=None,
    convert_inspect: Callable[[SourceSpec], Awaitable[InspectResult]] | None = None,
) -> FastAPI:
    store = store or Store(cfg.db_path)
    client = client or AgentClient(cfg.cluster_token)
    if library is None:
        if str(cfg.db_path) != ":memory:":
            Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
        library = Library(cfg.db_path, cfg.models_dir, cfg.hf_token,
                          path_map=cfg.path_map, model_roots=cfg.model_roots)
    balancer = Balancer()
    metrics = RouterMetrics(balancer)
    notifier = Notifier(store, cfg.webhook_url)
    reconciler = Reconciler(store, cfg, client, meta_for or make_meta_provider(cfg, library),
                            balancer.outstanding, notifier=notifier)
    poller = Poller(store, client, cfg.poll_s)
    autoscaler = autoscaler or Autoscaler(store, cfg, balancer.outstanding, notifier,
                                          clock=reconciler.clock, wake=reconciler.wake)
    reconciler.autoscaler = autoscaler
    reconciler.poller = poller
    autoscaler.node_alive = reconciler.node_alive

    # The conversion feature. A missing toolchain (convert_dir, tools) is not an error: the manager
    # reports it through available() and refuses jobs. A broken converter package is a bug and
    # must fail loudly, not silently disable the feature.
    own_hf_http = external_client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True)
    hf = convert_source.HfClient(own_hf_http, cfg.hf_token)
    toolchain = Toolchain(cfg.convert_dir, cfg.convert_python, cfg.llama_tools_dir)
    if convert_manager is None:
        convert_manager = ConvertManager(cfg.db_path, cfg.models_dir, toolchain, library, hf,
                                         threads=cfg.convert_threads)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = []
        if start_background:
            library.resume()  # restart HF downloads interrupted by a previous run
            convert_manager.start()  # interrupted conversion jobs go back to the queue
            tasks = [asyncio.create_task(poller.run(), name="poller"),
                     asyncio.create_task(reconciler.run(), name="reconciler"),
                     asyncio.create_task(autoscaler.run(), name="autoscaler"),
                     asyncio.create_task(loop_lag_watchdog(), name="loop-watchdog")]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await autoscaler.aclose()
            await reconciler.shutdown()
            await convert_manager.shutdown()  # before the library: it may still be adding a file
            await own_hf_http.aclose()
            await library.shutdown()
            await notifier.aclose()
            await client.aclose()
            store.close()

    app = FastAPI(title="gpupool coordinator", lifespan=lifespan)
    app.state.store, app.state.reconciler, app.state.balancer = store, reconciler, balancer
    app.state.poller, app.state.library, app.state.notifier = poller, library, notifier
    app.state.autoscaler = autoscaler
    app.state.convert = convert_manager
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

    @app.post("/internal/join", dependencies=[cluster_auth])
    async def join(body: JoinBody) -> dict:
        """Self-registration of an agent (`gpupool agent --join ...`).

        Same probe as the UI's manual add, so only a reachable, correctly-tokened agent can
        register. Servers the operator removed get 403 so removal sticks.
        """
        url = normalize_agent_url(body.agent_url)
        try:
            report = await poller.probe(url)
        except AgentError as e:
            raise HTTPException(502, f"agent at {url} answered {e.status}: {e.body}")
        except httpx.HTTPError as e:
            raise HTTPException(502, f"cannot reach agent at {url}: {type(e).__name__} {e}".strip())
        except (ValidationError, ValueError) as e:
            raise HTTPException(502, f"{url} did not return a valid agent report: {e}")
        node_id = report.node_id
        if store.is_removed(node_id):
            raise HTTPException(403, f"server {node_id} was removed in the UI; re-add it there")
        now = reconciler.clock()
        existing = store.get_server(node_id)
        if existing is not None:
            if existing.agent_url != url:  # e.g. the server's IP changed after a reboot
                store.add_server(ServerRecord(node_id=node_id, agent_url=url, added_at=existing.added_at))
                log.info("server %s moved: %s -> %s", node_id, existing.agent_url, url)
            return {"node_id": node_id, "agent_url": url, "added_at": existing.added_at, "new": False}
        rec = ServerRecord(node_id=node_id, agent_url=url, added_at=now)
        store.add_server(rec)
        report.agent_url = url
        store.upsert_node(report, now)
        try:
            notifier.emit("info", "server_added", f"Server {node_id} joined from {url}", node_id=node_id)
        except Exception:
            log.exception("emitting event failed")
        return {"node_id": node_id, "agent_url": url, "added_at": now, "new": True}

    def model_uses_file(name: str) -> bool:
        return any(m.source == f"coordinator://{name}" for m in store.list_models())

    app.include_router(make_files_router(library, require_bearer(cfg.cluster_token)))
    app.include_router(make_library_router(library, require_bearer(cfg.admin_key), model_uses_file))
    app.include_router(make_api_router(store=store, reconciler=reconciler, poller=poller,
                                       balancer=balancer, library=library, cfg=cfg,
                                       admin_dep=admin_auth, notifier=notifier,
                                       autoscaler=autoscaler))

    # ---- conversion
    def cluster_vram() -> ClusterVram:
        """Usable MB of the largest enabled CUDA GPU and of all of them, over live registered
        servers. Same selection and per-device number (usable_mb) as /api/capacity."""
        registered = {s.node_id for s in store.list_servers()}
        flags = store.gpu_flags()
        now = reconciler.clock()
        usable: list[int] = []
        for n in store.list_nodes():
            if n.report.node_id not in registered or not reconciler.node_alive(n, now):
                continue
            for d in n.report.devices:
                if d.kind == "cuda" and flags.get((n.report.node_id, gpu_key(d)), True):
                    usable.append(d.usable_mb)
        return ClusterVram(largest_gpu_mb=max(usable, default=0), pool_mb=sum(usable))

    async def default_inspect(spec: SourceSpec) -> InspectResult:
        return await convert_source.inspect_source(
            spec, hf=hf, locate_dir=library.locate_dir, cluster=cluster_vram(),
            supported_architectures=await toolchain.supported_architectures())

    app.include_router(make_convert_router(
        convert_manager, require_bearer(cfg.admin_key), cluster_vram,
        convert_inspect or default_inspect))

    # ---- admin
    @app.post("/admin/models", dependencies=[admin_auth])
    async def put_model(spec: ModelSpec) -> ModelSpec:
        # async, not a threadpool route: wake() sets an asyncio.Event, which is not thread-safe
        store.put_model(spec)
        reconciler.wake()  # act now rather than after reconcile_s
        return spec

    @app.delete("/admin/models/{name}", dependencies=[admin_auth])
    async def delete_model(name: str) -> dict:
        await drain_and_delete_model(store, reconciler, name)  # no event here, unlike /api
        reconciler.wake()
        return {"ok": True}

    @app.post("/admin/models/{name}/scale", dependencies=[admin_auth])
    async def scale(name: str, replicas: int) -> ModelSpec:  # async: see put_model
        if replicas < 0:
            raise HTTPException(422, "replicas must be >= 0")
        spec = spec_or_404(store, name).model_copy(update={"replicas": replicas})
        store.put_model(spec)
        reconciler.wake()
        return spec

    @app.post("/admin/deploy/{model}", dependencies=[admin_auth])
    async def deploy(model: str, dry_run: int = 0):
        spec = spec_or_404(store, model)
        if dry_run:
            try:
                return await reconciler.plan_for(spec)
            except Exception as e:
                raise plan_http_error(e) from e
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
                    "alive": reconciler.node_alive(n, now),
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
    # The per-request path must not parse every node report and model spec. The snapshot holds
    # only what routing needs and is rebuilt when the store version moved. Liveness is NOT in
    # it: a node that just went silent triggers no write, so it is judged on every call.
    snap: tuple = (-1, [], {}, {})  # (version, names, ready per model, NodeRecords)

    def _snapshot() -> tuple:
        nonlocal snap
        cur = snap
        v = store.version  # read BEFORE building: a racing write only causes one extra rebuild
        if cur[0] == v:
            return cur
        names = [m.name for m in store.list_models()]
        ready: dict[str, list[tuple[str, str, int, float]]] = {}
        for r in store.list_replicas(states={"ready"}):
            ready.setdefault(r.model, []).append(
                (r.replica_id, r.placement.head_node, r.placement.head_port,
                 r.placement.est_decode_tps or 0.0))
        nodes = {n.report.node_id: n for n in store.list_nodes()}
        snap = cur = (v, names, ready, nodes)
        return cur

    def get_candidates(model: str) -> list[ReplicaEndpoint]:
        now = reconciler.clock()
        _, _, ready, nodes = _snapshot()
        # Weight = the placement's estimated speed. Not llama-server's measured one: that is a rate
        # over the last scrape interval (its bucket resets on every /metrics read), 0 when idle and
        # lower when busy, and every change of a weight moves prefixes to a replica without them in
        # its KV cache. The estimate improves anyway as the speed model learns from those samples.
        rows = list(ready.get(model, ()))
        known = [tps for *_, tps in rows if tps > 0]
        default = sum(known) / len(known) if known else 1.0  # no estimate: an average replica
        out = []
        for replica_id, head_node, head_port, tps in rows:
            n = nodes.get(head_node)
            if n is None or not reconciler.node_alive(n, now):
                continue
            out.append(ReplicaEndpoint(
                replica_id=replica_id, model=model, base_url=f"http://{n.report.host}:{head_port}",
                weight=tps if tps > 0 else default))
        return out

    app.include_router(make_router(
        get_candidates=get_candidates,
        list_models=lambda: list(_snapshot()[1]),
        balancer=balancer, metrics=metrics, api_keys=cfg.api_keys,
        on_replica_error=reconciler.note_error,
        max_body_bytes=cfg.max_request_mb * 1024 * 1024,
        on_request=autoscaler.note_request, can_cold_start=autoscaler.can_cold_start,
        cold_start_timeout_s=cfg.cold_start_timeout_s,
    ))

    @app.get("/metrics")
    def render_metrics() -> PlainTextResponse:
        now = reconciler.clock()
        lines = [metrics.render().rstrip("\n")]
        lines.append("# TYPE gpupool_device_free_mb gauge")
        free, usable, alive = [], [], []
        for n in store.list_nodes():
            nid = prom_label_escape(n.report.node_id)
            alive.append(f'gpupool_node_alive{{node="{nid}"}} '
                         f'{int(reconciler.node_alive(n, now))}')
            for d in n.report.devices:
                lab = f'node="{nid}",device="{prom_label_escape(d.device_id)}"'
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
            for s in ALL_REPLICA_STATES:
                counts.setdefault((m.name, s), 0)
        for (m, s), c in sorted(counts.items()):
            lines.append(f'gpupool_replicas{{model="{prom_label_escape(m)}",state="{s}"}} {c}')
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    # The web UI. Mounted last: a mount at "/" would otherwise shadow the API routes.
    if UI_DIR.is_dir():
        app.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return app


def startup_banner(cfg: CoordinatorConfig, lan_ip: str | None = None) -> list[str]:
    """Plain lines telling the operator where the UI is and how to attach GPU servers."""
    ui = cfg.public_url.rstrip("/") if cfg.public_url else f"http://{lan_ip or detect_local_ip()}:{cfg.port}"
    join = f"{ui}#{cfg.cluster_token}"
    lines = [
        "=" * 64,
        "gpupool coordinator is ready",
        f"  UI:         {ui}",
        f"  Admin key:  {cfg.admin_key}",
        "",
        "Add a GPU server: run this on it (NVIDIA Container Toolkit required):",
        f"  docker run -d --name gpupool-agent --gpus all --network host --pid host "
        f"-v gpupool-agent:/data -e GPUPOOL_JOIN='{join}' ghcr.io/longduongbao29/gpupool-agent",
        "Without Docker:",
        f"  uv run gpupool agent --join '{join}' --llama-dir /path/to/llama.cpp/bin",
    ]
    if not cfg.api_keys:
        lines += ["",
                  "WARNING: the OpenAI-compatible API at /v1 accepts requests WITHOUT a key.",
                  "Set one with GPUPOOL_API_KEYS (comma-separated) or api_keys in the TOML."]
    if not cfg.public_url:
        lines += ["",
                  "If this IP is not reachable from your servers (e.g. the coordinator runs in",
                  "Docker and shows a container IP), use this machine's LAN IP instead, or set",
                  "GPUPOOL_PUBLIC_URL."]
    lines.append("=" * 64)
    return lines


def run_coordinator(cfg: CoordinatorConfig) -> None:
    import uvicorn

    # Secrets are filled in here, not in create_app: tests build apps with explicit (or empty,
    # i.e. open) keys and must keep that behaviour.
    admin_key, cluster_token = load_or_create_secrets(cfg.db_path, cfg.admin_key, cfg.cluster_token)
    cfg = cfg.model_copy(update={"admin_key": admin_key, "cluster_token": cluster_token})
    for line in startup_banner(cfg):
        log.info("%s", line)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
