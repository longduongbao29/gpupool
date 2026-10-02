"""Node agent HTTP API + heartbeat."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path

import httpx
import psutil
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from gpupool.agent.gpu import probe_devices
from gpupool.agent.models_cache import ensure_model, list_models
from gpupool.agent.procs import EngineExists, PortInUse, ProcessManager, llama_version
from gpupool.common.auth import bearer_headers, require_bearer
from gpupool.common.config import AgentConfig
from gpupool.common.models import EngineSpec, EngineStatus, NodeReport

log = logging.getLogger(__name__)


class EnsureBody(BaseModel):
    name: str
    source: str


def create_app(cfg: AgentConfig, pm: ProcessManager | None = None, probe=probe_devices,
               start_heartbeat: bool | None = None) -> FastAPI:
    pm = pm or ProcessManager(cfg.llama_dir, cfg.log_dir, cfg.host)
    version_cache: dict[str, str] = {}
    # Default follows cfg.push_heartbeat: the coordinator pulls /report, and a server
    # removed from the UI must not re-register itself by pushing.
    if start_heartbeat is None:
        start_heartbeat = cfg.push_heartbeat

    def version() -> str:
        if "v" not in version_cache:
            version_cache["v"] = llama_version(cfg.llama_dir)
        return version_cache["v"]

    def build_report() -> NodeReport:
        # Non-blocking (interval=None): /report is polled every 2 s per server.
        try:
            cpu_pct: float | None = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_used: int | None = int((vm.total - vm.available) // (1024 * 1024))
            ram_total: int | None = int(vm.total // (1024 * 1024))
        except Exception:
            cpu_pct = ram_used = ram_total = None
        return NodeReport(
            cpu_pct=cpu_pct, ram_used_mb=ram_used, ram_total_mb=ram_total,
            node_id=cfg.node_id, agent_url=f"http://{cfg.host}:{cfg.port}", host=cfg.host,
            devices=probe(cfg), engines=pm.list(), llama_version=version(),
            models=list_models(cfg.cache_dir), ts=time.time())

    async def heartbeat_loop() -> None:
        last_log = 0.0
        url = f"{cfg.coordinator_url.rstrip('/')}/internal/heartbeat"
        async with httpx.AsyncClient(timeout=5.0,
                                     headers=bearer_headers(cfg.cluster_token)) as client:
            while True:
                try:
                    report = await asyncio.to_thread(build_report)
                    r = await client.post(url, content=report.model_dump_json(),
                                          headers={"Content-Type": "application/json"})
                    r.raise_for_status()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    now = time.monotonic()
                    if now - last_log > 30:
                        last_log = now
                        log.warning("heartbeat to %s failed: %r", url, e)
                await asyncio.sleep(cfg.heartbeat_s)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        psutil.cpu_percent(interval=None)  # prime: the first call always returns 0.0
        await asyncio.to_thread(version)
        task = asyncio.create_task(heartbeat_loop()) if start_heartbeat else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            await asyncio.to_thread(pm.stop_all)

    app = FastAPI(title="gpupool-agent", lifespan=lifespan)
    app.state.pm = pm
    auth = Depends(require_bearer(cfg.cluster_token))

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/report", response_model=NodeReport, dependencies=[auth])
    def report():
        return build_report()

    @app.post("/engines", response_model=EngineStatus, dependencies=[auth])
    def start_engine(spec: EngineSpec):
        if spec.kind == "server":
            if not spec.model_path:
                raise HTTPException(422, "server engine requires model_path")
            if not Path(spec.model_path).is_file():
                raise HTTPException(422, f"model file not found: {spec.model_path}")
        elif len(spec.devices) != 1:
            raise HTTPException(422, "rpc engine needs exactly one device")
        try:
            return pm.start(spec, spec.model_path)
        except EngineExists:
            raise HTTPException(409, f"engine {spec.engine_id} already running")
        except PortInUse as e:
            raise HTTPException(422, str(e))
        except FileNotFoundError as e:
            raise HTTPException(500, str(e))

    @app.get("/engines/{engine_id}", response_model=EngineStatus, dependencies=[auth])
    def get_engine(engine_id: str):
        st = pm.get(engine_id)
        if st is None:
            raise HTTPException(404, "unknown engine")
        return st

    @app.delete("/engines/{engine_id}", response_model=EngineStatus, dependencies=[auth])
    def delete_engine(engine_id: str):
        st = pm.stop(engine_id)
        if st is None:
            raise HTTPException(404, "unknown engine")
        return st

    @app.post("/models/ensure", dependencies=[auth])
    async def models_ensure(body: EnsureBody):
        try:
            path, size = await asyncio.to_thread(
                ensure_model, body.name, body.source, cfg.cache_dir, cfg.coordinator_url,
                cfg.cluster_token)
        except FileNotFoundError as e:
            raise HTTPException(422, str(e))
        except (httpx.HTTPError, OSError, ValueError) as e:
            raise HTTPException(502, f"model download failed: {e}")
        return {"path": str(path), "bytes": size}

    return app


def run_agent(cfg: AgentConfig) -> None:
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)
