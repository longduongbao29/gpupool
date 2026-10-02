from __future__ import annotations

import asyncio
import json
import sys

import httpx
import pytest
import respx

from gpupool.agent import procs
from gpupool.agent.app import create_app
from gpupool.agent.procs import ProcessManager
from gpupool.common.config import AgentConfig

from tests.test_agent_unit import BINS, LISTENER, free_port  # noqa: F401

FAKE = json.dumps([{"device_id": "CUDA0", "kind": "cuda", "name": "g", "total_mb": 4000,
                    "free_mb": 3500}])
H = {"Authorization": "Bearer secret"}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUPOOL_FAKE_DEVICES", FAKE)
    monkeypatch.setattr(procs, "llama_version", lambda d: "b1")
    import gpupool.agent.app as appmod
    monkeypatch.setattr(appmod, "llama_version", lambda d: "b1")
    return AgentConfig(node_id="n1", port=7071, host="127.0.0.1", llama_dir=tmp_path, cluster_token="secret",
                       cache_dir=tmp_path / "cache", log_dir=tmp_path / "logs",
                       coordinator_url="http://coord:8080", heartbeat_s=0.05)


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_auth_and_health(cfg):
    app = create_app(cfg, start_heartbeat=False)
    async with client(app) as c:
        assert (await c.get("/health")).json() == {"ok": True}
        assert (await c.get("/report")).status_code == 401
        assert (await c.get("/report", headers={"Authorization": "Bearer no"})).status_code == 401
        assert (await c.post("/engines", json={})).status_code == 401


async def test_report(cfg):
    (cfg.cache_dir).mkdir()
    (cfg.cache_dir / "a.gguf").write_text("x")
    (cfg.cache_dir / "b.gguf.part").write_text("x")
    app = create_app(cfg, start_heartbeat=False)
    async with client(app) as c:
        r = (await c.get("/report", headers=H)).json()
    assert r["node_id"] == "n1" and r["agent_url"] == "http://127.0.0.1:7071"
    assert r["llama_version"] == "b1" and r["models"] == ["a.gguf"]
    assert r["devices"][0]["usable_mb"] == 2988 and r["engines"] == []


async def test_engine_endpoints(cfg, monkeypatch, tmp_path):
    port = free_port()
    monkeypatch.setattr(procs, "build_command",
                        lambda s, b, h, m: [sys.executable, "-c", LISTENER.format(port=port)])
    monkeypatch.setattr(ProcessManager, "_binaries", lambda self: BINS)
    pm = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1")
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    spec = {"engine_id": "e1", "kind": "rpc", "port": port, "devices": ["CPU"]}
    try:
        async with client(app) as c:
            r = await c.post("/engines", json=spec, headers=H)
            assert r.status_code == 200 and r.json()["engine_id"] == "e1"
            assert (await c.post("/engines", json=spec, headers=H)).status_code == 409
            for _ in range(100):
                st = (await c.get("/engines/e1", headers=H)).json()
                if st["state"] == "running":
                    break
                await asyncio.sleep(0.1)
            assert st["state"] == "running"
            # Only once e1 is running has its child bound the port; checking earlier races the bind.
            assert (await c.post("/engines", json={**spec, "engine_id": "e2"},
                                 headers=H)).status_code in (409, 422)
            assert (await c.get("/engines/zzz", headers=H)).status_code == 404
            r = await c.delete("/engines/e1", headers=H)
            assert r.json()["state"] == "exited"
            assert (await c.delete("/engines/zzz", headers=H)).status_code == 404
    finally:
        pm.stop_all()


async def test_server_validation(cfg):
    app = create_app(cfg, start_heartbeat=False)
    base = {"engine_id": "h", "kind": "server", "port": 1, "devices": ["CUDA0"], "model": "m"}
    async with client(app) as c:
        assert (await c.post("/engines", json=base, headers=H)).status_code == 422
        r = await c.post("/engines", json={**base, "model_path": "/nope.gguf"}, headers=H)
        assert r.status_code == 422


async def test_models_ensure(cfg, tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"12345")
    app = create_app(cfg, start_heartbeat=False)
    async with client(app) as c:
        r = await c.post("/models/ensure", json={"name": "m", "source": str(f)}, headers=H)
        assert r.json() == {"path": str(f), "bytes": 5}
        r = await c.post("/models/ensure", json={"name": "m", "source": "/nope.gguf"}, headers=H)
        assert r.status_code == 422


async def test_heartbeat_posts_and_survives_errors(cfg):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"ok": True})

    app = create_app(cfg, start_heartbeat=True)
    with respx.mock(assert_all_called=False) as mock:
        mock.post("http://coord:8080/internal/heartbeat").mock(side_effect=handler)
        async with app.router.lifespan_context(app):
            for _ in range(100):
                if len(calls) >= 3:
                    break
                await asyncio.sleep(0.05)
    assert len(calls) >= 3  # kept going after the 500
    assert calls[0].headers["authorization"] == "Bearer secret"
    assert json.loads(calls[-1].content)["node_id"] == "n1"


async def test_shutdown_stops_engines(cfg, monkeypatch, tmp_path):
    port = free_port()
    monkeypatch.setattr(procs, "build_command",
                        lambda s, b, h, m: [sys.executable, "-c", LISTENER.format(port=port)])
    monkeypatch.setattr(ProcessManager, "_binaries", lambda self: BINS)
    pm = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1")
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    async with app.router.lifespan_context(app):
        pm.start(__import__("gpupool.common.models", fromlist=["EngineSpec"]).EngineSpec(
            engine_id="e1", kind="rpc", port=port, devices=["CPU"]))
        proc = pm._engines["e1"].proc  # stop() forgets the entry, keep the process
    assert proc.poll() is not None and pm.list() == []


async def test_heartbeat_off_by_default_and_report_has_host_telemetry(cfg):
    assert cfg.push_heartbeat is False
    app = create_app(cfg)
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://coord:8080/internal/heartbeat").mock(
            return_value=httpx.Response(200, json={"ok": True}))
        async with app.router.lifespan_context(app):
            async with client(app) as c:
                r = await c.get("/report", headers=H)
            await asyncio.sleep(0.3)  # several heartbeat periods (0.05 s) would have fired
        assert route.call_count == 0
    j = r.json()
    assert j["ram_total_mb"] > 0 and 0 <= j["ram_used_mb"] <= j["ram_total_mb"]
    assert j["cpu_pct"] is not None


async def test_heartbeat_starts_when_push_enabled(cfg):
    app = create_app(cfg.model_copy(update={"push_heartbeat": True}))
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post("http://coord:8080/internal/heartbeat").mock(
            return_value=httpx.Response(200, json={"ok": True}))
        async with app.router.lifespan_context(app):
            for _ in range(100):
                if route.call_count:
                    break
                await asyncio.sleep(0.05)
        assert route.call_count >= 1
