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
                        lambda s, b, h, m, *_: [sys.executable, "-c", LISTENER.format(port=port)])
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


class _FakePM:
    """Records start() calls; the API gate is what is under test, not the process."""

    def __init__(self):
        self.started = []

    def start(self, spec, model_path):
        from gpupool.common.models import EngineStatus
        self.started.append((spec, model_path))
        return EngineStatus(engine_id=spec.engine_id, kind=spec.kind, state="starting",
                            port=spec.port)

    def list(self):
        return []

    def stop_all(self):
        pass


SERVER = {"engine_id": "h", "kind": "server", "port": 1, "devices": ["CUDA0"], "model": "m"}


async def test_extra_args_rejected(cfg):
    pm = _FakePM()
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    async with client(app) as c:
        r = await c.post("/engines", json={"engine_id": "r", "kind": "rpc", "port": 1,
                                           "devices": ["CPU"], "extra_args": ["--evil"]},
                         headers=H)
        assert r.status_code == 422 and "extra_args" in r.json()["detail"]
    assert pm.started == []


async def test_server_model_path_inside_cache_accepted(cfg):
    cfg.cache_dir.mkdir()
    f = cfg.cache_dir / "m.gguf"
    f.write_bytes(b"x")
    pm = _FakePM()
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    async with client(app) as c:
        r = await c.post("/engines", json={**SERVER, "model_path": str(f)}, headers=H)
    assert r.status_code == 200 and pm.started[0][1] == str(f)


async def test_server_model_path_outside_cache_rejected(cfg, tmp_path):
    cfg.cache_dir.mkdir()
    outside = tmp_path / "secret.gguf"
    outside.write_bytes(b"x")
    pm = _FakePM()
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    async with client(app) as c:
        r = await c.post("/engines", json={**SERVER, "model_path": str(outside)}, headers=H)
        assert r.status_code == 422
        # a ../ escape out of the cache must not pass as "inside"
        sneaky = str(cfg.cache_dir / ".." / "secret.gguf")
        r = await c.post("/engines", json={**SERVER, "model_path": sneaky}, headers=H)
        assert r.status_code == 422
    assert pm.started == []


async def test_server_model_path_from_ensure_accepted(cfg, tmp_path):
    local = tmp_path / "local.gguf"  # absolute-file source: outside the cache
    local.write_bytes(b"12345")
    pm = _FakePM()
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    async with client(app) as c:
        r = await c.post("/engines", json={**SERVER, "model_path": str(local)}, headers=H)
        assert r.status_code == 422  # not ensured yet
        await c.post("/models/ensure", json={"name": "m", "source": str(local)}, headers=H)
        r = await c.post("/engines", json={**SERVER, "model_path": str(local)}, headers=H)
        assert r.status_code == 200
    assert len(pm.started) == 1


async def test_draft_model_path_allowlist(cfg, tmp_path):
    cfg.cache_dir.mkdir()
    main = cfg.cache_dir / "m.gguf"
    main.write_bytes(b"x")
    draft = cfg.cache_dir / "d.gguf"
    draft.write_bytes(b"x")
    outside = tmp_path / "secret.gguf"
    outside.write_bytes(b"x")
    pm = _FakePM()
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    spec = {**SERVER, "model_path": str(main), "spec_type": "draft", "draft_device": "CUDA0"}
    async with client(app) as c:
        r = await c.post("/engines", json={**spec, "draft_model_path": str(outside)}, headers=H)
        assert r.status_code == 422 and "draft_model_path" in r.json()["detail"]
        r = await c.post("/engines", json={**spec, "draft_model_path": str(cfg.cache_dir / "x.gguf")},
                         headers=H)
        assert r.status_code == 422 and "not found" in r.json()["detail"]
        assert pm.started == []
        r = await c.post("/engines", json={**spec, "draft_model_path": str(draft)}, headers=H)
        assert r.status_code == 200
        # a draft outside the cache is fine once /models/ensure handed it out
        await c.post("/models/ensure", json={"name": "s", "source": str(outside)}, headers=H)
        r = await c.post("/engines", json={**spec, "engine_id": "h2",
                                           "draft_model_path": str(outside)}, headers=H)
        assert r.status_code == 200
    assert len(pm.started) == 2


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
                        lambda s, b, h, m, *_: [sys.executable, "-c", LISTENER.format(port=port)])
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


async def test_engine_memory_endpoint(cfg, monkeypatch, tmp_path):
    port = free_port()
    monkeypatch.setattr(procs, "build_command",
                        lambda s, b, h, m, *_: [sys.executable, "-c", LISTENER.format(port=port)])
    monkeypatch.setattr(ProcessManager, "_binaries", lambda self: BINS)
    pm = ProcessManager(tmp_path, cfg.log_dir, "127.0.0.1")
    app = create_app(cfg, pm=pm, start_heartbeat=False)
    spec = {"engine_id": "m1", "kind": "rpc", "port": port, "devices": ["CPU"]}
    try:
        async with client(app) as c:
            assert (await c.get("/engines/m1/memory")).status_code == 401
            assert (await c.get("/engines/zzz/memory", headers=H)).status_code == 404
            assert (await c.post("/engines", json=spec, headers=H)).status_code == 200
            log = cfg.log_dir / "m1.log"
            # The child writes its own 100 lines after binding; appending before it is done lets
            # its later writes land over ours (flaky under a loaded CI machine). Wait for them.
            for _ in range(200):
                if log.is_file() and "line 99" in log.read_text(encoding="utf-8", errors="replace"):
                    break
                await asyncio.sleep(0.05)
            # the child owns the log; append the lines llama-server would print at load
            with open(log, "a", encoding="utf-8") as f:
                f.write("0.00.847.109 I load_tensors:        CUDA0 model buffer size =   373.73 MiB\n"
                        "0.01.024.240 I llama_kv_cache:      CUDA0 KV buffer size =    24.00 MiB\n"
                        "0.00.847.107 I load_tensors:   CPU_Mapped model buffer size =    89.26 MiB\n")
            r = (await c.get("/engines/m1/memory", headers=H)).json()
            assert r["engine_id"] == "m1"
            assert r["devices"] == {"CUDA0": {"model_mb": 373.73, "kv_mb": 24.0,
                                              "compute_mb": 0.0, "total_mb": 397.73}}
            log.write_text("nothing useful\n")
            assert (await c.get("/engines/m1/memory", headers=H)).json()["devices"] == {}
    finally:
        pm.stop_all()


async def test_rpc_engine_device_list_and_features(cfg, monkeypatch, tmp_path):
    import gpupool.agent.app as appmod
    monkeypatch.setattr(appmod, "llama_version", lambda d: "b11413")
    seen = []
    monkeypatch.setattr(ProcessManager, "start", lambda self, spec, mp=None: seen.append(spec) or
                        procs.EngineStatus(engine_id=spec.engine_id, kind=spec.kind, state="starting",
                                           port=spec.port))
    app = create_app(cfg, start_heartbeat=False)
    async with client(app) as c:
        two = {"engine_id": "e1", "kind": "rpc", "port": 9123, "devices": ["CUDA0", "CUDA1"]}
        assert (await c.post("/engines", json=two, headers=H)).status_code == 200
        for bad in ([], ["CUDA0", "CUDA0"]):
            r = await c.post("/engines", json={**two, "engine_id": "e2", "devices": bad}, headers=H)
            assert r.status_code == 422, bad
        assert (await c.get("/report", headers=H)).json()["features"] == [
            "kv_unified", "rpc_multi_device", "spec_mtp"]
    assert seen[0].devices == ["CUDA0", "CUDA1"]


@pytest.mark.parametrize("version, features", [
    ("b11413", ["kv_unified", "rpc_multi_device", "spec_mtp"]), ("b11342", ["kv_unified", "rpc_multi_device", "spec_mtp"]),
    ("b9000", []), ("unknown", []), ("x1", [])])
def test_features_follow_the_llama_build(version, features):
    from gpupool.agent.app import features_of
    assert features_of(version) == features


def test_old_engine_logs_are_pruned_but_not_those_of_known_engines(tmp_path):
    import os
    import time as _time
    from gpupool.agent.procs import ProcessManager
    pm = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1")
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    now = _time.time()
    for i in range(6):  # m-0 newest ... m-5 oldest
        p = logs / f"m-{i}-head.log"
        p.write_text("x")
        os.utime(p, (now - i * 60, now - i * 60))
    old = logs / "ancient-head.log"
    old.write_text("x")
    os.utime(old, (now - 30 * 86400, now - 30 * 86400))
    pm._engines["m-5-head"] = object()  # still known to the agent: kept whatever its age
    assert pm.prune_logs(keep=3, now=now) == 3  # m-3, m-4 (past the 3 newest) and the 30-day-old one
    assert sorted(p.name for p in logs.glob("*.log")) == ["m-0-head.log", "m-1-head.log", "m-2-head.log",
                                                          "m-5-head.log"]
