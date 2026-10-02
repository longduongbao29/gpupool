import httpx
import pytest

from gpupool.coordinator.app import create_app
from gpupool.coordinator.library import Library
from gpupool.coordinator.store import ServerRecord, Store
from tests.test_coordinator_helpers import (
    META, SPEC, FakeClient, make_cfg, make_planner, node, put_replica,
)

CT = {"Authorization": "Bearer ctok"}
AD = {"Authorization": "Bearer adm"}


@pytest.fixture
async def env(tmp_path):
    cfg = make_cfg(tmp_path, cluster_token="ctok", admin_key="adm")
    cfg.models_dir.mkdir()
    (cfg.models_dir / "m.gguf").write_bytes(b"GGUFdata")
    (tmp_path / "secret.txt").write_text("secret")

    async def meta_for(spec):
        return META

    store = Store(":memory:")
    # Pull mode: only registered servers count, and /files serves library items only.
    store.add_server(ServerRecord(node_id="a", agent_url="http://a:7070", added_at=0.0))
    library = Library(":memory:", cfg.models_dir)
    library.add_path(str((cfg.models_dir / "m.gguf").resolve()))
    app = create_app(cfg, store=store, client=FakeClient(), meta_for=meta_for, start_background=False,
                     library=library)
    app.state.reconciler.planner = make_planner()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c, app, store, cfg


async def test_heartbeat_then_status(env):
    c, app, store, _ = env
    r = await c.post("/internal/heartbeat", json=node("a").model_dump(), headers=CT)
    assert r.status_code == 200
    put_replica(store, "m-1", now=app.state.reconciler.clock())
    store.put_model(SPEC)
    st = (await c.get("/admin/status", headers=AD)).json()
    assert st["nodes"][0]["node_id"] == "a" and st["nodes"][0]["alive"] is True
    assert st["nodes"][0]["devices"][0]["device_id"] == "CUDA0"
    assert st["models"][0]["name"] == "m"
    assert st["replicas"][0]["state"] == "ready" and st["replicas"][0]["outstanding"] == 0


async def test_auth(env):
    c, *_ = env
    assert (await c.post("/internal/heartbeat", json=node("a").model_dump())).status_code == 401
    assert (await c.post("/internal/heartbeat", json=node("a").model_dump(), headers=AD)).status_code == 401
    assert (await c.get("/admin/status")).status_code == 401
    assert (await c.get("/admin/status", headers=CT)).status_code == 401
    assert (await c.post("/admin/models", json=SPEC.model_dump())).status_code == 401
    assert (await c.get("/files/m.gguf")).status_code == 401


async def test_files_served_and_traversal_rejected(env):
    c, *_ = env
    r = await c.get("/files/m.gguf", headers=CT)
    assert r.status_code == 200 and r.content == b"GGUFdata"
    for bad in ["..", "../secret.txt", "..%2Fsecret.txt", "%2e%2e%2fsecret.txt", "..%5Csecret.txt",
                "sub/m.gguf", "nope.gguf"]:
        r = await c.get(f"/files/{bad}", headers=CT)
        # "/files/.." is normalised by the client to "/", which is the web UI page: fine, as
        # long as no file outside the library is ever returned.
        is_ui_page = r.status_code == 200 and b"<html" in r.content.lower()
        assert r.status_code in (400, 404) or is_ui_page, bad
        assert b"secret" not in r.content and b"GGUFdata" not in r.content


async def test_dry_run_has_no_side_effects(env):
    c, app, store, _ = env
    await c.post("/admin/models", json=SPEC.model_dump(), headers=AD)
    await c.post("/internal/heartbeat", json=node("a").model_dump(), headers=CT)
    r = await c.post("/admin/deploy/m?dry_run=1", headers=AD)
    assert r.status_code == 200 and r.json()["head_node"] == "a"
    assert store.list_replicas() == [] and app.state.reconciler._launches == {}
    r2 = await c.post("/admin/deploy/m?dry_run=1", headers=AD)
    assert r2.json()["head_port"] == r.json()["head_port"]


async def test_dry_run_nofit_is_409(env):
    c, app, *_ = env
    app.state.reconciler.planner = make_planner(est_mb=10**7)
    await c.post("/admin/models", json=SPEC.model_dump(), headers=AD)
    await c.post("/internal/heartbeat", json=node("a").model_dump(), headers=CT)
    assert (await c.post("/admin/deploy/m?dry_run=1", headers=AD)).status_code == 409
    assert (await c.post("/admin/deploy/ghost?dry_run=1", headers=AD)).status_code == 404


async def test_scale_delete_model_and_drain_replica(env):
    c, app, store, _ = env
    await c.post("/admin/models", json=SPEC.model_dump(), headers=AD)
    r = await c.post("/admin/models/m/scale?replicas=3", headers=AD)
    assert r.json()["replicas"] == 3 and store.get_model("m").replicas == 3
    put_replica(store, "m-1")
    assert (await c.delete("/admin/replicas/nope", headers=AD)).status_code == 404
    assert (await c.delete("/admin/models/m", headers=AD)).status_code == 200
    assert store.get_model("m") is None and store.get_replica("m-1").state == "draining"


async def test_router_candidates_and_models(env):
    c, app, store, _ = env
    await c.post("/admin/models", json=SPEC.model_dump(), headers=AD)
    await c.post("/internal/heartbeat", json=node("a").model_dump(), headers=CT)
    put_replica(store, "m-1", now=app.state.reconciler.clock())
    r = await c.get("/v1/models")
    assert r.status_code == 200 and "m" in r.text


async def test_metrics(env):
    c, app, store, _ = env
    await c.post("/admin/models", json=SPEC.model_dump(), headers=AD)
    await c.post("/internal/heartbeat", json=node("a").model_dump(), headers=CT)
    put_replica(store, "m-1", now=app.state.reconciler.clock())
    text = (await c.get("/metrics")).text
    assert 'gpupool_device_free_mb{node="a",device="CUDA0"} 8000' in text
    assert 'gpupool_device_usable_mb{node="a",device="CUDA0"} 7000' in text
    assert 'gpupool_node_alive{node="a"} 1' in text
    assert 'gpupool_replicas{model="m",state="ready"} 1' in text


async def test_heartbeat_from_unregistered_server_is_rejected(env):
    # A server removed in the UI must not re-register itself by pushing heartbeats.
    c, _, store, _ = env
    r = await c.post("/internal/heartbeat", json=node("zz").model_dump(), headers=CT)
    assert r.status_code == 403
    assert all(n.report.node_id != "zz" for n in store.list_nodes())


async def test_ui_and_healthz_served(env):
    c, *_ = env
    assert (await c.get("/healthz")).json() == {"ok": True}
    r = await c.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    # the UI mount at "/" must not shadow the API
    assert (await c.get("/api/state")).status_code == 401


async def test_meta_provider_sums_split_parts(tmp_path):
    # Reading only part 1 of a split GGUF would under-count layer bytes (silent VRAM under-estimate).
    from gpupool.coordinator.app import make_meta_provider
    from tests.test_scheduler_gguf_meta import _split_files
    cfg = make_cfg(tmp_path)
    cfg.models_dir.mkdir()
    whole, parts = _split_files(cfg.models_dir)
    lib = Library(":memory:", cfg.models_dir)
    lib._exec(
        "INSERT INTO library(name,path,source,bytes,downloaded,status,created_at)"
        " VALUES(?,?,'hf',1,1,'ready',0)", ("m-00001-of-00003.gguf", parts[0]))
    meta_for = make_meta_provider(cfg, lib)
    meta = await meta_for(SPEC.model_copy(update={"source": "coordinator://m-00001-of-00003.gguf"}))
    from gpupool.scheduler.gguf_meta import read_meta
    assert meta.layer_bytes == read_meta(whole).layer_bytes and sum(meta.layer_bytes) > 0
    lib._db.close()


@pytest.fixture
def routing(tmp_path, monkeypatch):
    """App built with start_background=False; captures the router's callables."""
    from gpupool.coordinator import app as app_module
    from tests.test_coordinator_helpers import Clock

    captured = {}
    real = app_module.make_router
    monkeypatch.setattr(app_module, "make_router", lambda **kw: captured.update(kw) or real(**kw))
    store = Store(":memory:")
    calls = {"nodes": 0, "models": 0}
    for name in ("list_nodes", "list_models"):
        orig = getattr(store, name)

        def wrap(orig=orig, key=name.split("_")[1]):
            calls[key] += 1
            return orig()
        monkeypatch.setattr(store, name, wrap)
    cfg = make_cfg(tmp_path)
    app = create_app(cfg, store=store, client=FakeClient(), start_background=False,
                     library=Library(":memory:", cfg.models_dir))
    clock = Clock(1000.0)
    app.state.reconciler.clock = clock
    return captured, store, calls, clock, cfg


def test_candidates_cached_until_a_write(routing):
    cap, store, calls, clock, _ = routing
    store.upsert_node(node("a"), clock())
    store.put_model(SPEC)
    put_replica(store, "m-1", now=clock())
    first = cap["get_candidates"]("m")
    assert [e.base_url for e in first] == ["http://10.0.0.1:9000"]
    n = dict(calls)
    assert cap["get_candidates"]("m") == first and cap["list_models"]() == ["m"]
    assert calls == n  # no second trip to the store
    put_replica(store, "m-2", now=clock(), head_port=9100)
    assert [e.base_url for e in cap["get_candidates"]("m")] == [
        "http://10.0.0.1:9000", "http://10.0.0.1:9100"]
    assert calls["nodes"] == n["nodes"] + 1
    store.set_replica_state("m-1", "draining")
    assert [e.replica_id for e in cap["get_candidates"]("m")] == ["m-2"]
    store.delete_model("m")
    assert cap["list_models"]() == []
    assert cap["get_candidates"]("other") == []


def test_stale_node_drops_out_without_any_write(routing):
    cap, store, calls, clock, cfg = routing
    store.upsert_node(node("a"), clock())
    put_replica(store, "m-1", now=clock())
    assert len(cap["get_candidates"]("m")) == 1
    n = dict(calls)
    clock.t += cfg.heartbeat_timeout_s + 1  # node died: no heartbeat, no store write
    assert cap["get_candidates"]("m") == []
    assert calls == n  # judged from the cached snapshot, liveness evaluated per call
    store.upsert_node(node("a"), clock())  # heartbeat resumes
    assert len(cap["get_candidates"]("m")) == 1
