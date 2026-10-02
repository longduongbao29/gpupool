import httpx
import pytest

from gpupool.coordinator.app import create_app
from gpupool.coordinator.store import Store
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
    app = create_app(cfg, store=store, client=FakeClient(), meta_for=meta_for, start_background=False)
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
        assert r.status_code in (400, 404), bad
        assert b"secret" not in r.content


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
