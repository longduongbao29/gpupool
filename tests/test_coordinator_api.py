import httpx
import pytest
from fastapi import Depends, FastAPI

from gpupool.common.auth import require_bearer
from gpupool.common.models import LibraryItem
from gpupool.coordinator.agent_client import AgentError
from gpupool.coordinator.api import make_api_router
from gpupool.coordinator.store import ServerRecord
from gpupool.router.balancer import Balancer
from tests.test_coordinator_helpers import (
    SPEC, FakeAutoscaler, dev, make_cfg, make_planner, make_reconciler, node, put_replica,
)

AD = {"Authorization": "Bearer adm"}


def item(name="x.gguf", status="ready"):
    return LibraryItem(name=name, path="/m/" + name, source="path", status=status, created_at=1.0)


class FakeLibrary:
    def __init__(self, *items):
        self.items = {i.name: i for i in items}

    def list(self):
        return list(self.items.values())

    def get(self, name):
        return self.items.get(name)


class FakePoller:
    def __init__(self):
        self.reports = {}  # url -> NodeReport | Exception

    async def probe(self, url):
        r = self.reports[url]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
async def env(request):
    cfg = make_cfg(admin_key="adm", cluster_token="ctok", public_url="http://coord:8080", api_keys=["k"])
    rec, store, clock = make_reconciler(cfg=cfg)
    poller, balancer, lib = FakePoller(), Balancer(), FakeLibrary(item(), item("big.gguf", "downloading"))
    app = FastAPI()
    app.include_router(make_api_router(store=store, reconciler=rec, poller=poller, balancer=balancer,
                                       library=lib, cfg=cfg, admin_dep=Depends(require_bearer("adm")),
                                       autoscaler=getattr(request, "param", None)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", headers=AD) as c:
        yield c, store, rec, poller, clock, lib
    await rec.shutdown()


def register(store, clock, *nodes):
    for n in nodes:
        store.add_server(ServerRecord(node_id=n.node_id, agent_url=n.agent_url, added_at=clock()))
        store.upsert_node(n, clock())


async def test_auth_required_on_every_route(env):
    c, *_ = env
    for method, path in [("GET", "/api/state"), ("POST", "/api/servers"), ("DELETE", "/api/servers/a"),
                         ("PUT", "/api/servers/a/gpus/CUDA0"), ("PUT", "/api/models/m"),
                         ("POST", "/api/models/m/start"), ("POST", "/api/models/m/stop"),
                         ("DELETE", "/api/models/m"), ("POST", "/api/models/m/plan"),
                         ("GET", "/api/events"), ("POST", "/api/events/read"),
                         ("GET", "/api/capacity"), ("POST", "/api/recommend")]:
        r = await c.request(method, path, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401, (method, path)


# ---------------------------------------------------------------- servers
async def test_add_server_happy_path_normalizes_url(env):
    c, store, rec, poller, clock, _ = env
    poller.reports["http://10.0.0.1:7070"] = node("a")
    r = await c.post("/api/servers", json={"agent_url": " http://10.0.0.1:7070/ "})
    assert r.status_code == 200
    j = r.json()
    assert j["node_id"] == "a" and j["agent_url"] == "http://10.0.0.1:7070" and j["alive"] is True
    assert j["report"]["node_id"] == "a" and j["gpu_enabled"] == {"CUDA0": True} and j["last_seen"] == clock()
    assert store.get_server("a") is not None and len(store.list_nodes()) == 1
    assert store.list_events()[0].kind == "server_added"
    # same node again -> 409
    r = await c.post("/api/servers", json={"agent_url": "http://10.0.0.1:7070"})
    assert r.status_code == 409


async def test_add_server_errors(env):
    c, store, rec, poller, *_ = env
    poller.reports["http://bad:1"] = AgentError(401, "invalid", "http://bad:1/report")
    r = await c.post("/api/servers", json={"agent_url": "http://bad:1"})
    assert r.status_code == 400 and "check the cluster token" in r.json()["detail"]
    poller.reports["http://down:1"] = httpx.ConnectError("refused")
    r = await c.post("/api/servers", json={"agent_url": "http://down:1"})
    assert r.status_code == 400 and "cannot reach" in r.json()["detail"]
    poller.reports["http://err:1"] = AgentError(500, "boom", "x")
    assert (await c.post("/api/servers", json={"agent_url": "http://err:1"})).status_code == 400
    for bad in ["ftp://x", "10.0.0.1:7070", "", "http://"]:
        r = await c.post("/api/servers", json={"agent_url": bad})
        assert r.status_code == 400, bad
    assert store.list_servers() == []


async def test_delete_server_and_404(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a"))
    put_replica(store, "m-1", head="a", now=clock())
    assert (await c.delete("/api/servers/nope")).status_code == 404
    assert (await c.delete("/api/servers/a")).status_code == 200
    assert store.get_server("a") is None and store.get_replica("m-1").state == "stopped"
    assert store.list_events()[0].kind == "server_removed"


async def test_gpu_switch(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a"))
    r = await c.put("/api/servers/a/gpus/CUDA0", json={"enabled": False})
    assert r.status_code == 200 and store.gpu_flags() == {("a", "CUDA0"): False}
    assert (await c.put("/api/servers/zzz/gpus/CUDA0", json={"enabled": True})).status_code == 404
    assert (await c.put("/api/servers/a/gpus/CUDA0", json={})).status_code == 422
    st = (await c.get("/api/state")).json()
    assert st["servers"][0]["gpu_enabled"] == {"CUDA0": False}


async def test_set_gpu_stores_uuid_key_and_ui_stays_device_id(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a", devices=[dev("CUDA0", uuid="GPU-0"), dev("CUDA1", uuid="GPU-1")]))
    woke = []
    rec.wake = lambda: woke.append(1)
    r = await c.put("/api/servers/a/gpus/CUDA1", json={"enabled": False})
    assert r.json() == {"node_id": "a", "device_id": "CUDA1", "enabled": False}
    assert store.gpu_flags() == {("a", "GPU-1"): False} and woke
    st = (await c.get("/api/state")).json()
    assert st["servers"][0]["gpu_enabled"] == {"CUDA0": True, "CUDA1": False}
    assert st["summary"]["gpus_enabled"] == 1
    # the card moves to CUDA0: the flag follows it
    store.upsert_node(node("a", devices=[dev("CUDA0", uuid="GPU-1")]), clock())
    assert (await c.get("/api/state")).json()["servers"][0]["gpu_enabled"] == {"CUDA0": False}
    # device_id not in the report: stored as given
    await c.put("/api/servers/a/gpus/CUDA7", json={"enabled": False})
    assert store.gpu_flags()[("a", "CUDA7")] is False


# ---------------------------------------------------------------- state
async def test_state_shape_and_summary(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a", devices=[dev("CUDA0"), dev("CUDA1")]), node("b"))
    cpu = node("c")
    cpu.devices[0].kind = "cpu"
    register(store, clock, cpu)
    store.add_server(ServerRecord(node_id="never", agent_url="http://n:1", added_at=clock() + 100))  # no report yet
    store.upsert_node(node("ghost"), clock())  # unregistered: invisible
    store.set_gpu_enabled("a", "CUDA1", False)
    clock.t += 11  # everybody ages out...
    store.upsert_node(node("a", devices=[dev("CUDA0"), dev("CUDA1")]), clock())  # ...except a
    st = (await c.get("/api/state")).json()
    assert set(st) == {"summary", "servers", "models", "library", "settings", "events", "unread_events"}
    assert [s["node_id"] for s in st["servers"]] == ["a", "b", "c", "never"]
    never = st["servers"][3]
    assert never["alive"] is False and never["report"] is None and never["last_seen"] == 0.0
    assert never["gpu_enabled"] == {}
    assert st["servers"][1]["alive"] is False
    assert st["summary"] == {
        "servers_total": 4, "servers_online": 1,
        "gpus_total": 3, "gpus_enabled": 2,  # a:2 cuda, b:1 cuda, c is cpu
        "pool_total_mb": 10000, "pool_usable_mb": 7000,  # enabled cuda of alive servers: a/CUDA0
        "models_running": 0}
    assert st["settings"] == {"public_url": "http://coord:8080", "cluster_token": "ctok", "api_keys_set": True}
    assert {i["name"] for i in st["library"]} == {"x.gguf", "big.gguf"}


def model_state(st, name="m"):
    return next(m for m in st["models"] if m["spec"]["name"] == name)


@pytest.mark.parametrize("desired,replicas,expected,error", [
    (1, [("ready", None)], "running", None),
    (1, [("launching", None)], "starting", None),
    (1, [("pending", None)], "starting", None),
    (1, [("failed", "boom")], "failed", "boom"),
    (1, [("failed", "old"), ("launching", None)], "starting", "old"),  # retry shows the last error
    (1, [("failed", "old"), ("ready", None)], "running", None),
    (1, [], "starting", None),  # nothing yet: the next tick launches it
    (0, [("draining", None)], "stopping", None),
    (0, [("ready", None)], "stopping", None),
    (0, [("launching", None)], "stopping", None),
    (0, [("stopped", None)], "stopped", None),
    (0, [("failed", "x")], "stopped", None),
    (0, [], "stopped", None),
])
async def test_model_state_derivation(env, desired, replicas, expected, error):
    c, store, rec, _, clock, _ = env
    store.put_model(SPEC.model_copy(update={"replicas": desired}))
    for i, (state, err) in enumerate(replicas):
        put_replica(store, f"m-{i}", state=state, now=clock() + i)
        if err:
            store.set_replica_state(f"m-{i}", "failed", err)
    m = model_state((await c.get("/api/state")).json())
    assert m["state"] == expected and m["error"] == error


async def test_model_file_replicas_listing_and_outstanding(env):
    c, store, rec, _, clock, _ = env
    store.put_model(SPEC.model_copy(update={"source": "coordinator://x.gguf", "replicas": 1}))
    put_replica(store, "m-old", state="failed", now=1.0)
    put_replica(store, "m-new", state="failed", now=2.0, head_port=9001)
    put_replica(store, "m-live", state="ready", now=3.0, head_port=9002)
    put_replica(store, "m-gone", state="stopped", now=4.0, head_port=9003)
    st = (await c.get("/api/state")).json()
    m = model_state(st)
    assert m["file"] == "x.gguf"
    assert [r["replica_id"] for r in m["replicas"]] == ["m-new", "m-live"]  # active + newest failed only
    assert all(r["outstanding"] == 0 for r in m["replicas"])
    assert model_state(st)["spec"]["source"] == "coordinator://x.gguf"


async def test_file_is_null_for_non_library_source(env):
    c, store, *_ = env
    store.put_model(SPEC)
    assert model_state((await c.get("/api/state")).json())["file"] is None


async def test_nofit_error_surfaces_on_model(env):
    c, store, rec, _, clock, _ = env
    rec.planner = make_planner(est_mb=10**6)
    register(store, clock, node("a"))
    store.put_model(SPEC.model_copy(update={"replicas": 1}))
    await rec.tick()
    m = model_state((await c.get("/api/state")).json())
    assert m["state"] == "failed" and "nothing fits" in m["error"]


async def test_running_counts_in_summary(env):
    c, store, rec, _, clock, _ = env
    store.put_model(SPEC)
    put_replica(store, "m-1", state="ready", now=clock())
    assert (await c.get("/api/state")).json()["summary"]["models_running"] == 1


# ---------------------------------------------------------------- models
async def test_put_model_create_update_validation(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a"))
    r = await c.put("/api/models/qwen", json={"file": "x.gguf"})
    assert r.status_code == 200
    assert r.json() == {"name": "qwen", "source": "coordinator://x.gguf", "ctx_size": 4096,
                        "parallel": 1, "replicas": 0, "pin_devices": [], "priority": 50, "spread": "gpu",
                        "min_replicas": None, "max_replicas": None, "autoscale": None, "idle_unload_s": None}
    store.put_model(store.get_model("qwen").model_copy(update={"replicas": 2}))
    r = await c.put("/api/models/qwen", json={"file": "x.gguf", "ctx_size": 8192, "parallel": 2,
                                              "pin_devices": ["a/CUDA0", "a/CUDA0"]})
    assert r.status_code == 200 and r.json()["replicas"] == 2  # update keeps replicas
    assert r.json()["ctx_size"] == 8192 and r.json()["pin_devices"] == ["a/CUDA0"]
    assert (await c.put("/api/models/qwen", json={"file": "nope.gguf"})).status_code == 422
    assert (await c.put("/api/models/qwen", json={"file": "big.gguf"})).status_code == 422  # not ready
    for pins in (["a"], ["a/"], ["/CUDA0"], ["a/b/c"], ["zzz/CUDA0"]):
        r = await c.put("/api/models/qwen", json={"file": "x.gguf", "pin_devices": pins})
        assert r.status_code == 422, pins
    for name in ("bad name", "a" * 65, "wé"):
        assert (await c.put(f"/api/models/{name}", json={"file": "x.gguf"})).status_code == 422, name
    assert (await c.put("/api/models/q", json={"file": "x.gguf", "ctx_size": 0})).status_code == 422
    assert store.get_model("qwen").pin_devices == ["a/CUDA0"]


async def test_start_stop_delete_flow(env):
    c, store, rec, _, clock, _ = env
    rec.planner = make_planner()
    woke = []
    rec.wake = lambda: woke.append(1)
    register(store, clock, node("a"))
    await c.put("/api/models/m", json={"file": "x.gguf"})
    woke.clear()
    assert (await c.post("/api/models/m/start", json={"replicas": 0})).status_code == 422
    r = await c.post("/api/models/m/start")  # body optional
    assert r.status_code == 200 and r.json()["replicas"] == 1
    assert woke == [1]  # start only wakes the loop; the placement happens on the next tick
    assert store.list_replicas(model="m") == []
    await rec.tick()
    assert store.list_replicas(model="m")
    r = await c.post("/api/models/m/start", json={"replicas": 3})
    assert r.json()["replicas"] == 3 and store.get_model("m").replicas == 3
    r = await c.post("/api/models/m/stop")
    assert r.json()["replicas"] == 0 and store.get_model("m").replicas == 0
    store.put_model(store.get_model("m").model_copy(update={"replicas": 1}))
    put_replica(store, "m-live", model="m", state="ready", now=clock(), head_port=9100)
    assert (await c.delete("/api/models/m")).status_code == 200
    assert store.get_model("m") is None and store.get_replica("m-live").state == "draining"
    ks = [e.kind for e in store.list_events(limit=50)]
    assert "model_started" in ks and "model_stopped" in ks
    await rec.shutdown()


async def test_start_does_not_wait_for_a_tick(env):
    c, store, rec, *_ = env
    store.put_model(SPEC)

    async def boom():
        raise AssertionError("start must not run a tick inside the request")

    rec.tick = boom
    r = await c.post("/api/models/m/start")
    assert r.status_code == 200 and store.get_model("m").replicas == 1
    assert rec._wake.is_set()


async def test_stop_delete_put_model_wake_the_loop(env):
    c, store, rec, *_ = env
    store.put_model(SPEC)
    for call in (lambda: c.post("/api/models/m/stop"), lambda: c.put("/api/models/m", json={"file": "x.gguf"}),
                 lambda: c.delete("/api/models/m")):
        rec._wake.clear()
        assert (await call()).status_code == 200
        assert rec._wake.is_set()


async def test_plan_dry_run_and_nofit(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a"))
    store.put_model(SPEC)
    r = await c.post("/api/models/m/plan")
    assert r.status_code == 200 and r.json()["head_node"] == "a" and store.list_replicas() == []
    rec.planner = make_planner(est_mb=10**7)
    r = await c.post("/api/models/m/plan")
    assert r.status_code == 409 and "nothing fits" in r.json()["detail"]


async def test_unknown_model_404_everywhere(env):
    c, *_ = env
    for method, path in [("POST", "/api/models/ghost/start"), ("POST", "/api/models/ghost/stop"),
                         ("DELETE", "/api/models/ghost"), ("POST", "/api/models/ghost/plan")]:
        assert (await c.request(method, path)).status_code == 404, path


async def test_pin_affects_plan_via_api(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a"), node("b"))
    await c.put("/api/models/m", json={"file": "x.gguf", "pin_devices": ["b/CUDA0"]})
    r = await c.post("/api/models/m/plan")
    assert r.json()["head_node"] == "b"


# ---------------------------------------------------------------- events
async def test_events_endpoints_and_unread(env):
    c, store, rec, *_ = env
    ids = [rec.notifier.emit("info", "k", f"e{i}").id for i in range(3)]
    st = (await c.get("/api/state")).json()
    assert st["unread_events"] == 3 and [e["message"] for e in st["events"]] == ["e2", "e1", "e0"]
    r = (await c.get("/api/events?limit=2")).json()
    assert [e["message"] for e in r["events"]] == ["e2", "e1"] and r["unread"] == 3
    r = (await c.get(f"/api/events?after_id={ids[0]}")).json()
    assert [e["id"] for e in r["events"]] == [ids[2], ids[1]]
    r = await c.post("/api/events/read", json={"up_to_id": ids[1]})
    assert r.json() == {"unread": 1}
    assert (await c.get("/api/state")).json()["unread_events"] == 1
    assert (await c.post("/api/events/read", json={})).status_code == 422


# ---------------------------------------------------------------- policy fields, capacity, recommend
async def test_put_model_policy_fields_stored_and_defaulted(env):
    c, store, *_ = env
    r = await c.put("/api/models/q", json={"file": "x.gguf", "priority": 80, "spread": "node"})
    assert r.status_code == 200 and r.json()["priority"] == 80 and r.json()["spread"] == "node"
    r = await c.put("/api/models/q", json={"file": "x.gguf", "ctx_size": 8192})  # omitted: keep stored
    assert r.json()["priority"] == 80 and r.json()["spread"] == "node"
    r = await c.put("/api/models/q", json={"file": "x.gguf", "priority": 0, "spread": "none"})
    assert store.get_model("q").priority == 0 and store.get_model("q").spread == "none"
    r = await c.put("/api/models/fresh", json={"file": "x.gguf"})
    assert r.json()["priority"] == 50 and r.json()["spread"] == "gpu"
    for bad in ({"priority": 101}, {"priority": -1}, {"spread": "rack"}):
        assert (await c.put("/api/models/q", json={"file": "x.gguf", **bad})).status_code == 422, bad


async def test_capacity_reserved_disabled_and_dead(env):
    c, store, rec, _, clock, _ = env
    cfg_timeout = rec.cfg.heartbeat_timeout_s
    a = node("a", devices=[dev("CUDA0", usable=5000, free=6000), dev("CUDA1", usable=3000)])
    a.devices[0].bandwidth_gbps = 900.0
    b = node("b", devices=[dev("CUDA0", usable=8000)])
    register(store, clock, a, b)
    store.put_model(SPEC)
    put_replica(store, "m-1", model="m", state="launching", head="a", now=clock())  # reserves 1000 on a/CUDA0
    store.set_gpu_enabled("a", "CUDA1", False)
    clock.t += cfg_timeout + 1
    store.upsert_node(a, clock())  # a stays fresh, b goes silent
    j = (await c.get("/api/capacity")).json()
    g = {(x["node_id"], x["device_id"]): x for x in j["gpus"]}
    a0, a1, b0 = g[("a", "CUDA0")], g[("a", "CUDA1")], g[("b", "CUDA0")]
    assert (a0["usable_mb"], a0["free_for_new_mb"], a0["reserved_mb"]) == (5000, 4000, 1000)
    assert a0["bandwidth_gbps"] == 900.0 and a0["alive"] and a0["enabled"]
    assert a0["replicas"] == [{"replica_id": "m-1", "model": "m", "est_mb": 1000, "busy": 0.0}]
    assert (a1["enabled"], a1["free_for_new_mb"], a1["reserved_mb"], a1["usable_mb"]) == (False, 0, 0, 3000)
    assert (b0["alive"], b0["free_for_new_mb"], b0["reserved_mb"]) == (False, 0, 0)
    assert j["summary"] == {"gpus": 3, "free_for_new_mb": 4000, "largest_single_gpu_mb": 4000,
                            "largest_single_node_mb": 4000}


async def test_capacity_subtracts_own_replicas_from_a_budget(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a", devices=[dev("CUDA0", free=8000, usable=2500, budget=2500)]))
    store.put_model(SPEC)
    put_replica(store, "m-1", model="m", state="ready", head="a", now=clock())
    clock.t += rec.READY_REPORT_GRACE_S + 1
    store.upsert_node(node("a", devices=[dev("CUDA0", free=8000, usable=2500, budget=2500)]), clock())
    g = (await c.get("/api/capacity")).json()["gpus"][0]
    assert (g["usable_mb"], g["free_for_new_mb"]) == (2500, 1500)


def placement(tier="single_gpu", node_id="a", est=1000):
    from gpupool.common.models import DeviceAssignment, Placement
    asg = [DeviceAssignment(node_id=node_id, device_id="CUDA0", llama_device="CUDA0", layers=4, est_mb=est)]
    return Placement(model="x", replica_id="", tier=tier, head_node=node_id, head_port=0, assignments=asg,
                     tensor_split=[1.0], est_total_mb=est, score=0.9, est_decode_tps=42.0, reasons=["fast"])


async def test_recommend_passthrough_and_max_ctx(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a"))
    calls = []

    def ranker(meta, spec, nodes, occupants=(), limit=5):
        calls.append((spec.name, spec.priority, spec.spread, limit))
        if spec.ctx_size <= 2048:
            return [placement()]
        if spec.ctx_size <= 8192:
            return [placement("single_node")] * 5
        return []

    rec.ranker = ranker
    r = await c.post("/api/recommend", json={"file": "x.gguf", "ctx_size": 4096, "priority": 70,
                                             "spread": "node", "limit": 2})
    assert r.status_code == 200
    j = r.json()
    assert calls[0] == ("x", 70, "node", 2)
    assert j["need_mb"] > 0 and j["not_possible"] is None and len(j["options"]) == 2
    assert j["options"][0] == {
        "rank": 1, "score": 0.9, "tier": "single_node", "fits_now": True,
        "assignments": [{"node_id": "a", "device_id": "CUDA0", "layers": 4, "est_mb": 1000}],
        "est_decode_tps": 42.0, "est_total_mb": 1000, "reasons": ["fast"]}
    assert j["max_ctx_single_gpu"] == 2048


async def test_recommend_not_possible_reports_largest_fitting_ctx(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a", devices=[dev(usable=3000)]))
    rec.ranker = lambda meta, spec, nodes, occupants=(), limit=5: [placement()] if spec.ctx_size <= 3000 else []
    j = (await c.post("/api/recommend", json={"file": "x.gguf", "ctx_size": 16384})).json()
    assert j["options"] == [] and j["max_ctx_single_gpu"] == 2816
    np = j["not_possible"]
    assert np["max_ctx_that_fits"] == 2816 and np["need_mb"] == j["need_mb"]
    assert np["largest_single_gpu_mb"] == 3000 and np["largest_single_node_mb"] == 3000
    rec.ranker = lambda *a, **k: []
    j = (await c.post("/api/recommend", json={"file": "x.gguf"})).json()
    assert j["not_possible"]["max_ctx_that_fits"] is None and j["max_ctx_single_gpu"] is None


async def test_recommend_errors(env):
    c, store, rec, *_ = env
    assert (await c.post("/api/recommend", json={"file": "nope.gguf"})).status_code == 422
    assert (await c.post("/api/recommend", json={"file": "big.gguf"})).status_code == 422
    assert (await c.post("/api/recommend", json={"file": "x.gguf", "limit": 11})).status_code == 422

    async def boom(spec):
        raise ValueError("bad gguf")

    rec.meta_for = boom
    r = await c.post("/api/recommend", json={"file": "x.gguf"})
    assert r.status_code == 400 and "bad gguf" in r.json()["detail"]


# ---------------------------------------------------------------- autoscaling
async def test_put_model_scaling_fields_validation_and_keep(env):
    c, store, rec, _, clock, _ = env
    register(store, clock, node("a"))
    body = {"file": "x.gguf", "min_replicas": 0, "max_replicas": 3, "idle_unload_s": 600,
            "autoscale": {"target_busy": 0.5}}
    r = await c.put("/api/models/q", json=body)
    assert r.status_code == 200
    j = r.json()
    assert (j["min_replicas"], j["max_replicas"], j["idle_unload_s"]) == (0, 3, 600)
    assert j["autoscale"]["target_busy"] == 0.5
    r = await c.put("/api/models/q", json={"file": "x.gguf", "ctx_size": 8192})  # omitted -> kept
    assert r.status_code == 200 and (r.json()["max_replicas"], r.json()["idle_unload_s"]) == (3, 600)
    r = await c.put("/api/models/q", json={"file": "x.gguf", "min_replicas": 5})
    assert r.status_code == 422 and "max_replicas" in r.text
    r = await c.put("/api/models/q", json={"file": "x.gguf", "min_replicas": 1})  # idle_unload_s kept -> invalid
    assert r.status_code == 422 and "idle_unload_s" in r.text
    r = await c.put("/api/models/new", json={"file": "x.gguf", "idle_unload_s": 60})
    assert r.status_code == 422


async def test_start_keeps_scaling_policy(env):
    c, store, *_ = env
    store.put_model(SPEC.model_copy(update={"min_replicas": 0, "max_replicas": 4, "idle_unload_s": 90}))
    r = await c.post("/api/models/m/start", json={"replicas": 2})
    assert r.status_code == 200
    s = store.get_model("m")
    assert (s.replicas, s.min_replicas, s.max_replicas, s.idle_unload_s) == (2, 0, 4, 90)


async def test_scaling_endpoint_404_without_autoscaler(env):
    c, store, *_ = env
    store.put_model(SPEC)
    assert (await c.get("/api/models/m/scaling")).status_code == 404


async def test_state_scaling_block_without_autoscaler(env):
    c, store, *_ = env
    store.put_model(SPEC.model_copy(update={"replicas": 2, "min_replicas": 1, "max_replicas": 4}))
    m = model_state((await c.get("/api/state")).json())
    assert m["scaling"] == {"min": 1, "max": 4, "desired": 2, "avg_busy": None, "unloaded": False}


@pytest.mark.parametrize("env", [FakeAutoscaler({"m": 0}, avg_busy=0.25)], indirect=True)
async def test_state_idle_when_unloaded_and_scaling_view(env):
    c, store, *_ = env
    store.put_model(SPEC.model_copy(update={"replicas": 1, "min_replicas": 0, "max_replicas": 2}))
    m = model_state((await c.get("/api/state")).json())
    assert m["state"] == "idle" and m["error"] is None
    assert m["scaling"] == {"min": 0, "max": 2, "desired": 0, "avg_busy": 0.25, "unloaded": True}
    r = await c.get("/api/models/m/scaling")
    assert r.status_code == 200 and r.json()["avg_busy"] == 0.25
    assert (await c.get("/api/models/nope/scaling")).status_code == 404


@pytest.mark.parametrize("env", [FakeAutoscaler({"m": 0})], indirect=True)
async def test_state_not_idle_while_a_replica_is_still_live(env):
    c, store, _, _, clock, _ = env
    store.put_model(SPEC.model_copy(update={"replicas": 1, "min_replicas": 0}))
    put_replica(store, "m-1", state="ready", now=clock())
    assert model_state((await c.get("/api/state")).json())["state"] == "running"
