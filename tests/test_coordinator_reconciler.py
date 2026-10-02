import httpx
import pytest
import respx

from gpupool.common.models import EngineStatus
from tests.test_coordinator_helpers import (
    SPEC, Clock, FakeClient, dev, make_cfg, make_planner, make_reconciler, node, put_replica, settle,
)

HEALTH = r"http://10\.0\.0\.\d+:\d+/health"


def beat(store, clock, *nodes):
    for n in nodes:
        store.upsert_node(n, clock())


@pytest.fixture
def mock_health():
    with respx.mock(assert_all_called=False) as m:
        route = m.get(url__regex=HEALTH).mock(return_value=httpx.Response(200, json={"status": "ok"}))
        yield route


async def test_launch_happy_path_orders_steps(mock_health):
    client = FakeClient()
    rec, store, clock = make_reconciler(planner=make_planner(rpc=True), client=client)
    beat(store, clock, node("a"), node("b"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "ready"
    assert client.kinds() == ["start", "ensure", "start"]
    rpc, ensure, head = [c for c in client.calls if c[0] in ("start", "ensure")]
    assert rpc[2] == f"{r.replica_id}-rpc-CUDA0" and rpc[3].kind == "rpc" and rpc[1] == "http://10.0.0.2:7070"
    assert head[2] == f"{r.replica_id}-head" and head[1] == "http://10.0.0.1:7070"
    hs = head[3]
    assert hs.kind == "server" and hs.devices == ["CUDA0", "RPC0"] and hs.model == "m"
    assert hs.rpc_endpoints == [f"10.0.0.2:{rpc[3].port}"] and hs.model_path == "/cache/m.gguf"
    assert mock_health.called
    await rec.shutdown()


async def test_launch_failure_stops_every_created_engine(mock_health):
    client = FakeClient()
    client.fail_start_on = "-head"
    rec, store, clock = make_reconciler(planner=make_planner(rpc=True), client=client)
    beat(store, clock, node("a"), node("b"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "failed" and "boom" in r.error
    stopped = {c[2] for c in client.calls if c[0] == "stop"}
    assert stopped == {f"{r.replica_id}-rpc-CUDA0", f"{r.replica_id}-head"}
    assert client.engines == {}
    # backoff: next tick does not immediately relaunch
    await rec.tick()
    await settle(rec)
    assert len(store.list_replicas()) == 1


async def test_health_timeout_fails_and_rolls_back_with_log_tail():
    client = FakeClient()
    cfg = make_cfg(launch_timeout_s=0.1)
    rec, store, clock = make_reconciler(cfg=cfg, client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    with respx.mock() as m:
        m.get(url__regex=HEALTH).mock(return_value=httpx.Response(503))
        await rec.tick()
        await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "failed" and "not healthy" in r.error and "head says hi" in r.error
    assert client.engines == {}


async def test_reservation_prevents_double_booking(mock_health):
    mock_health.mock(return_value=httpx.Response(503))  # first launch stays 'launching'
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=1000))
    beat(store, clock, node("a", devices=[dev(free=1800, usable=1500)]))
    store.put_model(SPEC.model_copy(update={"replicas": 2}))
    await rec.tick()
    await rec.tick()  # usable 1500 - 1000 reserved = 500: second must NoFit
    reps = store.list_replicas()
    assert len(reps) == 1 and reps[0].state == "launching"
    await rec.shutdown()


async def test_port_alloc_unique_across_replicas_and_reports(mock_health):
    mock_health.mock(return_value=httpx.Response(503))
    from gpupool.common.models import EngineStatus
    busy = EngineStatus(engine_id="foreign", kind="rpc", state="running", port=9000)
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a", devices=[dev(usable=9000)], engines=[busy]))
    store.put_model(SPEC.model_copy(update={"replicas": 3}))
    for _ in range(3):
        await rec.tick()
    ports = [r.placement.head_port for r in store.list_replicas()]
    assert len(ports) == 3 and len(set(ports)) == 3 and 9000 not in ports
    await rec.shutdown()


async def test_plan_for_has_no_side_effects():
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    p1 = await rec.plan_for(SPEC)
    p2 = await rec.plan_for(SPEC)
    assert p1.head_port == p2.head_port and store.list_replicas() == []


async def test_dead_node_fails_replica_and_stops_survivors():
    client = FakeClient()
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"), node("b"))
    put_replica(store, "m-1", rpc_node="b", now=clock())
    clock.t += 5
    beat(store, clock, node("a"))  # b stops heartbeating
    clock.t += 8  # b last seen 13s ago, a 8s ago
    await rec.tick()
    r = store.get_replica("m-1")
    assert r.state == "failed" and "b" in r.error
    assert {c[1:3] for c in client.calls if c[0] == "stop"} == {("http://10.0.0.1:7070", "m-1-head")}


async def test_exited_and_missing_engines_fail_ready_replica():
    rec, store, clock = make_reconciler()
    put_replica(store, "m-1", now=clock())
    clock.t += 1
    exited = EngineStatus(engine_id="m-1-head", kind="server", state="exited", port=9000, exit_code=1)
    beat(store, clock, node("a", engines=[exited]))
    await rec.tick()
    assert store.get_replica("m-1").state == "failed"

    put_replica(store, "m-2", head_port=9001, now=clock())
    clock.t += 1
    beat(store, clock, node("a"))  # report seen after ready, engine missing
    await rec.tick()
    assert store.get_replica("m-2").state == "failed"
    assert "missing" in store.get_replica("m-2").error


async def test_note_error_health_checks_replica():
    rec, store, clock = make_reconciler()
    store.put_model(SPEC.model_copy(update={"replicas": 2}))
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    put_replica(store, "m-2", head_port=9001, now=clock())
    running = {e: EngineStatus(engine_id=e, kind="server", state="running", port=9000) for e in ("m-1-head", "m-2-head")}
    beat(store, clock, node("a", engines=list(running.values())))
    with respx.mock() as m:
        m.get("http://10.0.0.1:9000/health").mock(return_value=httpx.Response(503))
        m.get("http://10.0.0.1:9001/health").mock(return_value=httpx.Response(200))
        rec.note_error("m-1")
        rec.note_error("m-2")
        await rec.tick()
    assert store.get_replica("m-1").state == "failed"
    assert store.get_replica("m-2").state == "ready"
    await rec.shutdown()


async def test_drain_waits_for_outstanding_then_stops():
    client = FakeClient()
    out = {"m-1": 2}
    rec, store, clock = make_reconciler(client=client, outstanding=lambda rid: out.get(rid, 0))
    store.put_model(SPEC.model_copy(update={"replicas": 0}))
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    beat(store, clock, node("a", engines=[EngineStatus(engine_id="m-1-head", kind="server", state="running", port=9000)]))
    await rec.tick()  # replicas=0 -> extra -> drain
    assert store.get_replica("m-1").state == "draining"
    clock.t += 1
    beat(store, clock, node("a"))
    await rec.tick()
    assert store.get_replica("m-1").state == "draining" and "stop" not in client.kinds()
    out["m-1"] = 0
    await rec.tick()
    assert store.get_replica("m-1").state == "stopped"
    assert ("stop", "http://10.0.0.1:7070", "m-1-head") in client.calls


async def test_drain_timeout_forces_stop():
    cfg = make_cfg(drain_timeout_s=30)
    rec, store, clock = make_reconciler(cfg=cfg, outstanding=lambda rid: 5)
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    store.set_replica_state("m-1", "draining", now=clock())
    clock.t += 31
    beat(store, clock, node("a"))
    await rec.tick()
    assert store.get_replica("m-1").state == "stopped"


async def test_low_free_replaces_then_drains_old(mock_health):
    rec, store, clock = make_reconciler()
    store.put_model(SPEC)
    low = dev("CUDA0", free=100, usable=0)
    ok = dev("CUDA0", free=8000, usable=7000)
    beat(store, clock, node("a", devices=[low]), node("b", devices=[ok]))
    put_replica(store, "m-1", now=clock())
    beat(store, clock, node("a", devices=[low], engines=[
        EngineStatus(engine_id="m-1-head", kind="server", state="running", port=9000)]), node("b", devices=[ok]))
    await rec.tick()  # launches replacement on b
    await settle(rec)
    new = [r for r in store.list_replicas() if r.replica_id != "m-1"]
    assert len(new) == 1 and new[0].placement.head_node == "b" and new[0].state == "ready"
    assert store.get_replica("m-1").state == "ready"
    clock.t += 1
    nid = new[0].replica_id
    beat(store, clock, node("a", devices=[low], engines=[
        EngineStatus(engine_id="m-1-head", kind="server", state="running", port=9000)]),
        node("b", devices=[ok], engines=[
            EngineStatus(engine_id=f"{nid}-head", kind="server", state="running", port=new[0].placement.head_port)]))
    await rec.tick()
    assert store.get_replica("m-1").state == "draining"
    await rec.shutdown()


async def test_nofit_does_not_crash_and_logs_once(caplog):
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=10**6))
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    for _ in range(3):
        await rec.tick()
    assert store.list_replicas() == []
    assert len([r for r in caplog.records if "cannot place" in r.getMessage()]) == 1


async def test_source_inside_models_dir_becomes_coordinator_url(tmp_path):
    cfg = make_cfg(tmp_path)
    rec, store, clock = make_reconciler(cfg=cfg)
    spec = SPEC.model_copy(update={"source": str(cfg.models_dir / "big.gguf")})
    assert rec._model_source(spec) == ("big.gguf", "coordinator://big.gguf")
    assert rec._model_source(SPEC)[1] == SPEC.source


async def test_ready_replica_stays_reserved_until_a_fresh_report(mock_health):
    # The replica just became ready, but the node's last report may predate the model
    # load: its memory must still count as taken, or the next plan double-books it.
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=1000))
    beat(store, clock, node("a", devices=[dev(free=1800, usable=1500)]))
    put_replica(store, rid="m-old", state="ready", now=clock())
    store.put_model(SPEC.model_copy(update={"replicas": 2}))
    await rec.tick()
    assert [r.replica_id for r in store.list_replicas()] == ["m-old"]

    # A report arriving well after 'ready' reflects the load; the reservation is dropped.
    clock.t += rec.READY_REPORT_GRACE_S + 1
    beat(store, clock, node("a", devices=[dev(free=1800, usable=1500)]))
    await rec.tick()
    assert len(store.list_replicas()) == 2
    await rec.shutdown()
