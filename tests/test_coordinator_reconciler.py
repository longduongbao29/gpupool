import asyncio
import math
import httpx
import pytest
import respx

from gpupool.common.models import EngineStatus, ModelSpec
from gpupool.coordinator import preemption
from gpupool.coordinator.reconciler import _Realloc
from gpupool.coordinator.store import ServerRecord
from tests.test_coordinator_helpers import (
    META, SPEC, Clock, FakeAutoscaler, FakeClient, dev, make_cfg, make_planner, make_ranker, make_reconciler,
    make_scored_ranker, node, put_replica, settle,
)

HEALTH = r"http://10\.0\.0\.\d+:\d+/health"


def beat(store, clock, *nodes):
    # only registered servers count for the reconciler
    for n in nodes:
        if store.get_server(n.node_id) is None:
            store.add_server(ServerRecord(node_id=n.node_id, agent_url=n.agent_url, added_at=clock()))
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
    up = EngineStatus(engine_id="m-old-head", kind="server", state="running", port=9000)  # keeps it alive
    beat(store, clock, node("a", devices=[dev(free=1800, usable=1500)], engines=[up]))
    put_replica(store, rid="m-old", state="ready", now=clock())
    store.put_model(SPEC.model_copy(update={"replicas": 2}))
    await rec.tick()
    assert [r.replica_id for r in store.list_replicas()] == ["m-old"]

    # A report arriving well after 'ready' reflects the load; the reservation is dropped.
    clock.t += rec.READY_REPORT_GRACE_S + 1
    beat(store, clock, node("a", devices=[dev(free=1800, usable=1500)], engines=[up]))
    await rec.tick()
    assert len(store.list_replicas()) == 2
    await rec.shutdown()


# ---------------------------------------------------------------- registration, GPU switches, pinning
def spy_planner(seen, inner=None):
    inner = inner or make_planner()

    def planner(meta, spec, nodes, rid, port_alloc, exclude_nodes=frozenset(), **kw):
        seen.append({(n.node_id, d.device_id): d.usable_mb for n in nodes for d in n.devices})
        return inner(meta, spec, nodes, rid, port_alloc)

    return planner


async def test_unregistered_nodes_are_ignored():
    rec, store, clock = make_reconciler()
    store.upsert_node(node("ghost"), clock())  # heartbeat without a server record
    with pytest.raises(Exception, match="nothing fits"):
        await rec.plan_for(SPEC)
    beat(store, clock, node("ghost"))  # registering it makes it count
    assert (await rec.plan_for(SPEC)).head_node == "ghost"


async def test_disabled_gpu_gets_zero_usable_mb():
    seen = []
    rec, store, clock = make_reconciler(planner=spy_planner(seen))
    beat(store, clock, node("a", devices=[dev("CUDA0"), dev("CUDA1")]))
    store.set_gpu_enabled("a", "CUDA0", False)
    p = await rec.plan_for(SPEC)
    assert seen[-1] == {("a", "CUDA0"): 0, ("a", "CUDA1"): 7000}
    assert p.assignments[0].device_id == "CUDA1"
    # the stored report is untouched (planning works on copies)
    assert store.list_nodes()[0].report.devices[0].usable_mb == 7000


async def test_disabled_flag_follows_the_card_not_the_position():
    seen = []
    rec, store, clock = make_reconciler(planner=spy_planner(seen))
    beat(store, clock, node("a", devices=[dev("CUDA0", uuid="GPU-0"), dev("CUDA1", uuid="GPU-1")]))
    store.set_gpu_enabled("a", "GPU-0", False)
    await rec.plan_for(SPEC)
    assert seen[-1] == {("a", "CUDA0"): 0, ("a", "CUDA1"): 7000}
    beat(store, clock, node("a", devices=[dev("CUDA0", uuid="GPU-1")]))  # GPU-0 fell off the bus
    await rec.plan_for(SPEC)
    assert seen[-1] == {("a", "CUDA0"): 7000}  # the disabled card is gone, GPU-1 stays enabled


async def test_pin_devices_zero_everything_else():
    seen = []
    rec, store, clock = make_reconciler(planner=spy_planner(seen))
    beat(store, clock, node("a", devices=[dev("CUDA0")]), node("b", devices=[dev("CUDA0"), dev("CUDA1")]))
    p = await rec.plan_for(SPEC.model_copy(update={"pin_devices": ["b/CUDA1"]}))
    assert seen[-1] == {("a", "CUDA0"): 0, ("b", "CUDA0"): 0, ("b", "CUDA1"): 7000}
    assert (p.head_node, p.assignments[0].device_id) == ("b", "CUDA1")
    await rec.plan_for(SPEC)  # no pin: nothing zeroed
    assert all(v == 7000 for v in seen[-1].values())


async def test_whole_server_pin_allows_all_its_gpus_including_new_ones():
    seen = []
    rec, store, clock = make_reconciler(planner=spy_planner(seen))
    beat(store, clock, node("a", devices=[dev("CUDA0")]), node("b", devices=[dev("CUDA0"), dev("CUDA1")]))
    spec = SPEC.model_copy(update={"pin_devices": ["b/*"]})
    await rec.plan_for(spec)
    assert seen[-1] == {("a", "CUDA0"): 0, ("b", "CUDA0"): 7000, ("b", "CUDA1"): 7000}
    # a GPU added to server b later is allowed without editing the model
    beat(store, clock, node("b", devices=[dev("CUDA0"), dev("CUDA1"), dev("CUDA2")]))
    await rec.plan_for(spec)
    assert seen[-1][("b", "CUDA2")] == 7000 and seen[-1][("a", "CUDA0")] == 0
    # mixed: all of b plus one GPU of a
    await rec.plan_for(SPEC.model_copy(update={"pin_devices": ["b/*", "a/CUDA0"]}))
    assert all(v == 7000 for v in seen[-1].values())


async def test_pin_on_disabled_gpu_does_not_fit():
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.set_gpu_enabled("a", "CUDA0", False)
    with pytest.raises(Exception, match="nothing fits"):
        await rec.plan_for(SPEC.model_copy(update={"pin_devices": ["a/CUDA0"]}))


async def test_disabling_a_gpu_keeps_running_replicas():
    rec, store, clock = make_reconciler()
    store.put_model(SPEC)
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    beat(store, clock, node("a", engines=[EngineStatus(engine_id="m-1-head", kind="server",
                                                       state="running", port=9000)]))
    store.set_gpu_enabled("a", "CUDA0", False)
    await rec.tick()
    assert store.get_replica("m-1").state == "ready"


async def test_remove_node_stops_engines_and_marks_replicas_stopped():
    client = FakeClient()
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"), node("b"), node("c"))
    store.set_gpu_enabled("b", "CUDA0", False)
    put_replica(store, "m-1", rpc_node="b", now=clock())  # touches a and b
    put_replica(store, "o-1", model="o", head="c", head_port=9005, now=clock())  # untouched
    await rec.remove_node("b")
    assert store.get_replica("m-1").state == "stopped"
    assert store.get_replica("o-1").state == "ready"
    stops = {c[1:3] for c in client.calls if c[0] == "stop"}
    assert stops == {("http://10.0.0.1:7070", "m-1-head"), ("http://10.0.0.2:7070", "m-1-rpc-CUDA0")}
    assert store.get_server("b") is None and store.gpu_flags() == {}
    assert [n.report.node_id for n in store.list_nodes()] == ["a", "c"]


async def test_remove_node_then_tick_replaces_replica_elsewhere(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"), node("b"))
    store.put_model(SPEC)
    put_replica(store, "m-1", head="a", now=clock())
    await rec.remove_node("a")
    await rec.tick()
    await settle(rec)
    new = [r for r in store.list_replicas() if r.replica_id != "m-1"]
    assert len(new) == 1 and new[0].placement.head_node == "b" and new[0].state == "ready"
    await rec.shutdown()


async def test_ports_of_dead_engines_are_reused():
    dead = EngineStatus(engine_id="old", kind="rpc", state="exited", port=9000, exit_code=1)
    live = EngineStatus(engine_id="live", kind="rpc", state="running", port=9001)
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a", engines=[dead, live]))
    assert (await rec.plan_for(SPEC)).head_port == 9000  # exited holds nothing, running does


# --- crash loop after ready (C4) ---------------------------------------------------------

async def _ready_then_crash(rec, store, clock, after=10.0):
    """Wait until a replica is ready, then make its head engine exit `after` s later."""
    await rec.tick()
    await settle(rec)
    r = store.list_replicas(states={"ready"})[0]
    clock.t += after
    exited = EngineStatus(engine_id=f"{r.replica_id}-head", kind="server", state="exited", port=9000, exit_code=1)
    beat(store, clock, node("a", engines=[exited]))
    await rec.tick()
    assert store.get_replica(r.replica_id).state == "failed"
    return r


async def test_crash_after_ready_backs_off_then_relaunches(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await _ready_then_crash(rec, store, clock)
    n = len(store.list_replicas())
    clock.t += 4.0  # delay is 5 s
    beat(store, clock, node("a"))
    await rec.tick()
    await settle(rec)
    assert len(store.list_replicas()) == n  # no immediate relaunch
    clock.t += 1.5
    beat(store, clock, node("a"))
    await rec.tick()
    await settle(rec)
    assert len(store.list_replicas()) == n + 1
    await rec.shutdown()


async def test_repeated_crashes_double_delay_up_to_cap(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    expected = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0]
    for i, delay in enumerate(expected):
        await _ready_then_crash(rec, store, clock)
        n, not_before = rec._backoff["m"]
        assert n == i + 1 and not_before == pytest.approx(clock.t + delay)
        clock.t = not_before  # wait the backoff out; the next loop relaunches
        beat(store, clock, node("a"))
    await rec.shutdown()


async def test_stable_replica_clears_backoff(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await _ready_then_crash(rec, store, clock)
    clock.t = rec._backoff["m"][1]
    beat(store, clock, node("a"))
    await rec.tick()
    await settle(rec)
    assert "m" in rec._backoff  # becoming ready alone does not clear it
    clock.t += rec.STABLE_S - 1
    r = store.list_replicas(states={"ready"})[0]
    running = EngineStatus(engine_id=f"{r.replica_id}-head", kind="server", state="running", port=9000)
    beat(store, clock, node("a", engines=[running]))
    await rec.tick()
    assert "m" in rec._backoff
    clock.t += 2
    beat(store, clock, node("a", engines=[running]))
    await rec.tick()
    assert "m" not in rec._backoff
    await rec.shutdown()


async def test_node_death_does_not_set_backoff():
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    clock.t += 20  # node silent past heartbeat_timeout_s
    await rec.tick()
    assert store.get_replica("m-1").state == "failed"
    assert "m" not in rec._backoff


async def test_gpu_missing_does_not_set_backoff():
    rec, store, clock = make_reconciler()
    put_replica(store, "m-1", now=clock())
    beat(store, clock, node("a", devices=[dev("CUDA9")]))
    await rec.tick()
    assert store.get_replica("m-1").state == "failed" and "m" not in rec._backoff


def _running(rid="m-1"):
    return [EngineStatus(engine_id=f"{rid}-head", kind="server", state="running", port=9000)]


async def test_shifted_gpu_does_not_fail_healthy_replica():
    rec, store, clock = make_reconciler()
    store.put_model(SPEC)
    beat(store, clock, node("a", devices=[dev("CUDA0", uuid="GPU-0"), dev("CUDA1", uuid="GPU-1")]))
    put_replica(store, "m-1", now=clock(), head_device="CUDA1", head_uuid="GPU-1")
    clock.t += 1
    # GPU-0 fell off the bus: GPU-1 is now CUDA0 and CUDA1 no longer exists
    beat(store, clock, node("a", devices=[dev("CUDA0", uuid="GPU-1")], engines=_running()))
    await rec.tick()
    assert store.get_replica("m-1").state == "ready"
    assert "gpu_missing" not in [e.kind for e in store.list_events(limit=50)]


async def test_lost_uuid_fails_replica_even_if_old_device_id_is_taken():
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a", devices=[dev("CUDA0", uuid="GPU-0"), dev("CUDA1", uuid="GPU-1")]))
    put_replica(store, "m-1", now=clock(), head_device="CUDA1", head_uuid="GPU-1")
    clock.t += 1
    beat(store, clock, node("a", devices=[dev("CUDA0", uuid="GPU-0"), dev("CUDA1", uuid="GPU-9")],
                            engines=_running()))  # a different card now holds the id CUDA1
    await rec.tick()
    r = store.get_replica("m-1")
    assert r.state == "failed" and "a/CUDA1" in r.error
    assert "gpu_missing" in [e.kind for e in store.list_events(limit=50)]


async def test_no_uuids_in_report_falls_back_to_device_id():
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a", devices=[dev("CUDA0"), dev("CUDA1")]))
    put_replica(store, "m-1", now=clock(), head_device="CUDA1", head_uuid="GPU-1")
    put_replica(store, "m-2", now=clock(), head_device="CUDA0", head_uuid="GPU-0", head_port=9100)
    clock.t += 1
    beat(store, clock, node("a", devices=[dev("CUDA0")], engines=_running() + _running("m-2")))  # downgraded agent
    await rec.tick()
    assert store.get_replica("m-1").state == "failed"  # CUDA1 is not reported
    assert store.get_replica("m-2").state != "failed"  # CUDA0 is (no spec: drained, not failed)


async def test_late_crash_after_stable_period_sets_no_backoff():
    rec, store, clock = make_reconciler()
    put_replica(store, "m-1", now=clock())
    clock.t += rec.STABLE_S + 1
    exited = EngineStatus(engine_id="m-1-head", kind="server", state="exited", port=9000, exit_code=1)
    beat(store, clock, node("a", engines=[exited]))
    await rec.tick()
    assert store.get_replica("m-1").state == "failed" and "m" not in rec._backoff


async def test_crash_loop_event_on_second_crash(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await _ready_then_crash(rec, store, clock)
    assert not [e for e in store.list_events() if e.kind == "crash_loop"]
    clock.t = rec._backoff["m"][1]
    beat(store, clock, node("a"))
    await _ready_then_crash(rec, store, clock)
    ev = [e for e in store.list_events() if e.kind == "crash_loop"]
    assert len(ev) == 1 and ev[0].level == "warning" and ev[0].model == "m"
    assert "2 times" in ev[0].message and "10 s" in ev[0].message
    await rec.shutdown()


async def test_tick_prunes_old_terminal_replicas():
    rec, store, clock = make_reconciler()
    for i in range(rec.KEEP_TERMINAL_PER_MODEL + 3):
        put_replica(store, f"m-{i:02d}", state="failed", now=clock() + i)
    await rec.tick()
    assert len(store.list_replicas(model="m")) == rec.KEEP_TERMINAL_PER_MODEL


async def test_keyboard_interrupt_in_launch_propagates_without_failure_or_backoff():
    class Interrupting(FakeClient):
        async def start_engine(self, url, spec):
            raise KeyboardInterrupt

    rec, store, clock = make_reconciler(client=Interrupting())
    beat(store, clock, node("a"))
    r = put_replica(store, "m-1", state="launching", now=clock())
    with pytest.raises(KeyboardInterrupt):
        await rec._launch(r, SPEC)
    assert "m" not in rec._backoff
    assert store.get_replica("m-1").state == "launching"  # not recorded as a launch failure
    assert not any(e.kind == "launch_failed" for e in store.list_events())


async def test_replica_drained_while_launching_rolls_back_quietly(mock_health):
    class DrainOnHead(FakeClient):
        def __init__(self, store):
            super().__init__()
            self.store = store

        async def start_engine(self, url, spec):
            st = await super().start_engine(url, spec)
            if spec.engine_id.endswith("-head"):
                rid = spec.engine_id[: -len("-head")]
                self.store.set_replica_state(rid, "draining", None, now=1.0)
            return st

    client = DrainOnHead(None)
    rec, store, clock = make_reconciler(client=client)
    client.store = store
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "draining"
    assert client.engines == {}  # engines rolled back
    assert "m" not in rec._backoff
    assert not any(e.kind == "launch_failed" for e in store.list_events())
    await rec.shutdown()


# ---------------------------------------------------------------- priority, fairness, occupants, views
async def test_higher_priority_gets_scarce_vram_despite_name(mock_health):
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=1000))
    beat(store, clock, node("a", devices=[dev(free=1800, usable=1500)]))  # room for one
    store.put_model(SPEC.model_copy(update={"name": "aaa", "replicas": 1, "priority": 50}))
    store.put_model(SPEC.model_copy(update={"name": "zzz", "replicas": 1, "priority": 80}))
    await rec.tick()
    await settle(rec)
    assert [r.model for r in store.list_replicas()] == ["zzz"]
    await rec.shutdown()


async def test_every_model_gets_first_replica_before_a_second(mock_health):
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=1000))
    # CUDA0 is taken by a's first replica; CUDA1 is the only room left.
    beat(store, clock, node("a", devices=[dev("CUDA0", free=500, usable=500), dev("CUDA1", usable=1500)]))
    put_replica(store, "a-1", model="a", now=clock())
    store.put_model(SPEC.model_copy(update={"name": "a", "replicas": 2}))
    store.put_model(SPEC.model_copy(update={"name": "b", "replicas": 1}))
    await rec.tick()
    await settle(rec)
    by_model = {}
    for r in store.list_replicas():
        by_model.setdefault(r.model, []).append(r)
    assert len(by_model["a"]) == 1 and len(by_model["b"]) == 1  # name order would have given it to a
    await rec.shutdown()


async def test_occupants_resolve_device_ids_busy_and_include_draining():
    load = {"m-1": 3, "d-1": 1, "gone-1": 0}
    rec, store, clock = make_reconciler(outstanding=lambda rid: load.get(rid, 0))
    # uuid moved from CUDA0 to CUDA1 on node a; the stored assignment still says CUDA0
    beat_nodes = node("a", devices=[dev("CUDA0", uuid="U0"), dev("CUDA1", uuid="U1")])
    store.add_server(ServerRecord(node_id="a", agent_url=beat_nodes.agent_url, added_at=clock()))
    store.upsert_node(beat_nodes, clock())
    store.put_model(SPEC.model_copy(update={"parallel": 2}))
    put_replica(store, "m-1", model="m", head_device="CUDA0", head_uuid="U1")
    put_replica(store, "d-1", model="m", state="draining", head_device="CUDA0", head_uuid="U0", head_port=9100)
    put_replica(store, "gone-1", model="m", head_device="CUDA0", head_uuid="UX", head_port=9200)  # card lost
    put_replica(store, "old-1", model="m", state="stopped", head_port=9300)  # terminal: holds nothing
    occ = {o.replica_id: o for o in rec.occupants()}
    assert set(occ) == {"m-1", "d-1"}
    assert occ["m-1"].device_id == "CUDA1" and occ["m-1"].busy == 1.0  # 3/2 capped
    assert occ["d-1"].device_id == "CUDA0" and occ["d-1"].busy == 0.5 and occ["d-1"].est_mb == 1000
    await rec.shutdown()


async def _budget_setup(est=1000, budget=2500, state="ready"):
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=est))
    up = EngineStatus(engine_id="m-1-head", kind="server", state="running", port=9000)
    put_replica(store, "m-1", state=state, now=clock())
    clock.t += rec.READY_REPORT_GRACE_S + 1  # the grace reservation no longer applies
    # free memory still looks roomy (a shared card): only the budget can stop the next replica
    beat(store, clock, node("a", devices=[dev(free=8000, usable=budget, budget=budget)], engines=[up]))
    return rec, store, clock


async def test_budget_subtracts_own_replicas_after_the_grace():
    rec, store, clock = await _budget_setup()
    assert rec.available_reports()[0].devices[0].usable_mb == 1500  # 2500 - 1000
    await rec.shutdown()


async def test_budget_second_model_needing_more_than_left_gets_nofit(mock_health):
    rec, store, clock = await _budget_setup(est=2000)
    store.put_model(SPEC.model_copy(update={"replicas": 2}))
    await rec.tick()
    assert [r.replica_id for r in store.list_replicas()] == ["m-1"]
    assert rec.nofit_reason("m")
    await rec.shutdown()


async def test_budget_counts_draining_replicas():
    rec, store, clock = await _budget_setup(state="draining")
    assert rec.available_reports()[0].devices[0].usable_mb == 1500
    await rec.shutdown()


async def test_budget_matches_by_uuid_and_ignores_other_devices():
    rec, store, clock = make_reconciler()
    put_replica(store, "m-1", head_device="CUDA0", head_uuid="U1", now=clock())
    clock.t += rec.READY_REPORT_GRACE_S + 1
    # the card moved to CUDA1; CUDA0 is another physical card with its own budget
    beat(store, clock, node("a", devices=[dev("CUDA0", uuid="U0", usable=2500, budget=2500),
                                          dev("CUDA1", uuid="U1", usable=2500, budget=2500)]))
    got = {d.device_id: d.usable_mb for d in rec.available_reports()[0].devices}
    assert got == {"CUDA0": 2500, "CUDA1": 1500}
    await rec.shutdown()


async def test_unbudgeted_device_is_unaffected_by_own_replicas():
    rec, store, clock = make_reconciler()
    put_replica(store, "m-1", now=clock())
    clock.t += rec.READY_REPORT_GRACE_S + 1
    beat(store, clock, node("a", devices=[dev(usable=2500)]))
    assert rec.available_reports()[0].devices[0].usable_mb == 2500
    await rec.shutdown()


async def test_budget_exhausted_floors_at_zero_and_disabled_stays_zero():
    rec, store, clock = await _budget_setup(budget=500)  # estimate exceeds the budget
    assert rec.available_reports()[0].devices[0].usable_mb == 0
    store.set_gpu_enabled("a", "CUDA0", False)
    assert rec.available_reports()[0].devices[0].usable_mb == 0
    await rec.shutdown()


async def test_plan_for_passes_occupants_to_the_planner(mock_health):
    seen = []
    inner = make_planner()

    def planner(meta, spec, nodes, rid, port_alloc, exclude_nodes=frozenset(), **kw):
        seen.append(kw["occupants"])
        return inner(meta, spec, nodes, rid, port_alloc)

    rec, store, clock = make_reconciler(planner=planner)
    beat(store, clock, node("a", devices=[dev(usable=9000)]))
    put_replica(store, "other-1", model="other")
    await rec.plan_for(SPEC)
    assert [o.replica_id for o in seen[0]] == ["other-1"]
    await rec.shutdown()


async def test_rank_for_applies_pins_and_available_reports(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a", devices=[dev("CUDA0", usable=5000), dev("CUDA1", usable=6000)]),
         node("b", devices=[dev(usable=7000)]))
    store.set_gpu_enabled("a", "CUDA1", False)
    seen = {}

    def ranker(meta, spec, nodes, occupants=(), limit=5):
        seen.update({(n.node_id, d.device_id): d.usable_mb for n in nodes for d in n.devices}, limit=limit)
        return []

    rec.ranker = ranker
    assert await rec.rank_for(SPEC.model_copy(update={"pin_devices": ["a/CUDA0"]}), 4) == []
    assert seen == {("a", "CUDA0"): 5000, ("a", "CUDA1"): 0, ("b", "CUDA0"): 0, "limit": 4}
    await rec.shutdown()


async def test_autoscaler_desired_replaces_spec_replicas(mock_health):
    rec, store, clock = make_reconciler()
    rec.autoscaler = FakeAutoscaler({"m": 2})
    beat(store, clock, node("a", devices=[dev(usable=9000)]))
    store.put_model(SPEC.model_copy(update={"replicas": 1, "min_replicas": 2, "max_replicas": 3}))
    for _ in range(3):
        await rec.tick()
        await settle(rec)
    assert len(store.list_replicas(states={"ready", "launching"})) == 2  # desired, not replicas=1
    await rec.shutdown()


async def test_autoscaler_scale_down_drains_newest():
    rec, store, clock = make_reconciler()
    rec.autoscaler = FakeAutoscaler({"m": 1})
    store.put_model(SPEC.model_copy(update={"replicas": 2, "min_replicas": 1, "max_replicas": 2}))
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    put_replica(store, "m-2", head_port=9001, now=clock() + 5)
    await rec.tick()
    assert store.get_replica("m-1").state == "ready" and store.get_replica("m-2").state == "draining"
    await rec.shutdown()


async def test_unloaded_model_drains_everything_and_clears_realloc():
    rec, store, clock = make_reconciler()
    fake = rec.autoscaler = FakeAutoscaler({"m": 1})
    store.put_model(SPEC.model_copy(update={"replicas": 1, "min_replicas": 0, "max_replicas": 1}))
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    rec._realloc["m"] = _Realloc(since=clock(), lost=None)
    fake.want["m"] = 0  # idle unload
    await rec.tick()
    assert "m" not in rec._realloc
    assert store.get_replica("m-1").state == "draining"
    await rec.shutdown()


async def test_no_autoscaler_keeps_fixed_replicas(mock_health):
    rec, store, clock = make_reconciler()
    assert rec.autoscaler is None
    beat(store, clock, node("a"))
    store.put_model(SPEC.model_copy(update={"replicas": 1, "min_replicas": 0}))
    await rec.tick()
    await settle(rec)
    assert len(store.list_replicas()) == 1
    await rec.shutdown()


# ---------------------------------------------------------------- priority preemption
def _rig(usable_a=500, outstanding=None):
    """Node a is full: a ready replica of 'lo' holds 1000 MB and the node reports little usable."""
    rec, store, clock = make_reconciler(outstanding=outstanding)
    rec.ranker = make_ranker()
    beat(store, clock, node("a", devices=[dev(usable=usable_a)]))
    store.put_model(SPEC.model_copy(update={"name": "lo", "priority": 10}))
    put_replica(store, "lo-1", model="lo", now=clock())
    return rec, store, clock


def _hi(**kw):
    return SPEC.model_copy(update={"name": "hi", "priority": 80, **kw})


def _preempted_events(store):
    return [e for e in store.list_events() if e.kind == "preempted"]


async def test_preempts_lower_priority_below_minimum_then_places_when_memory_is_free(mock_health):
    rec, store, clock = _rig()
    store.put_model(_hi())
    await rec.tick()
    assert store.get_replica("lo-1").state == "draining"
    assert store.list_replicas(model="hi") == []  # waits: the victim still holds its memory
    ev = _preempted_events(store)
    assert len(ev) == 1 and ev[0].level == "warning" and ev[0].model == "lo" and ev[0].node_id == "a"
    assert ev[0].message == "Stopped replica lo-1 of lo (priority 10) to make room for hi (priority 80)"
    # lo is lower priority: it cannot evict hi back and is not relaunched over hi's claim
    assert store.list_replicas(model="lo", states={"launching", "pending"}) == []
    await rec.tick()  # drain finishes (nothing in flight)
    beat(store, clock, node("a", devices=[dev(usable=1500)]))  # the node reports the memory free again
    await rec.tick()
    await settle(rec)
    assert store.get_replica("lo-1").state == "stopped"
    assert [r.state for r in store.list_replicas(model="hi")] == ["ready"]
    await rec.shutdown()


def _claim_rig():
    """lo (needs 500) runs on a; hi (needs 1500) does not fit until lo's 1000 MB come back."""
    need = {"lo": 500, "hi": 1500}
    rec, store, clock = make_reconciler()
    rec.planner = lambda meta, spec, nodes, rid, pa, **kw: make_planner(need[spec.name])(meta, spec, nodes, rid, pa, **kw)
    rec.ranker = lambda meta, spec, nodes, **kw: make_ranker(need[spec.name])(meta, spec, nodes, **kw)
    lo_engine = EngineStatus(engine_id="lo-1-head", kind="server", state="running", port=9000)

    def report(usable):  # the agent's view: lo-1 until stopped, plus engines the fake client started
        engines = [st for (_u, _e), st in rec.client.engines.items()]
        if store.get_replica("lo-1").state in ("ready", "draining"):
            engines.append(lo_engine)
        clock.t += 1
        beat(store, clock, node("a", devices=[dev(usable=usable)], engines=engines))

    store.put_model(SPEC.model_copy(update={"name": "lo", "priority": 10}))
    beat(store, clock, node("a", devices=[dev(usable=1700)], engines=[lo_engine]))
    put_replica(store, "lo-1", model="lo", now=clock())
    clock.t += rec.READY_REPORT_GRACE_S + 1
    report(700)  # lo-1 loaded: 700 left, room for another lo (500) but not for hi (1500)
    return rec, store, clock, report


async def test_evicted_model_does_not_relaunch_into_the_preemptors_room(mock_health):
    # Real-hardware repro: the node still showed room for the small evicted model, which relaunched
    # in the same tick while its old replica drained, and the preemptor never got the memory.
    rec, store, clock, report = _claim_rig()
    store.put_model(_hi())
    await rec.tick()
    await settle(rec)
    assert store.get_replica("lo-1").state == "draining"
    assert store.list_replicas(model="lo", states={"pending", "launching", "ready"}) == []
    report(700)
    await rec.tick()  # drain completes; lo still must not take the room
    await settle(rec)
    assert store.get_replica("lo-1").state == "stopped"
    assert store.list_replicas(model="lo", states={"pending", "launching", "ready"}) == []
    report(1700)  # lo-1's memory is free now
    await rec.tick()
    await settle(rec)
    assert [r.state for r in store.list_replicas(model="hi")] == ["ready"]
    clock.t += rec.READY_REPORT_GRACE_S
    report(200)  # hi loaded: no room left for lo
    await rec.tick()
    await settle(rec)
    assert [r.state for r in store.list_replicas(model="hi")] == ["ready"]
    assert store.list_replicas(model="lo", states={"pending", "launching", "ready"}) == []
    await rec.shutdown()


async def test_preemption_claim_expires(mock_health):
    rec, store, clock, report = _claim_rig()
    store.put_model(_hi())
    await rec.tick()
    report(700)
    await rec.tick()
    # the memory never shows up free (e.g. the victim's process hangs on): once the claim expires
    # lower priorities may use what is there again
    clock.t += rec.cfg.drain_timeout_s + rec.PREEMPT_CLAIM_GRACE_S
    report(700)
    await rec.tick()
    await settle(rec)
    assert len(store.list_replicas(model="lo", states={"ready"})) == 1
    await rec.shutdown()


@pytest.mark.parametrize("victim", [
    {"priority": 80},  # equal priority never evicts
    {"priority": 90},  # higher priority is never a victim
    {"priority": 10, "preemptible": False},
])
async def test_no_preemption_for_equal_higher_or_non_preemptible(victim):
    rec, store, clock = _rig()
    store.put_model(store.get_model("lo").model_copy(update=victim))
    store.put_model(_hi())
    await rec.tick()
    assert store.get_replica("lo-1").state == "ready"
    assert not _preempted_events(store)
    await rec.shutdown()


async def test_autoscale_extra_replica_never_preempts():
    rec, store, clock = _rig()
    rec.autoscaler = FakeAutoscaler({"hi": 2})
    beat(store, clock, node("a", devices=[dev(usable=500)]), node("b", devices=[dev(usable=500)]))
    store.put_model(_hi(min_replicas=1, max_replicas=3))
    put_replica(store, "hi-1", model="hi", head="b", now=clock())  # hi already meets its minimum
    await rec.tick()
    assert store.get_replica("lo-1").state == "ready"
    assert not _preempted_events(store)
    await rec.shutdown()


async def test_cold_start_below_floor_may_preempt():
    rec, store, clock = _rig()
    rec.autoscaler = FakeAutoscaler({"hi": 1})  # a request woke the unloaded model: desired 1, active 0
    store.put_model(_hi(min_replicas=0, max_replicas=2))
    await rec.tick()
    assert store.get_replica("lo-1").state == "draining"
    await rec.shutdown()


async def test_preempt_cooldown_and_pending_victims():
    inflight = {"old-victim": 1}  # its drain is held open by a request in flight
    rec, store, clock = _rig(outstanding=lambda rid: inflight.get(rid, 0))
    store.put_model(_hi())
    rec._preempted["hi"] = (clock(), set())  # evicted a moment ago
    await rec.tick()
    assert store.get_replica("lo-1").state == "ready"
    clock.t += rec.PREEMPT_COOLDOWN_S + 1
    rec.cfg.heartbeat_timeout_s = 10_000  # the node report is old by now; a fresh one would show lo-1 gone
    rec._preempted["hi"] = (clock() - 1000, {"old-victim"})
    put_replica(store, "old-victim", model="other", state="draining", head="a", head_port=9100, now=clock())
    await rec.tick()  # cooldown over, but an earlier victim still drains: wait for it
    assert store.get_replica("lo-1").state == "ready"
    inflight.clear()
    await rec.tick()  # old-victim stops at the start of this tick, then hi may evict
    assert store.get_replica("lo-1").state == "draining"
    assert rec._preempted["hi"] == (clock(), {"lo-1"})
    await rec.shutdown()


async def test_disabled_gpu_is_not_made_usable_by_preemption():
    rec, store, clock = _rig()
    store.set_gpu_enabled("a", "CUDA0", False)
    store.put_model(_hi())
    await rec.tick()
    assert store.get_replica("lo-1").state == "ready"  # freeing a disabled GPU would not let hi use it
    await rec.shutdown()


async def test_preemption_failure_does_not_break_the_tick(monkeypatch):
    rec, store, clock = _rig()
    store.put_model(_hi())

    def boom(*a, **k):
        raise RuntimeError("bad victims")

    monkeypatch.setattr(preemption, "find_victims", boom)
    await rec.tick()
    assert store.get_replica("lo-1").state == "ready"
    await rec.shutdown()


# ---------------------------------------------------------------- simulate
def _snapshot(store):
    return ([(r.replica_id, r.state) for r in store.list_replicas()],
            [m.model_dump() for m in store.list_models()])


async def test_simulate_start_shapes_and_purity():
    client = FakeClient()
    rec, store, clock = make_reconciler(client=client)
    rec.ranker = make_ranker()
    beat(store, clock, node("a", devices=[dev(usable=2500)]))
    store.put_model(SPEC)
    before = _snapshot(store)
    out = await rec.simulate([SPEC.model_copy(update={"replicas": 3})])
    assert [s["model"] for s in out["start"]] == ["m", "m"]  # 2500 MB: two fit, the third does not
    assert out["start"][0] == {"model": "m", "tier": "single_gpu", "est_decode_tps": 40.0,
                               "assignments": [{"node_id": "a", "device_id": "CUDA0", "layers": 2, "est_mb": 1000}]}
    assert out["stop"] == [] and out["preempt"] == []
    assert out["unplaced"] == [{"model": "m", "missing": 1, "why": "NoFit: nothing fits"}]
    assert _snapshot(store) == before and client.calls == [] and rec._preempted == {}


async def test_simulate_stops_newest_and_uses_autoscaler_peek():
    rec, store, clock = make_reconciler()
    rec.ranker = make_ranker()
    rec.autoscaler = fake = FakeAutoscaler({"m": 1})
    beat(store, clock, node("a"))
    store.put_model(SPEC.model_copy(update={"replicas": 2}))
    put_replica(store, "m-1", now=clock())
    put_replica(store, "m-2", head_port=9001, now=clock() + 5)
    out = await rec.simulate([SPEC.model_copy(update={"replicas": 2})])
    assert out["stop"] == [{"replica_id": "m-2", "model": "m", "reason": "2 running, 1 wanted"}]
    assert out["start"] == [] and out["unplaced"] == []
    fake.want["m"] = 0
    out = await rec.simulate([SPEC.model_copy(update={"replicas": 2})])
    assert [s["replica_id"] for s in out["stop"]] == ["m-2", "m-1"]
    assert store.get_replica("m-2").state == "ready"


async def test_simulate_preempts_for_higher_priority():
    rec, store, clock = _rig()
    hi = _hi()
    store.put_model(hi)
    before = _snapshot(store)
    out = await rec.simulate([hi, store.get_model("lo")])
    assert out["preempt"] == [{"replica_id": "lo-1", "model": "lo", "priority": 10, "for_model": "hi"}]
    assert [s["model"] for s in out["start"]] == ["hi"]
    # lo lost its replica and cannot evict hi back: unplaced, not started
    assert [u["model"] for u in out["unplaced"]] == ["lo"]
    assert _snapshot(store) == before and not _preempted_events(store)
    # equal priority: nothing to evict, hi is unplaced
    out = await rec.simulate([hi.model_copy(update={"priority": 10}), store.get_model("lo")])
    assert out["preempt"] == [] and [u["model"] for u in out["unplaced"]] == ["hi"]
    await rec.shutdown()


async def test_simulate_autoscale_extra_does_not_preempt_and_new_model_starts_at_floor():
    rec, store, clock = _rig()
    rec.autoscaler = FakeAutoscaler({"hi": 2})
    beat(store, clock, node("a", devices=[dev(usable=500)]), node("b", devices=[dev(usable=500)]))
    hi = _hi(min_replicas=1, max_replicas=3)
    store.put_model(hi)
    put_replica(store, "hi-1", model="hi", head="b", now=clock())
    out = await rec.simulate([hi, store.get_model("lo")])
    assert out["preempt"] == [] and out["unplaced"] == [{"model": "hi", "missing": 1, "why": "NoFit: nothing fits"}]
    # a model that does not exist yet is wanted at max(min_replicas, 1)
    added = ModelSpec(name="new", source="coordinator://x.gguf", min_replicas=2, max_replicas=4)
    beat(store, clock, node("c", devices=[dev(usable=5000)]))
    out = await rec.simulate([hi, store.get_model("lo"), added])
    assert [s["model"] for s in out["start"]].count("new") == 2
    await rec.shutdown()


# ---------------------------------------------------------------- rebalancing
def _rb_rig(a=40.0, b=80.0, cfg=None, **kw):
    """m-1 ready on a/CUDA0 (score a); b/CUDA0 is free and scores b."""
    rec, store, clock = make_reconciler(cfg=cfg or make_cfg(rebalance_s=0), **kw)
    rec.ranker = make_scored_ranker({("a", "CUDA0"): a, ("b", "CUDA0"): b})
    _rb_beat(store, clock)
    store.put_model(SPEC)
    put_replica(store, "m-1", now=clock())
    return rec, store, clock


def _rb_beat(store, clock):
    # m-1's engine is reported, or later ticks would see it vanish and fail the old replica
    head = EngineStatus(engine_id="m-1-head", kind="server", state="running", port=9000)
    beat(store, clock, node("a", devices=[dev(usable=9000)], engines=[head]), node("b", devices=[dev(usable=9000)]))


def _events(store, kind):
    return [e for e in store.list_events() if e.kind == kind]


async def _mv(rec):
    return (await rec.rebalance_candidates())[0]


async def test_rebalance_candidates_threshold_and_shape():
    rec, store, _ = _rb_rig(a=40, b=64)  # gain 24: below the threshold
    assert await rec.rebalance_candidates() == []
    rec.ranker = make_scored_ranker({("a", "CUDA0"): 40, ("b", "CUDA0"): 65})
    [mv] = await rec.rebalance_candidates()
    assert mv == {"replica_id": "m-1", "model": "m", "from": [{"node_id": "a", "device_id": "CUDA0"}],
                  "to": [{"node_id": "b", "device_id": "CUDA0"}], "current_score": 40, "new_score": 65,
                  "gain": 25, "reasons": ["fits b/CUDA0"]}
    assert store.get_replica("m-1").state == "ready" and len(store.list_replicas()) == 1  # no side effects


async def test_rebalance_candidates_pass_current_as_extra_without_its_own_occupancy_and_sort():
    rec, store, clock = _rb_rig()
    store.put_model(SPEC.model_copy(update={"name": "n"}))
    put_replica(store, "n-1", model="n", now=clock(), head_port=9100)
    mvs = await rec.rebalance_candidates()
    assert {m["replica_id"] for m in mvs} == {"m-1", "n-1"}
    first = next(c for c in rec.ranker.calls if c["extra"][0].replica_id == "m-1")
    assert [p.replica_id for p in first["extra"]] == ["m-1"] and first["limit"] == 5
    assert [o.replica_id for o in first["occupants"]] == ["n-1"]  # itself removed, the other stays
    # real reports: the old replica's reservation is still subtracted (both must fit at once)
    assert next(d.usable_mb for n in first["nodes"] if n.node_id == "a" for d in n.devices) == 9000 - 2000

    def ranker(meta, spec, nodes, occupants=(), limit=5, extra=()):
        gain = 30 if spec.name == "m" else 50
        return [extra[0].model_copy(update={"score": 10.0}),
                extra[0].model_copy(update={"replica_id": "", "score": 10.0 + gain})]

    rec.ranker = ranker
    assert [m["replica_id"] for m in await rec.rebalance_candidates()] == ["n-1", "m-1"]


async def test_rebalance_candidates_apply_pins_and_skip_unready_and_unknown_models():
    rec, store, clock = _rb_rig()
    store.put_model(SPEC.model_copy(update={"pin_devices": ["a/CUDA0"]}))
    assert await rec.rebalance_candidates() == []  # b is pinned out, so nothing better exists
    store.put_model(SPEC)
    put_replica(store, "z-1", model="gone", now=clock(), head_port=9200)
    put_replica(store, "m-2", state="launching", now=clock(), head_port=9300)
    assert [m["replica_id"] for m in await rec.rebalance_candidates()] == ["m-1"]


async def test_start_move_refusals(mock_health):
    rec, store, clock = _rb_rig()
    mv = await _mv(rec)
    rec._move = {"model": "m", "old": "m-1", "new": "x", "since": clock()}
    assert await rec.start_move(mv) is False
    rec._move = None
    put_replica(store, "m-2", state="launching", now=clock(), head_port=9300)
    assert await rec.start_move(mv) is False
    store.set_replica_state("m-2", "stopped", None, now=clock())
    # an active claim: a high-priority model preempted and has not been placed yet
    store.put_model(SPEC.model_copy(update={"name": "hi", "priority": 80}))
    rec._preempted["hi"] = (clock(), {"v"})
    assert await rec.start_move(mv) is False
    rec._preempted.clear()
    store.set_replica_state("m-1", "draining", None, now=clock())
    assert await rec.start_move(mv) is False  # old no longer ready
    assert store.list_replicas(states={"launching", "pending"}) == []
    assert rec._move is None and not _events(store, "rebalance_started")
    await rec.shutdown()


async def test_make_before_break_flow(mock_health):
    rec, store, clock = _rb_rig()
    mock_health.mock(return_value=httpx.Response(503))  # the new replica stays launching
    assert await rec.start_move(await _mv(rec)) is True
    [new] = store.list_replicas(states={"launching"})
    assert [(a.node_id, a.device_id) for a in new.placement.assignments] == [("b", "CUDA0")]
    assert rec._move == {"model": "m", "old": "m-1", "new": new.replica_id, "since": clock()}
    assert store.get_model("m").pin_devices == []  # the pins were for planning only
    [ev] = _events(store, "rebalance_started")
    assert ev.level == "info" and ev.model == "m"
    assert ev.message == "Moving replica m-1 of m from a/CUDA0 to b/CUDA0 (score +40)"
    for _ in range(3):  # while launching, the extra replica is not surplus
        await rec.tick()
    assert store.get_replica("m-1").state == "ready" and store.get_replica(new.replica_id).state == "launching"
    assert rec._move is not None
    # new becomes ready: old is drained, move cleared
    mock_health.mock(return_value=httpx.Response(200))
    await settle(rec)
    assert store.get_replica(new.replica_id).state == "ready"
    await rec.tick()
    assert store.get_replica("m-1").state in ("draining", "stopped") and rec._move is None
    assert store.get_replica(new.replica_id).state == "ready"
    [done] = _events(store, "rebalanced")
    assert done.level == "info" and done.message == "Moved replica m-1 of m to b/CUDA0 (score +40)"
    assert not _events(store, "rebalance_failed")
    await rec.tick()
    assert store.get_replica("m-1").state == "stopped"
    await rec.shutdown()


async def test_move_failure_keeps_old_and_clears_move(mock_health):
    client = FakeClient()
    client.fail_start_on = "-head"
    rec, store, clock = _rb_rig(client=client)
    assert await rec.start_move(await _mv(rec))
    new_id = rec._move["new"]
    await settle(rec)
    assert store.get_replica(new_id).state == "failed"
    await rec.tick()
    assert rec._move is None and store.get_replica("m-1").state == "ready"
    [ev] = _events(store, "rebalance_failed")
    assert ev.level == "warning" and "failed" in ev.message and "boom" in ev.message and "m-1" in ev.message
    assert not _events(store, "rebalanced")
    await rec.shutdown()


async def test_move_abandoned_when_old_disappears(mock_health):
    mock_health.mock(return_value=httpx.Response(503))
    rec, store, clock = _rb_rig()
    assert await rec.start_move(await _mv(rec))
    store.set_replica_state("m-1", "failed", "gpu lost", now=clock())
    await rec.tick()
    assert rec._move is None
    [ev] = _events(store, "rebalance_failed")
    assert "old replica is failed" in ev.message
    await rec.shutdown()


async def test_move_times_out_and_leftover_new_replica_is_drained(mock_health):
    mock_health.mock(return_value=httpx.Response(503))
    rec, store, clock = _rb_rig()  # launch_timeout_s is 5, so the limit is 65 s
    store.put_replica(store.get_replica("m-1").model_copy(update={"created_at": 1.0}))  # older: not the surplus
    assert await rec.start_move(await _mv(rec))
    new_id = rec._move["new"]
    clock.t += 64
    _rb_beat(store, clock)
    await rec.tick()
    assert rec._move is not None
    clock.t += 2
    _rb_beat(store, clock)
    await rec.tick()
    assert rec._move is None and "in time" in _events(store, "rebalance_failed")[0].message
    assert store.get_replica("m-1").state == "ready"
    # the new replica is surplus now and goes through the normal drain
    assert store.get_replica(new_id).state in ("draining", "stopped", "failed")
    await rec.shutdown()


async def test_periodic_run_respects_interval_and_quiet(mock_health):
    rec, store, clock = _rb_rig(cfg=make_cfg(rebalance_s=100))
    await rec.tick()
    assert rec._move is None  # not due yet: boot counts as the last run
    clock.t += 100
    _rb_beat(store, clock)
    store.put_model(SPEC.model_copy(update={"name": "o"}))
    put_replica(store, "o-1", model="o", state="launching", now=clock(), head_port=9300)
    # a launch really in flight (a launching record without its task is an orphan and is failed)
    inflight = rec._launches["o-1"] = asyncio.ensure_future(asyncio.sleep(3600))
    await rec.tick()
    assert rec._move is None and not _events(store, "rebalance_started")  # busy: stays due
    rec._launches.pop("o-1").cancel()
    await asyncio.gather(inflight, return_exceptions=True)
    store.set_replica_state("o-1", "ready", None, now=clock())
    await rec.tick()
    assert rec._move is not None and rec._move["old"] in ("m-1", "o-1")
    assert rec.rebalance_state()["next_run_ts"] == clock() + 100
    await rec.shutdown()


async def test_periodic_run_records_last_run_when_nothing_qualifies_and_zero_disables():
    rec, store, clock = _rb_rig(a=40, b=50, cfg=make_cfg(rebalance_s=100))
    t0 = clock()
    clock.t += 100
    _rb_beat(store, clock)
    await rec.tick()
    assert rec._move is None and rec._last_rebalance == t0 + 100
    off, store2, clock2 = _rb_rig(cfg=make_cfg(rebalance_s=0))
    clock2.t += 10_000
    _rb_beat(store2, clock2)
    await off.tick()
    assert off._move is None and off.ranker.calls == [] and off.rebalance_state()["next_run_ts"] is None


class _FakePoller:
    def __init__(self, **failed):
        self.failed = failed

    def failed_polls(self, node_id):
        return self.failed.get(node_id, 0)


async def _stale_node_rig(failed):
    rec, store, clock = make_reconciler()
    rec.poller = _FakePoller(a=failed)
    store.put_model(SPEC)
    beat(store, clock, node("a", engines=_running()))
    put_replica(store, "m-1", now=clock())
    await rec.tick()  # first sighting: transitions are only reported after a known-alive state
    clock.t += 20  # report stale past heartbeat_timeout_s
    await rec.tick()
    return rec, store, clock


@pytest.mark.parametrize("failed", [0, 1])
async def test_stale_report_without_failed_polls_is_alive(failed):
    # the coordinator's own loop may have stalled: no evidence the agent is gone
    rec, store, clock = await _stale_node_rig(failed)
    assert store.get_replica("m-1").state == "ready"
    assert "node_offline" not in [e.kind for e in store.list_events(limit=50)]
    assert rec.node_alive(store.list_nodes()[0], clock())


async def test_stale_report_with_two_failed_polls_is_dead():
    rec, store, clock = await _stale_node_rig(2)
    assert store.get_replica("m-1").state == "failed"
    assert "node_offline" in [e.kind for e in store.list_events(limit=50)]
    assert not rec.node_alive(store.list_nodes()[0], clock())


async def test_fresh_report_is_alive_whatever_the_poller_says():
    rec, store, clock = make_reconciler()
    rec.poller = _FakePoller(a=9)
    beat(store, clock, node("a"))
    assert rec.node_alive(store.list_nodes()[0], clock())


async def test_no_poller_keeps_the_pure_time_rule():
    rec, store, clock = make_reconciler()
    assert rec.poller is None
    beat(store, clock, node("a"))
    clock.t += 20
    assert not rec.node_alive(store.list_nodes()[0], clock())


# ---------------------------------------------------------------- KV cache type and speculative decoding
DRAFT_META = META.model_copy(update={"vocab_size": 1000})
DSPEC = ModelSpec(name="m", source="coordinator://big.gguf", kv_cache_type="q8_0", speculative="draft",
                  draft="coordinator://small.gguf", draft_n_max=6, replicas=1)


def _meta_by_source(seen=None):
    async def meta_for(spec):
        if seen is not None:
            seen.append(spec.source)
        return DRAFT_META if spec.source.endswith("small.gguf") else META

    return meta_for


async def test_draft_meta_for_only_when_speculative_draft():
    rec, store, clock = make_reconciler()
    seen = []
    rec.meta_for = _meta_by_source(seen)
    assert await rec.draft_meta_for(DSPEC) is DRAFT_META
    assert seen == ["coordinator://small.gguf"]  # meta of the draft file, not of the model
    assert await rec.draft_meta_for(DSPEC.model_copy(update={"speculative": "ngram"})) is None
    assert await rec.draft_meta_for(DSPEC.model_copy(update={"speculative": "none"})) is None
    assert await rec.draft_meta_for(DSPEC.model_copy(update={"draft": None})) is None
    await rec.shutdown()


async def test_draft_meta_reaches_planner_ranker_and_preemption():
    rec, store, clock = make_reconciler()
    rec.meta_for = _meta_by_source()
    beat(store, clock, node("a", devices=[dev(usable=9000)]))
    got = {}

    def planner(meta, spec, nodes, rid, port_alloc, exclude_nodes=frozenset(), **kw):
        got["plan"] = kw.get("draft_meta")
        return make_planner()(meta, spec, nodes, rid, port_alloc)

    def ranker(meta, spec, nodes, occupants=(), limit=5, **kw):
        got.setdefault("rank", []).append(kw.get("draft_meta"))
        return []

    rec.planner, rec.ranker = planner, ranker
    await rec.plan_for(DSPEC)
    await rec.rank_for(DSPEC, 3)
    assert got["plan"] is DRAFT_META and got["rank"] == [DRAFT_META]
    # preemption: a lower-priority replica is a candidate, so find_victims calls the ranker
    put_replica(store, "o-1", model="o")
    store.put_model(ModelSpec(name="o", source="http://x/o.gguf", priority=1))
    got["rank"] = []
    assert await rec._find_victims(DSPEC.model_copy(update={"priority": 90}), exclude="m") is None
    assert got["rank"] and all(d is DRAFT_META for d in got["rank"])
    # a model without a draft passes no draft_meta at all
    got["rank"] = []
    await rec.rank_for(SPEC, 3)
    assert got["rank"] == [None]
    await rec.shutdown()


async def test_simulate_and_rebalance_take_the_draft_into_account():
    rec, store, clock = make_reconciler()
    rec.meta_for = _meta_by_source()
    beat(store, clock, node("a"))
    seen = []

    def ranker(meta, spec, nodes, occupants=(), limit=5, **kw):
        seen.append(kw.get("draft_meta"))
        return []

    rec.ranker = ranker
    out = await rec.simulate([DSPEC])
    assert seen and all(d is DRAFT_META for d in seen)
    assert out["unplaced"][0]["model"] == "m"
    store.put_model(DSPEC)
    put_replica(store, "m-1")
    seen.clear()
    await rec.rebalance_candidates()
    assert seen == [DRAFT_META]
    await rec.shutdown()


async def test_launch_with_draft_ensures_both_files_and_sends_engine_spec(mock_health):
    client = FakeClient()
    rec, store, clock = make_reconciler(client=client)
    rec.meta_for = _meta_by_source()
    beat(store, clock, node("a"))
    store.put_model(DSPEC)
    await rec.tick()
    await settle(rec)
    assert store.list_replicas()[0].state == "ready"
    ensures = [c for c in client.calls if c[0] == "ensure"]
    assert [(c[1], c[2], c[3]) for c in ensures] == [
        ("http://10.0.0.1:7070", "big.gguf", "coordinator://big.gguf"),
        ("http://10.0.0.1:7070", "small.gguf", "coordinator://small.gguf")]
    hs = next(c[3] for c in client.calls if c[0] == "start")
    assert (hs.model_path, hs.draft_model_path) == ("/cache/big.gguf", "/cache/small.gguf")
    assert (hs.cache_type, hs.spec_type, hs.draft_device, hs.draft_n_max) == ("q8_0", "draft", "CUDA0", 6)
    await rec.shutdown()


@pytest.mark.parametrize("mode", ["none", "ngram"])
async def test_launch_without_draft_sends_cache_and_spec_type_only(mock_health, mode):
    client = FakeClient()
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC.model_copy(update={"kv_cache_type": "q4_0", "speculative": mode, "replicas": 1,
                                            "draft": "coordinator://ignored.gguf"}))
    await rec.tick()
    await settle(rec)
    assert client.kinds() == ["ensure", "start"]  # the stale draft source is never fetched
    hs = next(c[3] for c in client.calls if c[0] == "start")
    assert (hs.cache_type, hs.spec_type, hs.draft_model_path, hs.draft_device) == ("q4_0", mode, None, None)
    await rec.shutdown()


@pytest.mark.parametrize("llama_device", ["CPU", "RPC0"])
async def test_draft_on_a_non_cuda_first_device_fails_the_launch_cleanly(mock_health, llama_device):
    client = FakeClient()
    inner = make_planner()

    def planner(*a, **kw):
        p = inner(*a)
        p.assignments[0].llama_device = llama_device
        return p

    rec, store, clock = make_reconciler(planner=planner, client=client)
    rec.meta_for = _meta_by_source()
    beat(store, clock, node("a"))
    store.put_model(DSPEC)
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "failed" and "LaunchError" in r.error and "local CUDA device" in r.error
    assert "start" not in client.kinds()  # nothing was started, so nothing pins VRAM
    await rec.shutdown()


# ---------------------------------------------------------------- persistence across a restart
def _again(rec, store, clock):
    """A new reconciler on the same store: what a coordinator restart builds."""
    from gpupool.coordinator.reconciler import Reconciler
    new = Reconciler(store, rec.cfg, rec.client, rec.meta_for, rec.outstanding, clock)
    new.planner, new.ranker, new.poll_s = rec.planner, rec.ranker, rec.poll_s
    return new


async def test_preemption_cooldown_survives_restart(tmp_path, mock_health):
    from gpupool.coordinator.store import Store
    rec, store, clock = make_reconciler(store=Store(tmp_path / "s.db"))
    rec.ranker = make_ranker()
    beat(store, clock, node("a", devices=[dev(usable=500)]))
    store.put_model(SPEC.model_copy(update={"name": "lo", "priority": 10}))
    put_replica(store, "lo-1", model="lo", now=clock())
    store.put_model(_hi())
    await rec.tick()
    assert store.get_replica("lo-1").state == "draining"
    when = rec._preempted["hi"][0]
    rec2 = _again(rec, store, clock)
    assert rec2._preempted == {"hi": (when, {"lo-1"})}
    # the cooldown is still active: a second victim is not evicted right after the restart
    put_replica(store, "lo-2", model="lo", now=clock(), head_port=9100)
    store.set_replica_state("lo-1", "stopped", None, now=clock())
    rec2.cfg.heartbeat_timeout_s = 10_000
    await rec2.tick()
    assert store.get_replica("lo-2").state == "ready"
    clock.t += rec2.PREEMPT_COOLDOWN_S + 1
    await rec2.tick()
    assert store.get_replica("lo-2").state == "draining"
    await rec.shutdown()
    await rec2.shutdown()


async def test_backoff_survives_restart_and_clears_when_stable(tmp_path, mock_health):
    from gpupool.coordinator.store import Store
    rec, store, clock = make_reconciler(store=Store(tmp_path / "s.db"))
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await _ready_then_crash(rec, store, clock)
    saved = rec._backoff["m"]
    rec2 = _again(rec, store, clock)
    assert rec2._backoff == {"m": saved}
    n = len(store.list_replicas())
    await rec2.tick()
    await settle(rec2)
    assert len(store.list_replicas()) == n  # still backing off after the restart
    clock.t = saved[1] + 1
    beat(store, clock, node("a"))
    await rec2.tick()
    await settle(rec2)
    clock.t += rec2.STABLE_S
    rec2._clear_stable_backoff(clock())
    assert rec2._backoff == {} and _again(rec2, store, clock)._backoff == {}
    await rec.shutdown()
    await rec2.shutdown()


async def test_move_in_flight_resumes_after_restart_and_completes(tmp_path, mock_health):
    from gpupool.coordinator.store import Store
    rec, store, clock = _rb_rig(store=Store(tmp_path / "s.db"))
    assert await rec.start_move(await _mv(rec)) is True
    await settle(rec)  # the new replica becomes ready; the move is still open
    new_id = rec._move["new"]
    assert rec._move is not None and store.get_replica(new_id).state == "ready"
    rec2 = _again(rec, store, clock)
    assert rec2._move == rec._move and rec2._move_text == rec._move_text
    await rec2.tick()
    # without the saved move the new replica would be drained as surplus; instead the old one goes
    assert store.get_replica("m-1").state in ("draining", "stopped")
    assert store.get_replica(new_id).state == "ready"
    assert rec2._move is None and [e.message for e in _events(store, "rebalanced")] == [
        "Moved replica m-1 of m to b/CUDA0 (score +40)"]
    assert _again(rec2, store, clock)._move is None  # cleared in the store too
    await rec.shutdown()
    await rec2.shutdown()


async def test_saved_move_with_missing_replicas_is_cleared_quietly(tmp_path):
    from gpupool.coordinator.store import Store
    rec, store, clock = _rb_rig(store=Store(tmp_path / "s.db"))
    store.put_state("move", {"move": {"model": "m", "old": "m-1", "new": "gone", "since": clock()},
                             "text": ["a/CUDA0", "b/CUDA0", 40.0]})
    rec2 = _again(rec, store, clock)
    assert rec2._move is None and store.get_state("move") is None
    await rec2.tick()
    assert not _events(store, "rebalance_failed")


async def test_unreadable_saved_state_starts_clean(tmp_path):
    from gpupool.coordinator.store import Store
    rec, store, clock = make_reconciler(store=Store(tmp_path / "s.db"))
    store.put_state("preempted", {"hi": "garbage"})
    store.put_state("move", {"nope": 1})
    rec2 = _again(rec, store, clock)
    assert rec2._preempted == {} and rec2._backoff == {} and rec2._move is None


async def test_store_write_failure_does_not_break_reconciliation(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.put_model(SPEC)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    store.put_state = boom
    await _ready_then_crash(rec, store, clock)
    assert "m" in rec._backoff  # kept in memory
    await rec.shutdown()


# ---------------------------------------------------------------- VRAM calibration
def _mem(**devs):
    return {"engine_id": "x", "devices": {k: {"model_mb": v, "kv_mb": 0, "compute_mb": 0, "total_mb": v}
                                          for k, v in devs.items()}}


def _cal_events(store):
    return [e for e in store.list_events() if e.kind == "calibrated"]


def test_calibration_sample_single_rpc_and_draft_and_missing():
    from gpupool.coordinator.reconciler import Reconciler
    store = make_reconciler()[1]
    one = put_replica(store, "m-1").placement  # CUDA0 est 1000
    two = put_replica(store, "m-2", rpc_node="b").placement  # CUDA0 + RPC0, est 1000 each
    # the per-device runtime context (128 MB) is not a llama.cpp buffer: left out of the estimate
    assert Reconciler.calibration_sample(one, _mem(CUDA0=900)) == (900, 872)
    assert Reconciler.calibration_sample(two, _mem(CUDA0=900, RPC0=700)) == (1600, 1744)
    # a draft brings its own context; its buffers are inside the head device's
    draft = one.model_copy(update={"draft_est_mb": 300})
    assert Reconciler.calibration_sample(draft, _mem(CUDA0=900)) == (900, 744)
    # data that does not cover the placement is ignored, never a skewed ratio
    assert Reconciler.calibration_sample(two, _mem(CUDA0=900)) is None
    assert Reconciler.calibration_sample(one, _mem(CUDA1=900)) is None
    assert Reconciler.calibration_sample(one, _mem(CUDA0=0)) is None
    assert Reconciler.calibration_sample(one, {"devices": {}}) is None
    assert Reconciler.calibration_sample(one, {}) is None
    tiny = one.model_copy(update={"assignments": [one.assignments[0].model_copy(update={"est_mb": 100})]})
    assert Reconciler.calibration_sample(tiny, _mem(CUDA0=50)) is None  # estimate <= 0
    # planned with factor 1.5: est_mb 1500 is unscaled back to 1000 before the context is removed
    scaled = one.model_copy(update={"mem_factor": 1.5, "assignments": [
        one.assignments[0].model_copy(update={"est_mb": 1500})]})
    assert Reconciler.calibration_sample(scaled, _mem(CUDA0=900)) == (900, 872)


async def test_calibration_converges_on_the_true_ratio_when_replanned_with_the_factor():
    # Regression: sampling against estimates already multiplied by the factor measured r/f, and the
    # EMA then settled on sqrt(r) (1.2 for r = 1.44), under-reserving VRAM. Every replica here is
    # planned with the current factor, as the reconciler does, and the factor must stay at r.
    rec, store, clock = make_reconciler()
    base = put_replica(store, "m-1")
    rec.client.memory = _mem(CUDA0=872 * 1.44)
    for _ in range(6):
        f = store.mem_factor("m")
        a = base.placement.assignments[0].model_copy(update={"est_mb": math.ceil(1000 * f)})
        r = base.model_copy(update={"placement": base.placement.model_copy(
            update={"mem_factor": f, "assignments": [a]})})
        await rec._calibrate(r, SPEC, "http://a")
    assert store.mem_factor("m") == pytest.approx(1.44, abs=0.005)


async def test_launch_calibrates_from_head_memory_and_emits_event(mock_health):
    client = FakeClient()
    client.memory = _mem(CUDA0=976)  # 976 / (1000 - 128) = 1.119
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "ready" and client.memory_calls == [("http://10.0.0.1:7070", f"{r.replica_id}-head")]
    cal = store.get_calibration("m")
    assert cal["samples"] == 1 and cal["factor"] == pytest.approx(976 / 872)
    [ev] = _cal_events(store)
    assert ev.level == "info" and ev.model == "m"
    assert ev.message == "m: measured 976 MB vs estimated 872 MB, planning factor 1.12"
    await rec.shutdown()


async def test_calibration_ema_clamp_and_event_threshold():
    rec, store, clock = make_reconciler()
    r = put_replica(store, "m-1")

    async def feed(total):
        rec.client.memory = _mem(CUDA0=total)
        await rec._calibrate(r, SPEC, "http://a")

    await feed(872 * 1.2)
    assert store.get_calibration("m")["factor"] == pytest.approx(1.2)  # first sample sets it
    assert len(_cal_events(store)) == 1
    await feed(872 * 1.4)  # EMA alpha 0.5
    cal = store.get_calibration("m")
    assert cal["factor"] == pytest.approx(1.3) and cal["samples"] == 2
    assert len(_cal_events(store)) == 2  # 1.2 -> 1.3 is more than 5 %
    await feed(872 * 1.31)  # 1.3 -> 1.305: below the threshold, still stored
    assert len(_cal_events(store)) == 2 and store.get_calibration("m")["samples"] == 3
    # wild samples: the planning factor is clamped to [0.9, 2.0] (the raw EMA is kept)
    for _ in range(8):
        await feed(872 * 10)
    assert store.mem_factor("m") == 2.0 and store.get_calibration("m")["factor"] > 2.0
    for _ in range(16):
        await feed(872 * 0.1)
    assert store.mem_factor("m") == 0.9
    assert _cal_events(store)[0].message.endswith("planning factor 0.90")  # newest first


async def test_first_sample_below_one_is_clamped_at_point_nine():
    rec, store, clock = make_reconciler()
    rec.client.memory = _mem(CUDA0=400)
    await rec._calibrate(put_replica(store, "m-1"), SPEC, "http://a")
    assert store.mem_factor("m") == 0.9
    assert _cal_events(store)[0].message.endswith("planning factor 0.90")  # 1.0 -> 0.9 is a > 5 % change


@pytest.mark.parametrize("memory", [None, {}, {"devices": {}}, _mem(CUDA1=900), RuntimeError("agent down"),
                                    {"devices": "junk"}])
async def test_missing_or_broken_memory_data_is_ignored_and_never_fails_the_launch(memory, mock_health):
    client = FakeClient()
    client.memory = memory
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    assert store.list_replicas()[0].state == "ready"
    assert store.get_calibration("m") is None and not _cal_events(store)
    await rec.shutdown()


async def test_calibration_store_failure_does_not_fail_the_launch(mock_health):
    client = FakeClient()
    client.memory = _mem(CUDA0=976)
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    store.put_calibration = boom
    await rec.tick()
    await settle(rec)
    assert store.list_replicas()[0].state == "ready" and rec._launches == {}
    await rec.shutdown()


async def test_calibration_runs_only_for_replicas_that_became_ready(mock_health):
    client = FakeClient()
    client.memory = _mem(CUDA0=976)
    client.fail_start_on = "-head"
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    assert store.list_replicas()[0].state == "failed" and client.memory_calls == []


async def test_calibrated_factor_is_passed_to_planner_and_rankers_only_when_not_one(mock_health):
    seen = {"plan": [], "rank": []}
    real_plan, real_rank = make_planner(), make_ranker()

    def planner(meta, spec, nodes, rid, port_alloc, exclude_nodes=frozenset(), **kw):
        seen["plan"].append({k: v for k, v in kw.items() if k != "occupants"})
        return real_plan(meta, spec, nodes, rid, port_alloc)

    def ranker(meta, spec, nodes, occupants=(), limit=5, extra=(), **kw):
        seen["rank"].append(kw)
        return real_rank(meta, spec, nodes, occupants, limit)

    rec, store, clock = make_reconciler(planner=planner)
    rec.ranker = ranker
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await rec.plan_for(SPEC)
    await rec.rank_for(SPEC, 3)
    assert seen == {"plan": [{}], "rank": [{}]}  # unmeasured: the call is exactly as before
    store.put_calibration("m", 1.25, 2)
    seen["plan"].clear(), seen["rank"].clear()
    await rec.plan_for(SPEC)
    await rec.rank_for(SPEC, 3)
    await rec.simulate([SPEC])  # simulation ranks through _ranker_for
    assert seen["plan"] == [{"mem_factor": 1.25}]
    assert len(seen["rank"]) >= 2 and all(kw == {"mem_factor": 1.25} for kw in seen["rank"])
    other = SPEC.model_copy(update={"name": "other"})  # another model is unaffected
    seen["rank"].clear()
    await rec.rank_for(other, 1)
    assert seen["rank"] == [{}]
    store.put_calibration("m", 0.3, 3)  # raw 0.3 plans as 0.9
    seen["plan"].clear()
    await rec.plan_for(SPEC)
    assert seen["plan"] == [{"mem_factor": 0.9}]


async def test_calibrated_factor_reaches_rebalance_and_preemption_rankers():
    rec, store, clock = _rb_rig()
    calls = []
    base = rec.ranker

    def ranker(meta, spec, nodes, occupants=(), limit=5, extra=(), **kw):
        calls.append(kw)
        return base(meta, spec, nodes, occupants=occupants, limit=limit, extra=extra)

    rec.ranker = ranker
    store.put_calibration("m", 1.5, 1)
    await rec.rebalance_candidates()
    assert calls and all(kw == {"mem_factor": 1.5} for kw in calls)
    calls.clear()
    store.put_model(_hi())
    store.put_calibration("hi", 1.1, 1)
    await rec.rank_with_preemption(_hi(), 1)
    assert all(kw == {"mem_factor": 1.1} for kw in calls)


async def test_orphaned_launch_after_a_hard_restart_is_failed_and_relaunched(mock_health):
    # Real-cluster regression: after `docker kill` of the coordinator mid-launch the replica stayed
    # "launching" forever and, counted as active, blocked the model from being launched again.
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    put_replica(store, "m-old", state="launching", now=clock())  # left by the killed coordinator
    await rec.tick()
    await settle(rec)
    old = store.get_replica("m-old")
    assert old.state == "failed" and "restarted during launch" in old.error
    assert ("stop", "http://10.0.0.1:7070", "m-old-head") in rec.client.calls  # half-started engines
    assert [r.state for r in store.list_replicas() if r.replica_id != "m-old"] == ["ready"]
    await rec.shutdown()


async def test_launch_in_flight_is_not_an_orphan(mock_health):
    rec, store, clock = make_reconciler()
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await rec.tick()  # spawns a launch task; the record is "launching" with its task
    rid = store.list_replicas()[0].replica_id
    await rec._fail_orphaned_launches(rec._node_map(), clock())
    assert store.get_replica(rid).state in ("launching", "ready")
    await settle(rec)
    assert store.get_replica(rid).state == "ready"
    await rec.shutdown()


# ---------------------------------------------------------------- stop order
async def test_stop_head_first_waits_for_the_head_before_rpc_servers():
    # Regression: all engines were stopped at once; ggml-rpc-server exits instantly while
    # llama-server still frees its remote buffers on shutdown, so every stop of a split replica
    # ended in "Remote RPC server crashed" -> SIGABRT -> a core dump (seen in the sim cluster logs).
    from gpupool.coordinator.reconciler import stop_head_first
    log = []

    async def stop(node, eid):
        log.append(("begin", eid))
        await asyncio.sleep(0.05 if eid.endswith("-head") else 0)
        log.append(("end", eid))

    await stop_head_first([("b", "r-rpc-CUDA0"), ("a", "r-head"), ("c", "r-rpc-CUDA1")], stop)
    head_end = log.index(("end", "r-head"))
    assert all(log.index(("begin", e)) > head_end for e in ("r-rpc-CUDA0", "r-rpc-CUDA1"))
    await stop_head_first([], stop)  # nothing to do is fine
    log.clear()
    await stop_head_first([("b", "r-rpc-CUDA0")], stop)  # rollback before the head started
    assert log == [("begin", "r-rpc-CUDA0"), ("end", "r-rpc-CUDA0")]


class _SlowHeadClient(FakeClient):
    """The agent answers a stop only once the process exited; a head takes a while."""

    def __init__(self):
        super().__init__()
        self.events = []

    async def stop_engine(self, url, engine_id):
        self.events.append(("begin", engine_id))
        await asyncio.sleep(0.05 if engine_id.endswith("-head") else 0)
        self.events.append(("end", engine_id))
        return await super().stop_engine(url, engine_id)


def _rpc_started_after_head_ended(events):
    head_end = events.index(("end", "m-1-head"))
    return all(i > head_end for i, e in enumerate(events) if e == ("begin", "m-1-rpc-CUDA0"))


async def test_stopping_a_split_replica_waits_for_the_head_first():
    client = _SlowHeadClient()
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"), node("b"))
    r = put_replica(store, "m-1", rpc_node="b")
    nodes = {n.report.node_id: n for n in store.list_nodes()}
    await rec._stop_engines(r, nodes, clock())
    assert _rpc_started_after_head_ended(client.events), client.events
    await rec.shutdown()


async def test_rollback_waits_for_the_head_first():
    client = _SlowHeadClient()
    rec, store, clock = make_reconciler(client=client)
    await rec._rollback([("http://b:7070", "m-1-rpc-CUDA0"), ("http://a:7070", "m-1-head")])
    assert _rpc_started_after_head_ended(client.events), client.events
    await rec.shutdown()


async def test_launch_sends_attention_and_batches(mock_health):
    client = FakeClient()
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC.model_copy(update={"flash_attn": "on", "ubatch": 1024, "batch": 4096, "replicas": 1}))
    await rec.tick()
    await settle(rec)
    hs = next(c[3] for c in client.calls if c[0] == "start")
    assert (hs.flash_attn, hs.ubatch, hs.batch) == ("on", 1024, 4096)
    await rec.shutdown()


async def test_launch_downloads_the_model_while_rpc_engines_start(mock_health):
    # The download must already be running when the RPC engines start: here it only finishes
    # once an RPC engine was started, which a download-after-engines order would never allow.
    class Overlap(FakeClient):
        def __init__(self):
            super().__init__()
            self.rpc_started = asyncio.Event()

        async def start_engine(self, url, spec):
            if spec.kind == "rpc":
                await asyncio.sleep(0)  # let the download begin first
                self.rpc_started.set()
            return await super().start_engine(url, spec)

        async def ensure_model(self, url, name, source):
            self.calls.append(("ensure", url, name, source))
            await asyncio.wait_for(self.rpc_started.wait(), 2)
            return "/cache/" + name

    client = Overlap()
    rec, store, clock = make_reconciler(planner=make_planner(rpc=True), client=client)
    beat(store, clock, node("a"), node("b"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    assert store.list_replicas()[0].state == "ready"
    assert client.kinds() == ["ensure", "start", "start"]  # head last, after both
    await rec.shutdown()


async def test_failed_rpc_start_cancels_the_download_and_rolls_back(mock_health):
    class SlowDownload(FakeClient):
        cancelled = False

        async def start_engine(self, url, spec):
            await asyncio.sleep(0.01)  # the download is under way when this start fails
            return await super().start_engine(url, spec)

        async def ensure_model(self, url, name, source):
            self.calls.append(("ensure", url, name, source))
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    client = SlowDownload()
    client.fail_start_on = "-rpc-CUDA0"
    rec, store, clock = make_reconciler(planner=make_planner(rpc=True), client=client)
    beat(store, clock, node("a"), node("b"))
    store.put_model(SPEC)
    await rec.tick()
    await asyncio.wait_for(settle(rec), 5)
    r = store.list_replicas()[0]
    assert r.state == "failed" and "boom" in (r.error or "")
    assert client.cancelled and client.engines == {}
    assert not any(c[0] == "start" and c[2].endswith("-head") for c in client.calls)
    await rec.shutdown()


async def test_launch_starts_one_rpc_engine_for_a_shared_endpoint(mock_health):
    from gpupool.common.models import DeviceAssignment, Placement
    from gpupool.coordinator.reconciler import engine_ids

    def planner(meta, spec, nodes, rid, port_alloc, **kw):
        port = port_alloc("b")
        ep = f"10.0.0.2:{port}"
        asg = [DeviceAssignment(node_id="a", device_id="CUDA0", llama_device="CUDA0", layers=2, est_mb=10),
               DeviceAssignment(node_id="b", device_id="CUDA0", llama_device="RPC0", rpc_endpoint=ep,
                                layers=2, est_mb=10),
               DeviceAssignment(node_id="b", device_id="CUDA1", llama_device="RPC1", rpc_endpoint=ep,
                                layers=2, est_mb=10)]
        return Placement(model=spec.name, replica_id=rid, tier="multi_node", head_node="a",
                         head_port=port_alloc("a"), assignments=asg, tensor_split=[2.0] * 3, est_total_mb=30)

    client = FakeClient()
    rec, store, clock = make_reconciler(planner=planner, client=client)
    beat(store, clock, node("a"), node("b"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "ready"
    starts = [c[3] for c in client.calls if c[0] == "start"]
    rpc = [s for s in starts if s.kind == "rpc"]
    head = next(s for s in starts if s.kind == "server")
    assert len(rpc) == 1 and rpc[0].devices == ["CUDA0", "CUDA1"]
    assert rpc[0].engine_id == f"{r.replica_id}-rpc-CUDA0"
    assert head.rpc_endpoints == [r.placement.assignments[1].rpc_endpoint]  # listed once
    assert head.devices == ["CUDA0", "RPC0", "RPC1"]
    assert engine_ids(r) == [("a", f"{r.replica_id}-head"), ("b", f"{r.replica_id}-rpc-CUDA0")]
    await rec.shutdown()


async def test_draft_goes_to_the_heads_gpu_even_when_remote_devices_come_first(mock_health):
    from gpupool.common.models import DeviceAssignment, Placement

    def planner(meta, spec, nodes, rid, port_alloc, **kw):
        ep = f"10.0.0.2:{port_alloc('b')}"
        asg = [DeviceAssignment(node_id="b", device_id="CUDA0", llama_device="RPC0", rpc_endpoint=ep,
                                layers=2, est_mb=10),
               DeviceAssignment(node_id="a", device_id="CUDA0", llama_device="CUDA0", layers=2, est_mb=10)]
        return Placement(model=spec.name, replica_id=rid, tier="multi_node", head_node="a",
                         head_port=port_alloc("a"), assignments=asg, tensor_split=[2.0, 2.0], est_total_mb=20)

    client = FakeClient()
    rec, store, clock = make_reconciler(planner=planner, client=client)
    rec.meta_for = _meta_by_source()
    beat(store, clock, node("a"), node("b"))
    store.put_model(DSPEC)
    await rec.tick()
    await settle(rec)
    assert store.list_replicas()[0].state == "ready"
    hs = next(c[3] for c in client.calls if c[0] == "start" and c[3].kind == "server")
    assert hs.devices == ["RPC0", "CUDA0"] and hs.draft_device == "CUDA0"
    await rec.shutdown()


async def test_launch_with_mtp_sends_spec_type_and_draft_tokens(mock_health):
    client = FakeClient()
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC.model_copy(update={"speculative": "mtp", "draft_n_max": 3, "replicas": 1}))
    await rec.tick()
    await settle(rec)
    hs = next(c[3] for c in client.calls if c[0] == "start")
    assert (hs.spec_type, hs.draft_n_max, hs.draft_model_path) == ("mtp", 3, None)
    assert client.kinds() == ["ensure", "start"]  # no second file
    await rec.shutdown()


async def test_rpc_engine_that_exits_fails_the_launch_with_its_log(mock_health):
    class DyingRpc(FakeClient):
        async def get_engine(self, url, engine_id):
            st = await super().get_engine(url, engine_id)
            if st is not None and "-rpc-" in engine_id:
                st = st.model_copy(update={"state": "failed", "log_tail": ["CUDA error: out of memory"]})
            return st

    client = DyingRpc()
    rec, store, clock = make_reconciler(planner=make_planner(rpc=True), client=client)
    beat(store, clock, node("a"), node("b"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "failed" and "out of memory" in r.error
    assert not any(c[0] == "start" and c[2].endswith("-head") for c in client.calls)
    assert client.engines == {}  # rolled back


async def test_head_that_exits_before_health_fails_the_launch():
    client = FakeClient()
    client.head_state = "exited"
    with respx.mock(assert_all_called=False) as m:
        m.get(url__regex=HEALTH).mock(side_effect=httpx.ConnectError("not listening"))
        rec, store, clock = make_reconciler(client=client)
        beat(store, clock, node("a"))
        store.put_model(SPEC)
        await rec.tick()
        await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "failed" and "head engine exited" in r.error
    assert client.engines == {}


async def test_mixed_llama_builds_raise_one_warning_per_change():
    rec, store, clock = make_reconciler()

    def report(node_id, version):
        n = node(node_id)
        n.llama_version = version
        return n

    beat(store, clock, report("a", "b11413"), report("b", "b11413"), report("c", "unknown"))
    await rec.tick()
    assert not [e for e in store.list_events() if e.kind == "llama_version_mismatch"]
    beat(store, clock, report("b", "b11342"))
    await rec.tick()
    await rec.tick()  # same situation: no second event
    ev = [e for e in store.list_events() if e.kind == "llama_version_mismatch"]
    assert len(ev) == 1 and ev[0].level == "warning"
    assert "b11342: b" in ev[0].message and "b11413: a" in ev[0].message
    await rec.shutdown()
