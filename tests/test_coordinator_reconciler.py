import httpx
import pytest
import respx

from gpupool.common.models import EngineStatus, ModelSpec
from gpupool.coordinator import preemption
from gpupool.coordinator.reconciler import _Realloc
from gpupool.coordinator.store import ServerRecord
from tests.test_coordinator_helpers import (
    SPEC, Clock, FakeAutoscaler, FakeClient, dev, make_cfg, make_planner, make_ranker, make_reconciler, node,
    put_replica, settle,
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
