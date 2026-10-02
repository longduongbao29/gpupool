import httpx
import pytest
import pytest
import respx

from gpupool.common.models import EngineStatus
from gpupool.coordinator.store import ServerRecord
from tests.test_coordinator_helpers import (
    SPEC, Clock, FakeClient, dev, make_cfg, make_planner, make_reconciler, node, put_replica, settle,
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

    def planner(meta, spec, nodes, rid, port_alloc, exclude_nodes=frozenset()):
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
