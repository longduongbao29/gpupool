import json

import httpx
import respx

from gpupool.common.models import EngineStatus
from gpupool.coordinator.events import Notifier
from gpupool.coordinator.store import Store
from tests.test_coordinator_helpers import (
    SPEC, Clock, FakeClient, dev, make_planner, make_reconciler, node, put_replica, settle,
)
from tests.test_coordinator_reconciler import beat, mock_health  # noqa: F401


def kinds(store):
    return [e.kind for e in reversed(store.list_events(limit=100))]


async def test_node_dies_offline_failed_realloc_started_then_done(mock_health):
    rec, store, clock = make_reconciler()
    store.put_model(SPEC)
    beat(store, clock, node("a"), node("b"))
    put_replica(store, "m-1", head="a", now=clock())
    beat(store, clock, node("a", engines=[EngineStatus(engine_id="m-1-head", kind="server",
                                                       state="running", port=9000)]), node("b"))
    await rec.tick()
    assert kinds(store) == []  # healthy: no events
    clock.t += 11
    beat(store, clock, node("b"))  # a stops reporting
    await rec.tick()
    await settle(rec)
    assert store.get_replica("m-1").state == "failed"
    evs = {e.kind: e for e in store.list_events(limit=100)}
    assert kinds(store) == ["node_offline", "realloc_started", "realloc_done"]
    assert "Server a went offline" in evs["node_offline"].message and "m" in evs["node_offline"].message
    assert evs["node_offline"].level == "warning" and evs["node_offline"].node_id == "a"
    assert "m-1" in evs["realloc_started"].message and "b/CUDA0" in evs["realloc_started"].message
    assert evs["realloc_done"].level == "info"
    # staying dead emits nothing more; coming back emits node_online once
    for _ in range(3):
        clock.t += 2
        beat(store, clock, node("b"))
        await rec.tick()
    assert kinds(store).count("node_offline") == 1
    beat(store, clock, node("a"))
    await rec.tick()
    await rec.tick()
    assert kinds(store)[-1] == "node_online" and kinds(store).count("node_online") == 1
    await rec.shutdown()


async def test_gpu_vanishes_from_live_node(mock_health):
    rec, store, clock = make_reconciler()
    store.put_model(SPEC)
    beat(store, clock, node("a", devices=[dev("CUDA0"), dev("CUDA1")]))
    put_replica(store, "m-1", head="a", now=clock())
    clock.t += 1
    beat(store, clock, node("a", devices=[dev("CUDA1")], engines=[
        EngineStatus(engine_id="m-1-head", kind="server", state="running", port=9000)]))
    await rec.tick()
    await settle(rec)
    r = store.get_replica("m-1")
    assert r.state == "failed" and "a/CUDA0" in r.error
    ks = kinds(store)
    assert ks[0] == "gpu_missing" and "realloc_started" in ks and ks[-1] == "realloc_done"
    ev = [e for e in store.list_events(limit=100) if e.kind == "gpu_missing"][0]
    assert ev.level == "error" and ev.node_id == "a" and ev.model == "m"
    await rec.shutdown()


async def test_engine_crash_event_has_exit_code_and_log():
    rec, store, clock = make_reconciler()
    put_replica(store, "m-1", now=clock())
    clock.t += 1
    beat(store, clock, node("a", engines=[EngineStatus(
        engine_id="m-1-head", kind="server", state="exited", port=9000, exit_code=139,
        log_tail=["CUDA error: out of memory"])]))
    await rec.tick()
    ev = store.list_events()[-1]
    assert ev.kind == "engine_crashed" and "139" in ev.message and "out of memory" in ev.message


async def test_nofit_after_loss_emits_one_realloc_failed_then_recovers(mock_health):
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=10**6))
    store.put_model(SPEC)
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    clock.t += 1
    beat(store, clock, node("a", devices=[]))
    for _ in range(5):
        await rec.tick()
    ks = kinds(store)
    assert ks.count("realloc_failed") == 1 and "realloc_done" not in ks
    assert "nothing fits" in [e for e in store.list_events() if e.kind == "realloc_failed"][0].message
    rec.planner = make_planner()
    beat(store, clock, node("a"))
    await rec.tick()
    await settle(rec)
    ks = kinds(store)
    assert ks[-2:] == ["realloc_started", "realloc_done"] and ks.count("realloc_failed") == 1
    await rec.shutdown()


async def test_launch_failed_event(mock_health):
    client = FakeClient()
    client.fail_start_on = "-head"
    rec, store, clock = make_reconciler(client=client)
    beat(store, clock, node("a"))
    store.put_model(SPEC)
    await rec.tick()
    await settle(rec)
    ev = store.list_events()[0]
    assert ev.kind == "launch_failed" and ev.level == "error" and "boom" in ev.message


async def test_stopping_a_model_clears_pending_realloc():
    rec, store, clock = make_reconciler(planner=make_planner(est_mb=10**6))
    store.put_model(SPEC)
    beat(store, clock, node("a"))
    await rec.tick()
    assert "m" in rec._realloc
    store.put_model(SPEC.model_copy(update={"replicas": 0}))
    await rec.tick()
    assert rec._realloc == {}


# ---------------------------------------------------------------- notifier / webhook
async def test_webhook_only_for_warning_and_error():
    store = Store(":memory:")
    n = Notifier(store, "http://hook/x", clock=Clock())
    with respx.mock() as m:
        route = m.post("http://hook/x").mock(return_value=httpx.Response(200))
        n.emit("info", "k", "fine")
        n.emit("warning", "k", "careful", node_id="a")
        n.emit("error", "k", "bad", model="m")
        await n.flush()
    assert route.call_count == 2
    body = json.loads(route.calls[0].request.content)
    assert body["text"] == "careful" and body["content"] == "careful"
    assert body["event"]["kind"] == "k" and body["event"]["node_id"] == "a"
    assert len(store.list_events()) == 3
    await n.aclose()


async def test_no_webhook_when_url_unset():
    n = Notifier(Store(":memory:"), "")
    with respx.mock() as m:
        n.emit("error", "k", "bad")
        await n.flush()
        assert m.calls.call_count == 0


async def test_webhook_failure_never_breaks_the_tick(caplog):
    rec, store, clock = make_reconciler()
    rec.notifier = Notifier(store, "http://hook/x", clock=clock)
    beat(store, clock, node("a"))
    put_replica(store, "m-1", now=clock())
    store.put_model(SPEC)
    clock.t += 1
    beat(store, clock, node("a", devices=[]))
    with respx.mock() as m:
        hook = m.post("http://hook/x").mock(return_value=httpx.Response(500))
        await rec.tick()
        await rec.notifier.flush()
        assert hook.called
    assert store.get_replica("m-1").state == "failed"
    assert len([r for r in caplog.records if "webhook delivery failed" in r.getMessage()]) == 1
    await rec.shutdown()


async def test_webhook_connection_error_is_swallowed():
    n = Notifier(Store(":memory:"), "http://hook/x")
    with respx.mock() as m:
        m.post("http://hook/x").mock(side_effect=httpx.ConnectError("down"))
        n.emit("error", "k", "bad")
        await n.flush()
    await n.aclose()
