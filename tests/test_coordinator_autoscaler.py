"""Autoscaler: sustained-busy scale up/down, idle unload, cold start, /metrics scraping."""
from __future__ import annotations

import httpx
import pytest
import respx

from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import AutoscalePolicy, ModelSpec, Placement, ReplicaRecord
from gpupool.coordinator.autoscaler import Autoscaler, _Scrape, bounds, parse_metrics
from gpupool.coordinator.events import Notifier
from gpupool.coordinator.store import Store
from tests.test_coordinator_helpers import Clock, node


class Rig:
    def __init__(self, store=None, clock=None, **cfg_kw):
        self.store = store or Store(":memory:")
        self.clock = clock or Clock(1000.0)
        self.out: dict[str, int] = {}
        self.woken = 0
        cfg = CoordinatorConfig(poll_s=2.0, **cfg_kw)
        self.a = Autoscaler(self.store, cfg, lambda rid: self.out.get(rid, 0),
                            Notifier(self.store, clock=self.clock), self.clock,
                            wake=self._wake)

    def _wake(self):
        self.woken += 1

    def model(self, name="m", **kw) -> ModelSpec:
        spec = ModelSpec(name=name, source="x.gguf", **kw)
        self.store.put_model(spec)
        return spec

    def replica(self, rid, model="m", state="ready", port=9001, node_id="a") -> None:
        pl = Placement(model=model, replica_id=rid, tier="single_gpu", head_node=node_id, head_port=port,
                       assignments=[], tensor_split=[1.0], est_total_mb=1, est_decode_tps=50.0)
        t = self.clock()
        self.store.put_replica(ReplicaRecord(replica_id=rid, model=model, placement=pl, state=state,
                                             created_at=t, updated_at=t))

    def feed(self, rid, processing, deferred=0.0):
        """Plant a fresh scrape (the HTTP path has its own tests)."""
        self.a._scrapes[rid] = _Scrape(ts=self.clock(), processing=processing, deferred=deferred, tps=40.0)

    def tick(self, secs=0.0, fresh=True):
        """Advance the clock and evaluate. `fresh` re-stamps the planted scrapes, as if the poll
        loop had re-read the same values; stale-scrape tests pass fresh=False."""
        self.clock.t += secs
        if fresh:
            for sc in self.a._scrapes.values():
                sc.ts = self.clock()
        self.a.evaluate()

    def kinds(self):
        return [e.kind for e in reversed(self.store.list_events())]


def autoscaled(rig, **kw):
    kw.setdefault("replicas", 1)
    kw.setdefault("min_replicas", 1)
    kw.setdefault("max_replicas", 3)
    kw.setdefault("parallel", 4)
    kw.setdefault("autoscale", AutoscalePolicy(target_busy=0.7, up_after_s=30, down_after_s=300))
    return rig.model(**kw)


def test_bounds_fixed():
    assert bounds(ModelSpec(name="m", source="x", replicas=2)) == (2, 2)


def test_fixed_model_is_replicas_and_never_changes():
    r = Rig()
    spec = r.model(replicas=2)
    assert r.a.desired(spec) == 2
    r.replica("r1")
    r.feed("r1", 4)
    r.tick(1000)
    assert r.a.desired(spec) == 2
    assert r.kinds() == []
    assert r.a.view("m")["state"] == "fixed"


def test_stopped_is_zero_and_resets_state():
    r = Rig()
    spec = autoscaled(r, max_replicas=3)
    r.a._state.clear()
    assert r.a.desired(spec) == 1
    stopped = spec.model_copy(update={"replicas": 0})
    r.store.put_model(stopped)
    assert r.a.desired(stopped) == 0
    assert "m" not in r.a._state
    assert r.a.view("m")["state"] == "stopped"


def test_scale_up_only_after_sustained_busy():
    r = Rig()
    spec = autoscaled(r)
    r.replica("r1")
    assert r.a.desired(spec) == 1
    r.feed("r1", 4)  # 100% busy
    r.tick()
    r.tick(29)
    assert r.a.desired(spec) == 1  # 29 s < up_after_s
    r.feed("r1", 4)
    r.tick(1)
    assert r.a.desired(spec) == 2
    assert r.kinds() == ["scaled_up"]
    assert "busy 100% > 70% for 30 s" in r.store.list_events()[0].message


def test_busy_condition_break_resets_timer():
    r = Rig()
    spec = autoscaled(r)
    r.replica("r1")
    r.a.desired(spec)
    r.feed("r1", 4)
    r.tick()
    r.tick(20)
    r.feed("r1", 1)  # 25%: neither up nor down
    r.tick(5)
    r.feed("r1", 4)
    r.tick(10)  # 35 s since first busy, but only 10 s since it resumed
    assert r.a.desired(spec) == 1
    r.feed("r1", 4)
    r.tick(20)
    assert r.a.desired(spec) == 1  # 20 s
    r.feed("r1", 4)
    r.tick(10)
    assert r.a.desired(spec) == 2


def test_no_second_scale_up_while_launching_and_cap_at_max():
    r = Rig()
    spec = autoscaled(r, max_replicas=2)
    r.replica("r1")
    r.a.desired(spec)
    r.feed("r1", 4)
    r.tick()
    r.tick(30)
    assert r.a.desired(spec) == 2
    r.replica("r2", state="launching")
    r.feed("r1", 4)
    r.tick(100)
    assert r.a.desired(spec) == 2  # waiting for r2 (and at max anyway)
    assert r.a.view("m")["state"] == "steady"  # ready 1 + launching 1 == desired 2
    # r2 ready, still saturated: capped at max
    r.replica("r2", state="ready")
    r.feed("r1", 4)
    r.feed("r2", 4)
    r.tick(100)
    r.feed("r1", 4)
    r.feed("r2", 4)
    r.tick(100)
    assert r.a.desired(spec) == 2
    assert r.kinds() == ["scaled_up"]


def test_waits_for_launching_replica_before_next_step():
    r = Rig()
    spec = autoscaled(r, max_replicas=3)
    r.replica("r1")
    r.a.desired(spec)
    r.replica("r2", state="pending")
    r.feed("r1", 4)
    r.tick()
    r.feed("r1", 4)
    r.tick(60)
    assert r.a.desired(spec) == 1  # held back by the pending replica
    r.store.set_replica_state("r2", "ready")
    r.feed("r1", 4)
    r.feed("r2", 4)
    r.tick(1)
    assert r.a.desired(spec) == 2


def test_scale_down_after_sustained_idle_respects_floor():
    r = Rig()
    spec = autoscaled(r, min_replicas=1, max_replicas=3)
    r.replica("r1")
    r.replica("r2")
    r.a._state["m"] = r.a._state_for(spec, r.clock())
    r.a._state["m"].desired = 2
    r.feed("r1", 0)
    r.feed("r2", 1)  # avg 12.5% < 35%
    r.tick()
    r.feed("r1", 0)
    r.feed("r2", 0)
    r.tick(299)
    assert r.a.desired(spec) == 2
    r.feed("r1", 0)
    r.feed("r2", 0)
    r.tick(1)
    assert r.a.desired(spec) == 1
    assert r.kinds() == ["scaled_down"]
    for _ in range(5):
        r.feed("r1", 0)
        r.tick(400)
    assert r.a.desired(spec) == 1  # floor


def test_queued_triggers_scale_up_even_at_low_busy():
    r = Rig()
    spec = autoscaled(r)
    r.replica("r1")
    r.a.desired(spec)
    r.feed("r1", 1, deferred=2)
    r.tick()
    r.feed("r1", 1, deferred=2)
    r.tick(30)
    assert r.a.desired(spec) == 2
    assert "requests queueing for 30 s" in r.store.list_events()[0].message
    assert r.a.view("m")["queued"] is True


def test_queued_blocks_scale_down():
    r = Rig()
    spec = autoscaled(r)
    r.a._state_for(spec, r.clock()).desired = 2
    r.replica("r1")
    r.feed("r1", 0, deferred=1)
    r.tick()
    r.feed("r1", 0, deferred=1)
    r.tick(1000)
    assert r.a.desired(spec) == 3 or r.a.desired(spec) == 2  # went up, never down
    assert "scaled_down" not in r.kinds()


def test_fallback_to_router_when_scrape_stale_or_failed():
    r = Rig()
    spec = autoscaled(r)
    r.replica("r1")
    r.a.desired(spec)
    r.a._scrapes["r1"] = _Scrape(ts=r.clock() - 100, processing=0.0, deferred=0.0)  # stale, says idle
    r.out["r1"] = 4
    assert r.a.view("m")["avg_busy"] == 1.0
    r.tick(fresh=False)
    r.out["r1"] = 4
    r.tick(30, fresh=False)
    assert r.a.desired(spec) == 2
    # failed scrape: also ignored even if recent
    r.a._scrapes["r1"] = _Scrape(ts=r.clock(), processing=0.0, deferred=0.0, ok=False)
    r.out["r1"] = 9  # more than parallel -> queued
    v = r.a.view("m")
    assert v["avg_busy"] == 1.0 and v["queued"] is True
    assert v["replicas"][0]["metrics_ok"] is False


def test_avg_busy_none_without_ready_replicas():
    r = Rig()
    autoscaled(r)
    r.replica("r1", state="launching")
    v = r.a.view("m")
    assert v["avg_busy"] is None and v["ready"] == 0 and v["launching"] == 1


def test_idle_unload_then_cold_start_once():
    r = Rig()
    spec = r.model(replicas=1, min_replicas=0, max_replicas=1, idle_unload_s=600)
    r.replica("r1")
    assert r.a.desired(spec) == 1
    assert r.a.can_cold_start("m")
    r.tick(599)
    assert r.a.desired(spec) == 1
    r.out["r1"] = 1  # in-flight request blocks unloading
    r.tick(10)
    assert r.a.desired(spec) == 1
    r.out["r1"] = 0
    r.tick(1)
    assert r.a.desired(spec) == 0
    assert r.kinds() == ["unloaded_idle"]
    assert r.a.view("m")["state"] == "unloaded"
    assert r.a.note_request("m") is True
    assert r.woken == 1
    assert r.a.desired(spec) == 1
    assert r.kinds() == ["unloaded_idle", "cold_start"]
    assert "loading m for an incoming request" in r.store.list_events()[0].message
    assert r.a.note_request("m") is False
    assert r.woken == 1
    r.tick(1)
    assert r.a.desired(spec) == 1  # idle timer restarted by the request


def test_lo_raised_above_zero_clamps_up_while_unloaded():
    r = Rig()
    spec = r.model(replicas=1, min_replicas=0, max_replicas=1, idle_unload_s=10)
    r.replica("r1")
    r.a.desired(spec)
    r.tick(11)
    assert r.a.desired(spec) == 0
    raised = spec.model_copy(update={"min_replicas": 2, "max_replicas": 2})
    r.store.put_model(raised)
    assert r.a.desired(raised) == 2
    assert not r.a.can_cold_start("m")


def test_note_request_unknown_and_stopped_never_raise():
    r = Rig()
    assert r.a.note_request("nope") is False
    r.model("s", replicas=0, min_replicas=0, max_replicas=1)
    assert r.a.note_request("s") is False
    assert not r.a.can_cold_start("s")
    assert not r.a.can_cold_start("nope")
    r.model("fixed", replicas=1)
    assert r.a.note_request("fixed") is False
    assert not r.a.can_cold_start("fixed")


def test_event_failure_does_not_break_evaluation():
    r = Rig()
    spec = autoscaled(r)

    def boom(*a, **k):
        raise RuntimeError("db down")

    r.a.notifier.emit = boom  # type: ignore[method-assign]
    r.replica("r1")
    r.a.desired(spec)
    r.feed("r1", 4)
    r.tick()
    r.feed("r1", 4)
    r.tick(30)
    assert r.a.desired(spec) == 2


def test_parse_metrics_with_junk():
    text = (
        "# HELP llamacpp:requests_processing Number of requests processing.\n"
        "# TYPE llamacpp:requests_processing gauge\n"
        "llamacpp:requests_processing 2\n"
        "llamacpp:requests_deferred 1\n"
        "llamacpp:predicted_tokens_seconds 41.5\n"
        "llamacpp:prompt_tokens_total 123\n"
        "garbage\n"
        "llamacpp:requests_deferred notanumber\n"
        "\n"
        "llamacpp:requests_processing\n"
    )
    assert parse_metrics(text) == {"llamacpp:requests_processing": 2.0, "llamacpp:requests_deferred": 1.0,
                                   "llamacpp:predicted_tokens_seconds": 41.5}
    assert parse_metrics("") == {}
    assert parse_metrics('llamacpp:requests_processing{slot="0"} 3') == {"llamacpp:requests_processing": 3.0}


async def test_scrape_once_reads_ready_heads_and_survives_failures():
    r = Rig()
    autoscaled(r)
    r.store.upsert_node(node("a", host="10.0.0.1"), r.clock())
    r.store.upsert_node(node("b", host="10.0.0.2"), r.clock() - 100)  # dead node: not scraped
    r.replica("r1", port=9001, node_id="a")
    r.replica("r2", port=9002, node_id="a")
    r.replica("r3", port=9003, node_id="b")
    r.replica("r4", port=9004, node_id="a", state="launching")
    r.replica("r5", port=9005, node_id="ghost")
    with respx.mock(assert_all_called=False) as mock:
        ok = mock.get("http://10.0.0.1:9001/metrics").mock(return_value=httpx.Response(
            200, text="llamacpp:requests_processing 3\nllamacpp:requests_deferred 2\n"
                      "llamacpp:predicted_tokens_seconds 33.0\n"))
        mock.get("http://10.0.0.1:9002/metrics").mock(side_effect=httpx.ConnectError("down"))
        dead = mock.get("http://10.0.0.2:9003/metrics").mock(return_value=httpx.Response(200, text=""))
        other = mock.get("http://10.0.0.1:9004/metrics").mock(return_value=httpx.Response(200, text=""))
        await r.a.scrape_once()
    assert ok.called and not dead.called and not other.called
    s1 = r.a._scrapes["r1"]
    assert (s1.processing, s1.deferred, s1.tps, s1.ok) == (3.0, 2.0, 33.0, True)
    assert r.a._scrapes["r2"].ok is False
    assert "r3" not in r.a._scrapes
    # a replica that is gone is dropped on the next scrape
    r.store.set_replica_state("r1", "stopped")
    with respx.mock(assert_all_called=False) as mock:
        mock.get("http://10.0.0.1:9002/metrics").mock(side_effect=httpx.ConnectError("down"))
        await r.a.scrape_once()
    assert "r1" not in r.a._scrapes
    await r.a.aclose()


async def test_scrape_failure_after_success_keeps_values_but_flags():
    r = Rig()
    autoscaled(r)
    r.store.upsert_node(node("a", host="10.0.0.1"), r.clock())
    r.replica("r1", port=9001)
    with respx.mock() as mock:
        route = mock.get("http://10.0.0.1:9001/metrics")
        route.mock(return_value=httpx.Response(200, text="llamacpp:requests_processing 1\n"))
        await r.a.scrape_once()
        route.mock(return_value=httpx.Response(500))
        await r.a.scrape_once()
    s = r.a._scrapes["r1"]
    assert s.ok is False and s.processing == 1.0
    await r.a.aclose()


def test_view_shape():
    r = Rig()
    spec = autoscaled(r)
    r.replica("r1")
    r.replica("r2", state="launching")
    r.a.desired(spec)
    r.feed("r1", 2, deferred=0)
    v = r.a.view("m")
    assert set(v) == {"model", "min", "max", "desired", "ready", "launching", "avg_busy", "queued", "idle_s",
                      "state", "last_decision", "replicas"}
    assert (v["min"], v["max"], v["desired"], v["ready"], v["launching"]) == (1, 3, 1, 1, 1)
    assert v["avg_busy"] == 0.5 and v["queued"] is False and v["idle_s"] == 0.0
    assert v["state"] == "scaling_down"  # desired 1 < ready 1 + launching 1
    assert v["last_decision"] is None
    assert v["replicas"] == [{
        "replica_id": "r1", "busy": 0.5, "requests_processing": 2.0, "requests_deferred": 0.0,
        "measured_decode_tps": 40.0, "est_decode_tps": 50.0, "metrics_ok": True}]
    assert r.a.view("ghost")["state"] == "stopped"


@pytest.mark.parametrize("desired,ready,launching,state", [(2, 1, 0, "scaling_up"), (1, 1, 0, "steady")])
def test_view_states(desired, ready, launching, state):
    r = Rig()
    spec = autoscaled(r)
    for i in range(ready):
        r.replica(f"r{i}")
    r.a.desired(spec)
    r.a._state["m"].desired = desired
    assert r.a.view("m")["state"] == state


def test_peek_matches_desired_without_touching_state():
    rig = Rig()
    spec = autoscaled(rig, min_replicas=2, max_replicas=4)
    assert rig.a.peek(spec) == 2 and rig.a._state == {}  # no state yet: the floor, none created
    rig.a._state_for(spec, rig.clock()).desired = 3
    assert rig.a.peek(spec) == rig.a.desired(spec) == 3
    # a hypothetical spec is clamped by ITS bounds, and the stored state is left alone
    assert rig.a.peek(spec.model_copy(update={"max_replicas": 2})) == 2
    assert rig.a.peek(spec.model_copy(update={"min_replicas": 4, "max_replicas": 4})) == 4
    assert rig.a._state["m"].desired == 3
    assert rig.a.peek(spec.model_copy(update={"replicas": 0})) == 0 and "m" in rig.a._state  # desired() would pop it


def test_peek_unloaded_model_stays_zero_and_new_model_is_not_registered():
    rig = Rig()
    spec = autoscaled(rig, min_replicas=0, max_replicas=2, idle_unload_s=10)
    rig.a._state_for(spec, rig.clock()).desired = 0
    assert rig.a.peek(spec) == 0
    fresh = ModelSpec(name="fresh", source="x.gguf", min_replicas=0)
    assert rig.a.peek(fresh) == 1 and "fresh" not in rig.a._state  # floor is max(min, 1)


async def test_scrape_uses_injected_liveness_rule():
    # stale report but the shared rule says alive (no failed polls): still scraped
    r = Rig()
    autoscaled(r)
    r.store.upsert_node(node("a", host="10.0.0.1"), r.clock() - 100)
    r.replica("r1", port=9001, node_id="a")
    r.a.node_alive = lambda n, now: True
    with respx.mock() as mock:
        route = mock.get("http://10.0.0.1:9001/metrics").mock(
            return_value=httpx.Response(200, text="llamacpp:requests_processing 1\n"))
        await r.a.scrape_once()
    assert route.called and r.a._scrapes["r1"].ok
    await r.a.aclose()


# ---------------------------------------------------------------- persistence across a restart
def _restart(r: Rig) -> Rig:
    return Rig(store=r.store, clock=r.clock)


def test_unloaded_model_stays_unloaded_after_restart(tmp_path):
    r = Rig(store=Store(tmp_path / "s.db"))
    spec = r.model(replicas=1, min_replicas=0, max_replicas=1, idle_unload_s=600)
    r.replica("r1")
    assert r.a.desired(spec) == 1
    r.tick(600)
    assert r.a.desired(spec) == 0
    r2 = _restart(r)
    assert r2.a.desired(spec) == 0  # not loaded again just because the coordinator restarted
    assert r2.a.view("m")["state"] == "unloaded"
    assert r2.a.note_request("m") is True  # a request still wakes it
    assert r2.a.desired(spec) == 1
    r3 = _restart(r2)
    assert r3.a.desired(spec) == 1  # and the cold start is saved too
    assert r3.a._state["m"].last_decision["action"] == "cold_start"


def test_scaled_up_count_and_last_request_survive_restart(tmp_path):
    r = Rig(store=Store(tmp_path / "s.db"))
    spec = autoscaled(r)
    r.replica("r1")
    r.feed("r1", 4)
    r.tick()
    r.tick(30)
    assert r.a.desired(spec) == 2
    r.clock.t += 5
    r.a.note_request("m")
    r2 = _restart(r)
    assert r2.a.desired(spec) == 2
    st = r2.a._state["m"]
    assert st.last_decision["action"] == "scaled_up"
    assert st.up_since is None and st.down_since is None  # timers are dropped on purpose


def test_idle_timer_survives_restart_and_requests_refresh_it_slowly(tmp_path):
    r = Rig(store=Store(tmp_path / "s.db"))
    spec = r.model(replicas=1, min_replicas=0, max_replicas=1, idle_unload_s=600)
    r.replica("r1")
    r.a.desired(spec)
    r.clock.t += 100
    r.a.note_request("m")  # first request: saved
    saved = r.store.get_state("autoscaler:m")["last_request"]
    r.clock.t += 10
    r.a.note_request("m")  # within SAVE_IDLE_S: not written again
    assert r.store.get_state("autoscaler:m")["last_request"] == saved
    r.clock.t += 100
    r.a.note_request("m")
    assert r.store.get_state("autoscaler:m")["last_request"] == r.clock()
    r2 = _restart(r)
    r2.a.desired(spec)
    assert r2.a._state["m"].last_request == r.clock()


def test_stopping_or_deleting_a_model_drops_saved_state(tmp_path):
    r = Rig(store=Store(tmp_path / "s.db"))
    spec = r.model(replicas=1, min_replicas=0, max_replicas=1, idle_unload_s=600)
    r.replica("r1")
    r.a.desired(spec)
    r.tick(600)
    assert r.store.get_state("autoscaler:m") is not None
    stopped = spec.model_copy(update={"replicas": 0})
    r.store.put_model(stopped)
    assert r.a.desired(stopped) == 0
    assert r.store.get_state("autoscaler:m") is None
    r.store.put_model(spec)  # started again: back at its floor, not the stale unloaded state
    assert _restart(r).a.desired(spec) == 1


def test_store_failure_does_not_break_scaling(tmp_path):
    r = Rig()
    spec = r.model(replicas=1, min_replicas=0, max_replicas=1, idle_unload_s=600)
    r.replica("r1")
    r.a.desired(spec)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    r.store.put_state = boom
    r.tick(600)
    assert r.a.desired(spec) == 0  # in-memory state still decided
