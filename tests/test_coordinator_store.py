import time

import pytest

from gpupool.coordinator.store import Store, gpu_key
from tests.test_coordinator_helpers import SPEC, dev, node, put_replica


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path):
    s = Store(":memory:" if request.param == "memory" else tmp_path / "sub" / "c.db")
    yield s
    s.close()


def test_nodes_roundtrip_and_upsert(store):
    store.upsert_node(node("a"), 1.0)
    store.upsert_node(node("a", host="9.9.9.9"), 2.0)
    store.upsert_node(node("b"), 3.0)
    nodes = store.list_nodes()
    assert [n.report.node_id for n in nodes] == ["a", "b"]
    assert nodes[0].last_seen == 2.0 and nodes[0].report.host == "9.9.9.9"


def test_models_roundtrip(store):
    assert store.get_model("m") is None
    store.put_model(SPEC)
    assert store.get_model("m") == SPEC
    store.put_model(SPEC.model_copy(update={"replicas": 3}))
    assert store.list_models()[0].replicas == 3
    store.delete_model("m")
    assert store.list_models() == []


def test_replicas_roundtrip_and_filters(store):
    r1 = put_replica(store, "m-1", "m", "ready")
    put_replica(store, "m-2", "m", "failed")
    put_replica(store, "o-1", "other", "ready")
    assert store.get_replica("m-1") == r1
    assert store.get_replica("nope") is None
    assert {r.replica_id for r in store.list_replicas(model="m")} == {"m-1", "m-2"}
    assert {r.replica_id for r in store.list_replicas(states={"ready"})} == {"m-1", "o-1"}
    assert [r.replica_id for r in store.list_replicas(model="m", states={"failed"})] == ["m-2"]
    assert store.list_replicas(states=set()) == []


def test_set_replica_state_updates_blob_column_and_timestamp(store):
    put_replica(store, "m-1", state="launching", now=1.0)
    before = time.time()
    store.set_replica_state("m-1", "failed", "oops")
    rec = store.get_replica("m-1")
    assert (rec.state, rec.error) == ("failed", "oops") and rec.updated_at >= before
    assert [r.replica_id for r in store.list_replicas(states={"failed"})] == ["m-1"]
    store.set_replica_state("ghost", "failed")  # no-op, no crash


def test_servers_roundtrip_and_delete_cascades(store):
    from gpupool.coordinator.store import ServerRecord

    assert store.get_server("a") is None
    store.add_server(ServerRecord(node_id="b", agent_url="http://b:1", added_at=2.0))
    store.add_server(ServerRecord(node_id="a", agent_url="http://a:1", added_at=1.0))
    assert [s.node_id for s in store.list_servers()] == ["a", "b"]
    assert store.get_server("a").agent_url == "http://a:1"
    store.upsert_node(node("a"), 1.0)
    store.upsert_node(node("b"), 1.0)
    store.set_gpu_enabled("a", "CUDA0", False)
    store.set_gpu_enabled("b", "CUDA0", False)
    store.delete_server("a")
    assert store.get_server("a") is None
    assert [n.report.node_id for n in store.list_nodes()] == ["b"]
    assert store.gpu_flags() == {("b", "CUDA0"): False}


def test_gpu_flags_default_enabled_and_toggle(store):
    assert store.gpu_flags() == {}
    store.set_gpu_enabled("a", "CUDA0", False)
    assert store.gpu_flags() == {("a", "CUDA0"): False}
    store.set_gpu_enabled("a", "CUDA0", True)
    assert store.gpu_flags() == {("a", "CUDA0"): True}


def test_gpu_key_prefers_uuid():
    assert gpu_key(dev("CUDA1", uuid="GPU-1")) == "GPU-1"
    assert gpu_key(dev("CUDA1")) == "CUDA1"


def test_upsert_migrates_legacy_flag_rows_to_uuid(store):
    store.set_gpu_enabled("a", "CUDA0", True)
    store.set_gpu_enabled("a", "CUDA1", False)
    store.set_gpu_enabled("b", "CUDA0", False)  # other node: untouched
    v = store.version
    store.upsert_node(node("a", devices=[dev("CUDA0", uuid="GPU-0"), dev("CUDA1", uuid="GPU-1")]), 1.0)
    assert store.gpu_flags() == {("a", "GPU-0"): True, ("a", "GPU-1"): False, ("b", "CUDA0"): False}
    assert store.version == v + 1  # the node write bumps, the flag rewrite adds nothing
    # numbering shifts later: the flags stay with their cards and nothing is rewritten again
    store.upsert_node(node("a", devices=[dev("CUDA0", uuid="GPU-1")]), 2.0)
    assert store.gpu_flags() == {("a", "GPU-0"): True, ("a", "GPU-1"): False, ("b", "CUDA0"): False}


def test_migration_keeps_existing_uuid_row_and_skips_uuidless_reports(store):
    store.set_gpu_enabled("a", "CUDA0", False)  # legacy
    store.set_gpu_enabled("a", "GPU-0", True)  # already has a uuid row: it wins
    store.upsert_node(node("a", devices=[dev("CUDA0", uuid="GPU-0")]), 1.0)
    assert store.gpu_flags() == {("a", "CUDA0"): False, ("a", "GPU-0"): True}
    store.upsert_node(node("b", devices=[dev("CUDA0")]), 1.0)  # no uuids: legacy rows stay
    store.set_gpu_enabled("b", "CUDA0", False)
    store.upsert_node(node("b", devices=[dev("CUDA0")]), 2.0)
    assert store.gpu_flags()[("b", "CUDA0")] is False


def test_events_order_filter_read_and_prune(store, monkeypatch):
    import gpupool.coordinator.store as st

    e1 = store.add_event(1.0, "info", "k", "one")
    e2 = store.add_event(2.0, "error", "k", "two", node_id="a", model="m")
    assert [e.id for e in store.list_events()] == [e2.id, e1.id]
    assert store.list_events()[0].node_id == "a" and store.list_events()[0].model == "m"
    assert [e.id for e in store.list_events(after_id=e1.id)] == [e2.id]
    assert [e.id for e in store.list_events(limit=1)] == [e2.id]
    assert store.unread_count() == 2
    store.mark_read(e1.id)
    assert store.unread_count() == 1 and store.list_events()[1].read is True
    monkeypatch.setattr(st, "EVENTS_KEEP", 3)
    for i in range(5):
        store.add_event(float(i), "info", "k", f"m{i}")
    assert len(store.list_events(limit=100)) == 3


def test_prune_replicas_keeps_newest_terminal_per_model(store):
    for i in range(5):
        put_replica(store, f"m-{i}", "m", "failed" if i % 2 else "stopped", now=1000.0 + i)
    for i in range(3):
        put_replica(store, f"o-{i}", "other", "stopped", now=2000.0 + i)
    assert store.prune_replicas(2) == 3 + 1
    assert [r.replica_id for r in store.list_replicas(model="m")] == ["m-3", "m-4"]
    assert [r.replica_id for r in store.list_replicas(model="other")] == ["o-1", "o-2"]
    assert store.prune_replicas(2) == 0


def test_prune_replicas_never_deletes_non_terminal(store):
    for i, st in enumerate(["pending", "launching", "ready", "draining"]):
        put_replica(store, f"m-{i}", "m", st, now=1000.0 + i)
    for i in range(3):
        put_replica(store, f"m-old{i}", "m", "failed", now=10.0 + i)  # older than the live ones
    assert store.prune_replicas(0) == 3
    assert {r.state for r in store.list_replicas()} == {"pending", "launching", "ready", "draining"}


def test_prune_replicas_tie_breaks_on_replica_id(store):
    for rid in ("m-a", "m-b", "m-c"):
        put_replica(store, rid, "m", "failed", now=1000.0)
    assert store.prune_replicas(1) == 2
    assert [r.replica_id for r in store.list_replicas()] == ["m-c"]


def test_version_bumps_on_routing_writes_only(store):
    from gpupool.coordinator.store import ServerRecord

    v = store.version
    assert isinstance(v, int)
    store.list_nodes(); store.list_models(); store.list_replicas(); store.list_servers()
    assert store.version == v  # reads never bump

    def bumped(fn):
        nonlocal v
        fn()
        assert store.version > v, fn
        v = store.version

    bumped(lambda: store.upsert_node(node("a"), 1.0))
    bumped(lambda: store.put_model(SPEC))
    bumped(lambda: put_replica(store, "m-1"))
    bumped(lambda: store.set_replica_state("m-1", "stopped"))
    bumped(lambda: store.prune_replicas(0))
    bumped(lambda: store.add_server(ServerRecord(node_id="a", agent_url="http://a", added_at=0.0)))
    bumped(lambda: store.delete_server("a"))
    bumped(lambda: store.delete_model("m"))
    # no-ops do not bump
    store.set_replica_state("missing", "stopped")
    assert store.prune_replicas(5) == 0
    assert store.version == v


# ---------------------------------------------------------------- control state / calibration
def test_control_state_roundtrip_overwrite_delete_and_no_version_bump():
    from gpupool.coordinator.store import Store
    s = Store(":memory:")
    v = s.version
    assert s.get_state("k") is None
    s.put_state("k", {"a": 1, "b": [1, 2]})
    assert s.get_state("k") == {"a": 1, "b": [1, 2]}
    s.put_state("k", {"a": 2})
    assert s.get_state("k") == {"a": 2}
    s.delete_state("k")
    s.delete_state("k")  # idempotent
    assert s.get_state("k") is None
    assert s.version == v  # routing does not depend on it


def test_control_state_survives_reopen_and_ignores_garbage(tmp_path):
    from gpupool.coordinator.store import Store
    path = tmp_path / "s.db"
    s = Store(path)
    s.put_state("k", {"x": 1})
    s._conn.execute("INSERT INTO control_state(key, value, updated_at) VALUES('bad', '{nope', 0)")
    s._conn.commit()
    s.close()
    s2 = Store(path)
    assert s2.get_state("k") == {"x": 1}
    assert s2.get_state("bad") is None
    s2.close()


def test_calibration_roundtrip_planning_clamp_and_delete_model():
    from gpupool.common.models import ModelSpec
    from gpupool.coordinator.store import Store, planning_factor
    s = Store(":memory:")
    assert s.get_calibration("m") is None and s.mem_factor("m") == 1.0
    s.put_model(ModelSpec(name="m", source="x"))
    s.put_calibration("m", 1.12, 1, now=5.0)
    s.put_calibration("m", 1.2, 2, now=6.0)
    assert s.get_calibration("m") == {"factor": 1.2, "samples": 2, "updated_at": 6.0}
    assert s.mem_factor("m") == 1.2
    s.put_calibration("m", 0.5, 3)
    assert s.get_calibration("m")["factor"] == 0.5 and s.mem_factor("m") == 0.9  # never below 0.9
    s.put_calibration("m", 5.0, 4)
    assert s.mem_factor("m") == 2.0
    assert planning_factor(None) == 1.0
    s.put_state("autoscaler:m", {"desired": 0})
    s.delete_model("m")
    assert s.get_calibration("m") is None and s.get_state("autoscaler:m") is None


def test_calibration_resets_when_the_estimator_changes(tmp_path, monkeypatch):
    import gpupool.coordinator.store as store_mod
    db = tmp_path / "c.db"
    s = store_mod.Store(db)
    s.put_calibration("m", 0.5, 3)
    s._conn.close()
    assert store_mod.Store(db).get_calibration("m") is not None  # same estimator: kept
    monkeypatch.setattr(store_mod, "ESTIMATOR_VERSION", store_mod.ESTIMATOR_VERSION + 1)
    s = store_mod.Store(db)
    assert s.get_calibration("m") is None and s.mem_factor("m") == 1.0
