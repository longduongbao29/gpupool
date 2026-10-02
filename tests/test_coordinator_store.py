import time

import pytest

from gpupool.coordinator.store import Store
from tests.test_coordinator_helpers import SPEC, node, put_replica


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
