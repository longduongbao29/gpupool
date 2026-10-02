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
