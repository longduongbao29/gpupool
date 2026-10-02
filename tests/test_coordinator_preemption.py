import copy

from gpupool.common.models import (
    Device, DeviceAssignment, ModelMeta, ModelSpec, NodeReport, Placement, ReplicaRecord,
)
from gpupool.coordinator.preemption import (
    Candidate, apply_placement, find_victims, free_replicas, placement_occupants, without_occupants,
)
from gpupool.scheduler.placement import rank

MB = 2**20


def dev(did, usable, total=1000, uuid=None):
    return Device(device_id=did, kind="cuda", name=did, total_mb=total, free_mb=usable,
                  usable_mb=usable, uuid=uuid)


def node(nid, *devs):
    return NodeReport(node_id=nid, agent_url=f"http://{nid}:7070", host=nid, devices=list(devs),
                      engines=[], llama_version="b1", models=[], ts=0.0)


def rec(rid, nid="n1", did="CUDA0", mb=400, uuid=None, created=0.0, model="low"):
    a = DeviceAssignment(node_id=nid, device_id=did, llama_device="CUDA0", layers=1, est_mb=mb,
                         device_uuid=uuid)
    pl = Placement(model=model, replica_id=rid, tier="single_gpu", head_node=nid, head_port=9000,
                   assignments=[a], tensor_split=[1.0], est_total_mb=mb)
    return ReplicaRecord(replica_id=rid, model=model, placement=pl, state="ready",
                         created_at=created, updated_at=created)


def cand(r, priority=1, preemptible=True, above_min=False, busy=0.0):
    return Candidate(replica=r, priority=priority, preemptible=preemptible, above_min=above_min, busy=busy)


META = ModelMeta(arch="llama", n_layers=4, n_embd=64, n_head=1, n_head_kv=1, head_dim=64,
                 layer_bytes=[100 * MB] * 4, other_bytes=0, output_bytes=0)
SPEC = ModelSpec(name="hi", source="x", ctx_size=256, priority=5)


def fake_rank(need):
    """Fits when some device has `need` MB usable."""
    def fn(meta, spec, nodes, occupants=(), limit=5):
        ok = any(d.usable_mb >= need for n in nodes for d in n.devices)
        return [object()] if ok else []
    return fn


def test_free_and_apply_roundtrip_and_caps():
    reports = [node("n1", dev("CUDA0", 100, total=1000))]
    r = rec("r1", mb=300)
    freed = free_replicas(reports, [r])
    assert freed[0].devices[0].usable_mb == 400
    assert reports[0].devices[0].usable_mb == 100  # input untouched
    assert free_replicas(reports, [rec("r2", mb=5000)])[0].devices[0].usable_mb == 1000  # capped
    assert apply_placement(freed, r.placement)[0].devices[0].usable_mb == 100
    assert apply_placement(reports, rec("r3", mb=5000).placement)[0].devices[0].usable_mb == 0  # floor


def test_unknown_node_ignored():
    reports = [node("n1", dev("CUDA0", 100))]
    assert free_replicas(reports, [rec("r1", nid="gone")])[0].devices[0].usable_mb == 100


def test_uuid_matching_after_device_id_shift():
    # GPU-b was CUDA1 when planned and is CUDA0 now; GPU-a is gone.
    reports = [node("n1", dev("CUDA0", 100, uuid="GPU-b"))]
    assert free_replicas(reports, [rec("r1", did="CUDA1", uuid="GPU-b", mb=200)])[0].devices[0].usable_mb == 300
    gone = rec("r2", did="CUDA0", uuid="GPU-a", mb=200)  # same id, different card: no match
    assert free_replicas(reports, [gone])[0].devices[0].usable_mb == 100
    # No uuid on the report: fall back to device_id.
    plain = [node("n1", dev("CUDA0", 100))]
    assert free_replicas(plain, [gone])[0].devices[0].usable_mb == 300
    assert apply_placement(reports, rec("r3", did="CUDA1", uuid="GPU-b", mb=60).placement)[0].devices[0].usable_mb == 40


def test_disabled_device_stays_zero():
    reports = [node("n1", dev("CUDA0", 0), dev("CUDA1", 0))]
    out = free_replicas(reports, [rec("r1", did="CUDA0"), rec("r2", did="CUDA1")],
                        disabled={("n1", "CUDA0")})
    assert [d.usable_mb for d in out[0].devices] == [0, 400]


def test_occupants_helpers():
    r = rec("r1", mb=250)
    occ = placement_occupants(r.placement, busy=0.5)
    assert [(o.node_id, o.device_id, o.model, o.replica_id, o.est_mb, o.busy) for o in occ] == [
        ("n1", "CUDA0", "low", "r1", 250, 0.5)]
    mixed = occ + placement_occupants(rec("r2").placement)
    assert [o.replica_id for o in without_occupants(mixed, {"r1"})] == ["r2"]


def test_eligibility():
    reports = [node("n1", dev("CUDA0", 0))]
    rank_fn = fake_rank(400)
    r = rec("r1")
    assert find_victims(META, SPEC, reports, [], [cand(r, priority=5)], rank_fn) is None  # equal
    assert find_victims(META, SPEC, reports, [], [cand(r, priority=9)], rank_fn) is None  # higher
    assert find_victims(META, SPEC, reports, [], [cand(r, preemptible=False)], rank_fn) is None
    assert find_victims(META, SPEC, reports, [], [], rank_fn) is None
    assert find_victims(META, SPEC, reports, [], [cand(r)], rank_fn) == [r]


def test_ordering_above_min_then_busy_then_newest():
    # One big card; every victim frees 400, so need 400*k takes exactly the first k in order.
    reports = [node("n1", dev("CUDA0", 0, total=10**6))]
    a = rec("a", created=1)
    b = rec("b", created=2)
    c = rec("c", created=3)
    d = rec("d", created=3)
    cands = [cand(a, busy=0.1), cand(b, busy=0.1, above_min=True), cand(c, busy=0.0), cand(d, busy=0.0)]
    # above_min b first; then c, d (busy 0, same age, replica_id order); then a.
    for k, want in [(1, ["b"]), (2, ["b", "c"]), (3, ["b", "c", "d"]), (4, ["b", "c", "d", "a"])]:
        got = find_victims(META, SPEC, reports, [], cands, fake_rank(400 * k))
        assert [r.replica_id for r in got] == want
    # busy ascending outranks age; newest first among equals; above_min outranks idleness.
    assert find_victims(META, SPEC, reports, [], [cand(a, busy=0.5), cand(b, busy=0.2)],
                        fake_rank(400))[0].replica_id == "b"
    assert find_victims(META, SPEC, reports, [], [cand(a), cand(b)], fake_rank(400))[0].replica_id == "b"
    assert find_victims(META, SPEC, reports, [], [cand(a, above_min=True, busy=0.9), cand(b)],
                        fake_rank(400))[0].replica_id == "a"


def test_minimisation_drops_unnecessary_victim():
    # small comes first (above_min) and frees 100, big frees 900; only big alone reaches 900.
    reports = [node("n1", dev("CUDA0", 0, total=2000))]
    small = rec("small", mb=100, created=5)
    big = rec("big", mb=900, created=1)
    got = find_victims(META, SPEC, reports, [], [cand(small, above_min=True), cand(big)], fake_rank(900))
    assert got == [big]


def test_none_when_impossible():
    reports = [node("n1", dev("CUDA0", 0))]
    assert find_victims(META, SPEC, reports, [], [cand(rec("r1"))], fake_rank(10**5)) is None


def test_rank_call_count_bounded_and_inputs_untouched():
    reports = [node("n1", dev("CUDA0", 0, total=10**6))]
    occ = placement_occupants(rec("r0").placement)
    before = (copy.deepcopy(reports), copy.deepcopy(occ))
    calls = []
    inner = fake_rank(10**5)

    def counting(*a, **k):
        calls.append(1)
        return inner(*a, **k)
    cs = [cand(rec(f"r{i}", mb=100, created=i)) for i in range(6)]
    assert find_victims(META, SPEC, reports, occ, cs, counting) is None
    assert len(calls) <= 6 * 6
    assert (reports, occ) == before


def test_real_rank_frees_enough():
    # The model needs several hundred MB; the card has 100 usable, the victim holds 3000.
    reports = [node("n1", dev("CUDA0", 100, total=4000))]
    victim = rec("v", mb=3000)
    got = find_victims(META, SPEC, reports, placement_occupants(victim.placement), [cand(victim)], rank)
    assert got == [victim]
    assert rank(META, SPEC, reports, limit=1) == []
