import pytest

from gpupool.common.models import Device, ModelMeta, ModelSpec, NodeReport, Occupant
from gpupool.scheduler.estimate import total_need_mb
from gpupool.scheduler.placement import NoFit, plan, rank
from gpupool.scheduler.scoring import ETA, HOP_S, est_decode_tps

MB = 2**20


def make_meta(n_layers=8, layer_mb=100, out_mb=100):
    return ModelMeta(
        arch="llama", n_layers=n_layers, n_embd=1024, n_head=8, n_head_kv=8, head_dim=128,
        layer_bytes=[layer_mb * MB] * n_layers, other_bytes=out_mb * 2 * MB,
        output_bytes=out_mb * MB,
    )


META = make_meta()
SPEC = ModelSpec(name="m", source="x", ctx_size=512)
NEED = total_need_mb(META, 512)


def dev(did, usable, bw=None, kind="cuda"):
    return Device(device_id=did, kind=kind, name=did, total_mb=usable + 500, free_mb=usable + 500,
                  usable_mb=usable, bandwidth_gbps=bw)


def node(nid, *devs):
    return NodeReport(node_id=nid, agent_url=f"http://{nid}:7070", host=nid, devices=list(devs),
                      engines=[], llama_version="b1", models=[], ts=0.0)


def occ(nid, did, model="other", busy=0.0):
    return Occupant(node_id=nid, device_id=did, model=model, replica_id="x", est_mb=100, busy=busy)


def where(pl):
    return [(a.node_id, a.device_id) for a in pl.assignments]


def go(spec=SPEC, nodes=(), occupants=()):
    return plan(META, spec, list(nodes), "r", lambda n: 9000, occupants=occupants)


def test_second_replica_spreads_over_gpus():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100), dev("CUDA1", NEED + 100, 100))]
    pl = go(nodes=nodes, occupants=[occ("a", "CUDA0", model="m")])
    assert where(pl) == [("a", "CUDA1")]
    assert any("another replica" in r for r in go(nodes=nodes, occupants=[occ("a", "CUDA1", "m")]).reasons) is False


def test_spread_node_prefers_other_node_and_none_ignores():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100), dev("CUDA1", NEED + 100, 100)),
             node("b", dev("CUDA0", NEED + 100, 100))]
    occ_a = [occ("a", "CUDA0", "m")]
    spec = SPEC.model_copy(update={"spread": "node"})
    assert where(go(spec, nodes, occ_a)) == [("b", "CUDA0")]
    # spread gpu is happy with another GPU of the same node
    assert where(go(SPEC, nodes, occ_a))[0][1] != "CUDA0"
    # spread none: only the 10-point sharing penalty remains, a free GPU still wins
    none = SPEC.model_copy(update={"spread": "none"})
    assert where(go(none, nodes, occ_a))[0] != ("a", "CUDA0")


def test_stacks_when_nothing_else_fits():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100))]
    pl = go(nodes=nodes, occupants=[occ("a", "CUDA0", "m")])
    assert where(pl) == [("a", "CUDA0")]
    assert any("another replica" in r for r in pl.reasons)


def test_new_model_avoids_busy_gpu():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100), dev("CUDA1", NEED + 100, 100))]
    pl = go(nodes=nodes, occupants=[occ("a", "CUDA0", busy=0.4)])
    assert where(pl) == [("a", "CUDA1")]
    # both occupied: the less busy one wins and the reason names the sharing
    pl = go(nodes=nodes, occupants=[occ("a", "CUDA0", busy=0.9), occ("a", "CUDA1", busy=0.2)])
    assert where(pl) == [("a", "CUDA1")]
    assert any("shares a/CUDA1 with other (busy 20%)" == r for r in pl.reasons)


def test_faster_gpu_wins_same_size():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100)), node("b", dev("CUDA0", NEED + 100, 400))]
    pl = go(nodes=nodes)
    assert where(pl) == [("b", "CUDA0")]
    assert pl.reasons[0].startswith("fastest option")


def test_best_fit_among_equal_speed():
    nodes = [node("a", dev("CUDA0", NEED + 5000, 100)),
             node("b", dev("CUDA0", NEED + 10, 100), dev("CUDA1", NEED + 800, 100))]
    assert where(go(nodes=nodes)) == [("b", "CUDA0")]


def test_tps_formula_two_devices_one_hop():
    meta = make_meta(8, 100, 100)
    a, b = dev("CUDA0", 10000, 200), dev("CUDA1", 10000, 100)
    got = est_decode_tps(meta, [(a, 4), (b, 4)], n_rpc=1)
    t = (400 * MB) / (200e9 * ETA) + (500 * MB) / (100e9 * ETA) + HOP_S  # output on the last
    assert got == pytest.approx(1 / t)


def test_split_placement_reports_tps_and_hops():
    half = NEED // 2 + 300
    nodes = [node("a", dev("CUDA0", half, 100)), node("b", dev("CUDA0", half, 100))]
    pl = go(nodes=nodes)
    assert pl.tier == "multi_node"
    t = (800 * MB + 100 * MB) / (100e9 * ETA) + HOP_S
    assert pl.est_decode_tps == round(1 / t, 1)
    assert any("network hop" in r for r in pl.reasons) and any("split over 2" in r for r in pl.reasons)


def test_unknown_bandwidth_fallbacks():
    cuda_a, cuda_b = dev("CUDA0", 5000, None), dev("CUDA1", 5000, 80)
    cpu = dev("CPU", 5000, None, kind="cpu")
    meta = make_meta(4, 100, 100)
    nodes = [node("a", cuda_a, cuda_b)]
    # unknown CUDA == lowest known CUDA (80)
    got = est_decode_tps(meta, [(cuda_a, 4)], 0, 80.0)
    assert got == pytest.approx(est_decode_tps(meta, [(cuda_b, 4)], 0, 80.0))
    # no CUDA bandwidth known anywhere: 100
    assert est_decode_tps(meta, [(cuda_a, 4)]) == pytest.approx(
        est_decode_tps(meta, [(dev("X", 1, 100), 4)]))
    # cpu without bandwidth: 25
    assert est_decode_tps(meta, [(cpu, 4)]) == pytest.approx(
        est_decode_tps(meta, [(dev("Y", 1, 25, kind="cpu"), 4)]))
    # mixed old/new agents rank without crashing, unknown GPU ties with the slowest known
    pl = rank(META, SPEC, [node("a", dev("CUDA0", NEED + 100, None)), node("b", dev("CUDA0", NEED + 100, 80))])
    assert len(pl) == 2 and pl[0].est_decode_tps == pl[1].est_decode_tps
    assert nodes  # keep the fixture used


def test_rank_sorted_limited_with_reasons():
    nodes = [node(f"n{i}", dev("CUDA0", NEED + 100 + 50 * i, 100 + 50 * i)) for i in range(4)]
    out = rank(META, SPEC, nodes, limit=3)
    assert len(out) == 3
    assert [p.score for p in out] == sorted((p.score for p in out), reverse=True)
    assert all(p.reasons and 1 <= len(p.reasons) <= 4 and p.replica_id == "" for p in out)
    assert all(p.head_port == 0 for p in out)
    assert out[0].assignments[0].node_id == "n3"
    assert out[0] == go(nodes=nodes).model_copy(update={"replica_id": "", "head_port": 0})
    assert rank(META, SPEC, [node("a", dev("CUDA0", 50))]) == []
    with pytest.raises(NoFit):
        go(nodes=[node("a", dev("CUDA0", 50))])


def test_multi_node_only_when_nothing_single_node_fits():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100), dev("CUDA1", NEED + 100, 100)),
             node("b", dev("CUDA0", NEED + 100, 100))]
    assert {p.tier for p in rank(META, SPEC, nodes)} <= {"single_gpu", "single_node"}
    half = NEED // 2 + 300
    nodes = [node("a", dev("CUDA0", half, 100)), node("b", dev("CUDA0", half, 100))]
    assert [p.tier for p in rank(META, SPEC, nodes)] == ["multi_node"]


def test_candidates_cover_every_fitting_gpu_and_node():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100), dev("CUDA1", NEED + 100, 100)),
             node("b", dev("CUDA0", NEED + 100, 100))]
    tiers = [p.tier for p in rank(META, SPEC, nodes, limit=20)]
    assert tiers.count("single_gpu") == 3 and tiers.count("single_node") == 1


def test_gpu_first_still_holds():
    # a GPU fits: CPU devices never appear in any candidate
    nodes = [node("a", dev("CUDA0", NEED + 100, 100), dev("CPU", 99999, kind="cpu"))]
    assert all(a.device_id == "CUDA0" for p in rank(META, SPEC, nodes) for a in p.assignments)


def _on(*nodes_):
    """A placement for the model on exactly these nodes (stands in for a running replica)."""
    return plan(META, SPEC, list(nodes_), "run-1", lambda n: 9100)


def test_extra_on_slow_gpu_ranks_below_fast_candidate():
    slow, fast = node("a", dev("CUDA0", NEED + 100, 100)), node("b", dev("CUDA0", NEED + 100, 400))
    cur = _on(slow)
    out = rank(META, SPEC, [slow, fast], extra=[cur])
    assert [where(p) for p in out] == [[("b", "CUDA0")], [("a", "CUDA0")]]
    got = out[1]
    # kept identity, re-scored in this pass (perf term is relative to the fast GPU)
    assert got.replica_id == "run-1" and got.head_port == 9100
    assert got.est_decode_tps < out[0].est_decode_tps and got.score < out[0].score
    assert out[0].replica_id == ""


def test_extra_ties_with_equal_alternative():
    a, b = node("a", dev("CUDA0", NEED + 100, 100)), node("b", dev("CUDA0", NEED + 100, 100))
    out = rank(META, SPEC, [a, b], extra=[_on(a)])
    # same speed, same waste, same devices: scores tie; the candidate on "a" is dropped as a
    # duplicate of the extra, so only the alternative on "b" remains next to it
    assert sorted(where(p)[0] for p in out) == [("a", "CUDA0"), ("b", "CUDA0")]
    assert out[0].score == out[1].score


def test_extra_colocation_excludes_itself_only_if_caller_omits_it():
    a = node("a", dev("CUDA0", NEED + 100, 100))
    cur = _on(a)
    own = occ("a", "CUDA0", model="m")
    # the caller is responsible for leaving the extra's own replica out of occupants
    clean = rank(META, SPEC, [a], extra=[cur])[0]
    dirty = rank(META, SPEC, [a], occupants=[own], extra=[cur])[0]
    assert not any("shares" in r for r in clean.reasons)
    assert any("shares" in r for r in dirty.reasons) and dirty.score < clean.score


def test_extra_limit_counts_candidates_only_and_duplicate_dropped():
    nodes = [node(f"n{i}", dev("CUDA0", NEED + 100 + 50 * i, 100 + 50 * i)) for i in range(4)]
    cur = _on(nodes[0])
    out = rank(META, SPEC, nodes, limit=2, extra=[cur])
    assert len(out) == 3
    assert sum(p.replica_id == "run-1" for p in out) == 1
    assert [p for p in out if p.replica_id == ""] and all(
        where(p) != [("n0", "CUDA0")] for p in out if p.replica_id == "")
    # limit 0: the extra is still returned
    only = rank(META, SPEC, nodes, limit=0, extra=[cur])
    assert [p.replica_id for p in only] == ["run-1"]


def test_extra_on_missing_device_is_skipped_and_no_extra_unchanged():
    nodes = [node("a", dev("CUDA0", NEED + 100, 100)), node("b", dev("CUDA0", NEED + 100, 200))]
    gone = _on(node("z", dev("CUDA0", NEED + 100, 100)))
    out = rank(META, SPEC, nodes, extra=[gone])
    assert all(p.replica_id == "" for p in out)
    assert [p.model_dump() for p in out] == [p.model_dump() for p in rank(META, SPEC, nodes)]
    assert [p.model_dump() for p in rank(META, SPEC, nodes, extra=())] == \
        [p.model_dump() for p in rank(META, SPEC, nodes)]
