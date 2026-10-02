import itertools

import pytest

from gpupool.common.models import Device, ModelMeta, ModelSpec, NodeReport
from gpupool.scheduler.estimate import device_need_mb, overhead_mb, total_need_mb
from gpupool.scheduler.placement import NoFit, plan

MB = 2**20


def make_meta(n_layers=32, layer_mb=100, out_mb=100):
    return ModelMeta(
        arch="llama", n_layers=n_layers, n_embd=1024, n_head=8, n_head_kv=8, head_dim=128,
        layer_bytes=[(layer_mb + i % 3) * MB for i in range(n_layers)],  # layers differ in size
        other_bytes=out_mb * 2 * MB, output_bytes=out_mb * MB,
    )


SPEC = ModelSpec(name="m", source="x", ctx_size=512)


def dev(did, usable, kind="cuda"):
    return Device(device_id=did, kind=kind, name=did, total_mb=usable + 500,
                  free_mb=usable + 500, usable_mb=usable)


def node(nid, *devs):
    return NodeReport(node_id=nid, agent_url=f"http://{nid}:7070", host=nid, devices=list(devs),
                      engines=[], llama_version="b1", models=[], ts=0.0)


class Ports:
    def __init__(self):
        self.calls = []
        self._c = itertools.count(9000)

    def __call__(self, nid):
        self.calls.append(nid)
        return next(self._c)


def check(meta, pl, nodes):
    usable = {(n.node_id, d.device_id): (d.usable_mb, d.kind) for n in nodes for d in n.devices}
    assert sum(a.layers for a in pl.assignments) == meta.n_layers
    start = 0
    for i, a in enumerate(pl.assignments):
        assert a.layers >= 1
        cap, kind = usable[(a.node_id, a.device_id)]
        want = device_need_mb(meta, range(start, start + a.layers), SPEC.ctx_size, kind,
                              i == len(pl.assignments) - 1)
        assert a.est_mb == want <= cap
        start += a.layers
    assert pl.tensor_split == [float(a.layers) for a in pl.assignments]
    assert pl.est_total_mb == sum(a.est_mb for a in pl.assignments)
    per = {}
    for a in pl.assignments:
        per[a.node_id] = per.get(a.node_id, 0) + a.layers
    assert per[pl.head_node] == max(per.values())


def test_single_gpu_best_fit():
    meta = make_meta(8)
    need = total_need_mb(meta, 512)
    nodes = [node("a", dev("CUDA0", need + 5000)), node("b", dev("CUDA0", need + 10), dev("CUDA1", need + 800))]
    ports = Ports()
    pl = plan(meta, SPEC, nodes, "r1", ports)
    assert pl.tier == "single_gpu"
    assert (pl.head_node, pl.assignments[0].device_id) == ("b", "CUDA0")
    assert pl.assignments[0].llama_device == "CUDA0" and pl.assignments[0].rpc_endpoint is None
    assert ports.calls == ["b"]
    check(meta, pl, nodes)


def test_two_gpus_one_node():
    meta = make_meta(32)
    need = total_need_mb(meta, 512)
    half = need // 2 + 400
    nodes = [node("a", dev("CUDA0", half), dev("CUDA1", half)), node("b", dev("CUDA0", half))]
    ports = Ports()
    pl = plan(meta, SPEC, nodes, "r1", ports)
    assert pl.tier == "single_node" and pl.head_node == "a"
    assert [a.llama_device for a in pl.assignments] == ["CUDA0", "CUDA1"] or pl.assignments[1].llama_device == "RPC0"
    check(meta, pl, nodes)


def test_three_nodes():
    meta = make_meta(32, layer_mb=120)  # ~ 4 GB of layers
    need = total_need_mb(meta, 512)
    ov = overhead_mb(meta, "cuda")
    base = need - ov
    # each node: its share of the weights + its own overhead + a little slack; no two suffice
    sizes = [base * 20 // 100 + ov + 100, base * 30 // 100 + ov + 100, base * 65 // 100 + ov + 100]
    nodes = [node(f"n{i}", dev("CUDA0", s)) for i, s in enumerate(sizes)]
    ports = Ports()
    pl = plan(meta, SPEC, nodes, "r1", ports)
    assert pl.tier == "multi_node"
    assert len(pl.assignments) == 3
    assert pl.head_node == "n2"
    assert pl.assignments[0].llama_device == "CUDA0"
    assert [a.llama_device for a in pl.assignments[1:]] == ["RPC0", "RPC1"]
    assert all(a.rpc_endpoint for a in pl.assignments[1:])
    assert len(ports.calls) == 3
    check(meta, pl, nodes)


def test_multi_node_drops_unneeded_devices():
    meta = make_meta(16)
    need = total_need_mb(meta, 512)
    nodes = [node("a", dev("CUDA0", need * 6 // 10)), node("b", dev("CUDA0", need * 6 // 10)),
             node("c", dev("CUDA0", 800))]
    pl = plan(meta, SPEC, nodes, "r1", Ports())
    assert {a.node_id for a in pl.assignments} == {"a", "b"}
    check(meta, pl, nodes)


def test_exclude_nodes():
    meta = make_meta(8)
    need = total_need_mb(meta, 512)
    nodes = [node("a", dev("CUDA0", need + 100)), node("b", dev("CUDA0", need + 5000))]
    pl = plan(meta, SPEC, nodes, "r1", Ports(), exclude_nodes={"a"})
    assert pl.head_node == "b"
    with pytest.raises(NoFit):
        plan(meta, SPEC, nodes, "r1", Ports(), exclude_nodes=frozenset({"a", "b"}))


def test_cpu_device_on_head_is_rpc():
    meta = make_meta(16)
    need = total_need_mb(meta, 512)
    nodes = [node("a", dev("CUDA0", need * 6 // 10), dev("CPU", need * 6 // 10, kind="cpu"))]
    ports = Ports()
    pl = plan(meta, SPEC, nodes, "r1", ports)
    cuda, cpu = pl.assignments
    assert (cuda.llama_device, cuda.rpc_endpoint) == ("CUDA0", None)
    assert cpu.device_id == "CPU" and cpu.llama_device == "RPC0" and cpu.rpc_endpoint == "a:9001"
    assert ports.calls == ["a", "a"]
    check(meta, pl, nodes)


def test_cpu_only_single_device():
    meta = make_meta(4)
    nodes = [node("a", dev("CPU", total_need_mb(meta, 512) + 100, kind="cpu"))]
    pl = plan(meta, SPEC, nodes, "r1", Ports())
    assert pl.assignments[0].llama_device == "RPC0"
    check(meta, pl, nodes)


def test_nofit_message():
    meta = make_meta(32)
    nodes = [node("a", dev("CUDA0", 1000)), node("b", dev("CUDA0", 1000), dev("CUDA1", 0))]
    with pytest.raises(NoFit, match=r"needs about \d+ MB.*\d+ MB usable"):
        plan(meta, SPEC, nodes, "r1", Ports())


def test_port_alloc_not_called_on_nofit():
    ports = Ports()
    with pytest.raises(NoFit):
        plan(make_meta(32), SPEC, [node("a", dev("CUDA0", 500))], "r1", ports)
    assert ports.calls == []


def test_pool_exactly_total_still_fits_or_nofit_cleanly():
    meta = make_meta(24)
    need = total_need_mb(meta, 512)
    # total usable barely above need but spread over 4 devices: overhead makes it infeasible
    nodes = [node(f"n{i}", dev("CUDA0", need // 4 + 10)) for i in range(4)]
    with pytest.raises(NoFit):
        plan(meta, SPEC, nodes, "r1", Ports())
    assert overhead_mb(meta, "cuda") > 0


def test_gpus_on_other_nodes_preferred_over_local_cpu():
    # Node a: small GPU + big CPU RAM; node b: GPU big enough to finish the job.
    # A GPU-only multi-node split exists, so no layer may land in host RAM.
    meta = make_meta(16)
    need = total_need_mb(meta, 512)
    nodes = [node("a", dev("CUDA0", need * 6 // 10), dev("CPU", need * 4, kind="cpu")),
             node("b", dev("CUDA0", need * 6 // 10))]
    pl = plan(meta, SPEC, nodes, "r1", Ports())
    assert all(a.device_id != "CPU" for a in pl.assignments)
    assert pl.tier == "multi_node"
    check(meta, pl, nodes)


def test_cpu_joins_only_when_gpus_cannot_hold_model():
    meta = make_meta(16)
    need = total_need_mb(meta, 512)
    nodes = [node("a", dev("CUDA0", need * 6 // 10), dev("CPU", need, kind="cpu"))]
    pl = plan(meta, SPEC, nodes, "r1", Ports())
    assert {a.device_id for a in pl.assignments} == {"CUDA0", "CPU"}
    check(meta, pl, nodes)


def test_plan_carries_device_uuid():
    meta = make_meta(8)
    need = total_need_mb(meta, 512)
    d0 = dev("CUDA0", need + 50)
    d0.uuid = "GPU-aaaa"
    d1 = dev("CUDA0", need + 50)  # an old agent: no uuid
    n = [node("a", d0)]
    assert plan(meta, SPEC, n, "r1", Ports()).assignments[0].device_uuid == "GPU-aaaa"
    assert plan(meta, SPEC, [node("b", d1)], "r2", Ports()).assignments[0].device_uuid is None


# ---- KV-cache quantisation and speculative draft reservation

from gpupool.scheduler.estimate import draft_need_mb, kv_bytes_per_layer  # noqa: E402
from gpupool.scheduler.placement import rank  # noqa: E402

DRAFT = make_meta(4, layer_mb=10, out_mb=10)
DSPEC = SPEC.model_copy(update={"speculative": "draft"})


def test_kv_cache_type_factors():
    m = make_meta(32)
    f16 = kv_bytes_per_layer(m, 4096)
    assert kv_bytes_per_layer(m, 4096, "f16") == f16 == 2 * 4096 * 8 * 128 * 2
    assert kv_bytes_per_layer(m, 4096, "q8_0") == f16 * 34 // 64
    assert kv_bytes_per_layer(m, 4096, "q4_0") == f16 * 18 // 64
    # Qwen2.5-3B-like: 36 layers, 2 kv heads x 128, ctx 8192 -> theory -135 / -207 MB
    q = ModelMeta(arch="qwen2", n_layers=36, n_embd=2048, n_head=16, n_head_kv=2, head_dim=128,
                  layer_bytes=[40 * MB] * 36, other_bytes=0, output_bytes=0)
    base = total_need_mb(q, 8192)
    assert base - total_need_mb(q, 8192, "q8_0") in (134, 135, 136)
    assert base - total_need_mb(q, 8192, "q4_0") in (206, 207, 208)


def test_cache_type_default_unchanged():
    m = make_meta(8)
    assert total_need_mb(m, 512) == total_need_mb(m, 512, "f16")
    assert device_need_mb(m, range(8), 512, "cuda", True) == total_need_mb(m, 512)
    nodes = [node("a", dev("CUDA0", 5000))]
    pl = plan(m, SPEC, nodes, "r", Ports())
    assert pl.draft_est_mb is None and pl.est_total_mb == total_need_mb(m, 512)


def test_plan_uses_spec_cache_type():
    m = make_meta(8)
    spec = SPEC.model_copy(update={"ctx_size": 8192, "kv_cache_type": "q4_0"})
    pl = plan(m, spec, [node("a", dev("CUDA0", 5000))], "r", Ports())
    assert pl.est_total_mb == total_need_mb(m, 8192, "q4_0") < total_need_mb(m, 8192)


def test_cache_type_lets_model_fit():
    m = make_meta(8)
    ctx = 16384
    cap = total_need_mb(m, ctx, "q4_0") + 5
    nodes = [node("a", dev("CUDA0", cap))]
    with pytest.raises(NoFit):
        plan(m, SPEC.model_copy(update={"ctx_size": ctx}), nodes, "r", Ports())
    pl = plan(m, SPEC.model_copy(update={"ctx_size": ctx, "kv_cache_type": "q4_0"}), nodes, "r", Ports())
    assert pl.tier == "single_gpu"


def test_draft_need_is_whole_model_on_cuda():
    assert draft_need_mb(DRAFT, 512) == device_need_mb(DRAFT, range(4), 512, "cuda", True)
    assert draft_need_mb(DRAFT, 512, "q8_0") < draft_need_mb(DRAFT, 512)


def test_draft_reserved_on_first_device():
    m = make_meta(8)
    dn = draft_need_mb(DRAFT, 512)
    nodes = [node("a", dev("CUDA0", total_need_mb(m, 512) + dn + 10))]
    base = plan(m, SPEC, nodes, "r", Ports())
    pl = plan(m, DSPEC, nodes, "r", Ports(), draft_meta=DRAFT)
    assert pl.draft_est_mb == dn
    assert pl.assignments[0].est_mb == base.assignments[0].est_mb + dn
    assert pl.est_total_mb == base.est_total_mb + dn
    assert pl.assignments[0].est_mb <= nodes[0].devices[0].usable_mb
    # without draft_meta, or with speculative != draft, nothing changes
    assert plan(m, DSPEC, nodes, "r", Ports()).draft_est_mb is None
    assert plan(m, SPEC, nodes, "r", Ports(), draft_meta=DRAFT).draft_est_mb is None
    assert rank(m, DSPEC, nodes, draft_meta=DRAFT)[0].draft_est_mb == dn


def test_draft_forces_different_head():
    m = make_meta(8)
    need, dn = total_need_mb(m, 512), draft_need_mb(DRAFT, 512)
    # best fit is "b" (tight) but only 10 MB of slack: the draft goes to "a"
    nodes = [node("a", dev("CUDA0", need + 5000)), node("b", dev("CUDA0", need + 10))]
    assert plan(m, SPEC, nodes, "r", Ports()).head_node == "b"
    pl = plan(m, DSPEC, nodes, "r", Ports(), draft_meta=DRAFT)
    assert pl.head_node == "a" and pl.draft_est_mb == dn
    assert all(p.head_node == "a" for p in rank(m, DSPEC, nodes, draft_meta=DRAFT))


def test_draft_spills_layers_to_another_node_when_the_head_cannot_hold_both():
    m = make_meta(8)
    need, dn = total_need_mb(m, 512), draft_need_mb(DRAFT, 512)
    # a alone fits the model but not model + draft: with a's room reduced by the draft (only a,
    # the head), the model spreads over a and b.
    nodes = [node("a", dev("CUDA0", need + 50)), node("b", dev("CUDA0", 700))]
    assert plan(m, SPEC, nodes, "r", Ports()).tier == "single_gpu"
    pl = plan(m, DSPEC, nodes, "r", Ports(), draft_meta=DRAFT)
    assert pl.tier == "multi_node" and pl.head_node == "a"
    assert pl.assignments[0].est_mb <= need + 50  # draft included
    assert pl.draft_est_mb == dn


def test_draft_head_without_local_cuda_dropped():
    m = make_meta(8)
    nodes = [node("a", dev("CPU", 5000, kind="cpu")), node("b", dev("CUDA0", total_need_mb(m, 512) + 5000))]
    pl = plan(m, DSPEC, nodes, "r", Ports(), draft_meta=DRAFT)
    assert pl.head_node == "b" and pl.assignments[0].llama_device == "CUDA0"


def test_draft_nofit_message():
    m = make_meta(8)
    dn = draft_need_mb(DRAFT, 512)
    nodes = [node("a", dev("CUDA0", total_need_mb(m, 512) + dn - 20))]
    assert plan(m, SPEC, nodes, "r", Ports()).tier == "single_gpu"
    with pytest.raises(NoFit, match="draft"):
        plan(m, DSPEC, nodes, "r", Ports(), draft_meta=DRAFT)
    with pytest.raises(NoFit, match="draft"):
        plan(m, DSPEC, [node("a", dev("CPU", 99999, kind="cpu"))], "r", Ports(), draft_meta=DRAFT)


def _qwen_like(n, layer_mb, out_mb, kv_heads=2):
    return ModelMeta(arch="qwen2", n_layers=n, n_embd=2048, n_head=16, n_head_kv=kv_heads,
                     head_dim=128, layer_bytes=[layer_mb * MB] * n,
                     other_bytes=out_mb * 2 * MB, output_bytes=out_mb * MB)


def test_draft_three_server_cluster_reduces_only_the_head():
    # Real defect: a=1300 (GTX 1650), b=c=1100; 3B-like model ~2.1 GB + 0.5B-like draft ~0.56 GB
    # at ctx 2048. Reducing every GPU by the draft (3 x 563 MB) made this NoFit, although
    # charging the draft to a alone leaves 737 + 1100 + 1100 for the model.
    m, dm = _qwen_like(36, 46, 150), _qwen_like(24, 10, 61)
    spec = SPEC.model_copy(update={"ctx_size": 2048, "speculative": "draft"})
    dn = draft_need_mb(dm, 2048)
    assert 550 <= dn <= 575 and 2080 <= total_need_mb(m, 2048) <= 2160
    nodes = [node("a", dev("CUDA0", 1300)), node("b", dev("CUDA0", 1100)),
             node("c", dev("CUDA0", 1100))]
    assert total_need_mb(m, 2048) + dn <= 3500  # the pool holds it only if the draft is charged once
    pl = plan(m, spec, nodes, "r", Ports(), draft_meta=dm)
    assert pl.draft_est_mb == dn and pl.head_node == "a"
    a0 = pl.assignments[0]
    assert (a0.node_id, a0.device_id, a0.llama_device) == ("a", "CUDA0", "CUDA0")
    usable = {n.node_id: n.devices[0].usable_mb for n in nodes}
    assert sum(a.layers for a in pl.assignments) == 36
    for i, a in enumerate(pl.assignments):
        assert a.est_mb <= usable[a.node_id]
    # the head's est_mb is its layer share plus the draft
    own = device_need_mb(m, range(a0.layers), 2048, "cuda", len(pl.assignments) == 1)
    assert a0.est_mb == own + dn and a0.est_mb <= 1300
    assert pl.est_total_mb == sum(a.est_mb for a in pl.assignments)
    for p in rank(m, spec, nodes, draft_meta=dm):
        assert p.assignments[0].llama_device == p.assignments[0].device_id
        assert p.assignments[0].est_mb <= usable[p.assignments[0].node_id]


def test_draft_goes_on_the_head_device_even_if_a_sibling_has_more_room():
    # Two GPUs on a: after D=small gives up the draft, the roomier sibling would sort first;
    # the draft must still sit on assignments[0], which has to be the intended head device.
    m = make_meta(8)
    need, dn = total_need_mb(m, 512), draft_need_mb(DRAFT, 512)
    nodes = [node("a", dev("CUDA0", need + dn + 10), dev("CUDA1", need + dn + 20))]
    for p in rank(m, DSPEC, nodes, draft_meta=DRAFT):
        a0 = p.assignments[0]
        assert a0.llama_device == a0.device_id
        assert a0.est_mb <= {"CUDA0": need + dn + 10, "CUDA1": need + dn + 20}[a0.device_id]
    assert plan(m, DSPEC, nodes, "r", Ports(), draft_meta=DRAFT).draft_est_mb == dn


def test_draft_many_gpus_stays_bounded():
    import time
    m = make_meta(32, layer_mb=100)
    dn = draft_need_mb(DRAFT, 512)
    nodes = [node(f"n{i}", dev("CUDA0", 700 + i), dev("CUDA1", 650 + i)) for i in range(12)]
    t = time.perf_counter()
    pl = plan(m, DSPEC, nodes, "r", Ports(), draft_meta=DRAFT)
    assert time.perf_counter() - t < 20
    assert pl.draft_est_mb == dn
    a0 = pl.assignments[0]
    assert a0.node_id == pl.head_node and a0.llama_device == a0.device_id
