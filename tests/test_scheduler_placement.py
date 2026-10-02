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
