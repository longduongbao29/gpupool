"""Placement for vLLM / SGLang: tensor parallelism over the GPUs of one server.

These engines shard every layer over k GPUs (k divides the attention heads; 1, 2, 4 or 8) and talk
over NCCL, so a replica never spans servers and never uses a CPU device. Each GPU holds 1/k of the
weights and of the KV cache (the whole latent cache with MLA, which is not sharded), plus the
engine's own activations and runtime.

The engines size their memory as a fraction of each GPU's TOTAL memory (vLLM
--gpu-memory-utilization covers everything, SGLang --mem-fraction-static covers weights + KV), so
on GPUs of different sizes the same fraction takes more on the bigger one; the estimate per device
follows that and must fit what the device has free.

Scores are the llama.cpp ones (scheduler/placement.py) with a tensor-parallel speed model: a token
reads 1/k of the weights on each GPU at once, so the slowest GPU sets the pace.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from gpupool.common.models import DeviceAssignment, ModelMeta, ModelSpec, NodeReport, Occupant, Placement
from gpupool.scheduler.estimate import compute_mb, kv_total_bytes
from gpupool.scheduler.placement import (
    NoFit, _Cand, _cand_key, _Dev, _extra_cand, _MEM_FACTOR, _score_all, _Scored, _scale,
)
from gpupool.scheduler.scoring import current_tps, decode_bytes, device_bw

_MB = 1024 * 1024
TP_SIZES = (1, 2, 4, 8)
# Per GPU beyond weights + KV: CUDA context, NCCL buffers, CUDA graphs, allocator slack.
ENGINE_RUNTIME_MB = 1024
# Activations of one forward pass are sized for a chunked prefill of this many tokens.
PREFILL_TOKENS = 2048


def has_engine(node: NodeReport, engine: str) -> bool:
    """The agent runs this engine (agent/engines.py reports "engine:<name>")."""
    return f"engine:{engine}" in node.features


def tp_sizes(meta: ModelMeta, spec: ModelSpec) -> list[int]:
    """Tensor-parallel sizes the engines accept for this model (k divides the attention heads)."""
    sizes = [spec.tensor_parallel] if spec.tensor_parallel else list(TP_SIZES)
    return [k for k in sizes if meta.n_head % k == 0]


def _weights(meta: ModelMeta) -> int:
    return meta.file_bytes or sum(meta.layer_bytes) + meta.other_bytes


def _mla(meta: ModelMeta) -> bool:
    return bool(meta.kv_k and meta.kv_v and any(meta.kv_k) and not any(meta.kv_v))


def need_parts_mb(meta: ModelMeta, spec: ModelSpec, k: int) -> tuple[int, int]:
    """(static MB: weights + KV, dynamic MB: activations + runtime) per GPU at tensor parallel k.
    The KV pool is shared by the requests (paged), so ctx_size is the total cached tokens; heads
    are sharded down to one per GPU (engines replicate KV heads when k > n_head_kv)."""
    kv = kv_total_bytes(meta, spec.ctx_size, "f16", spec.parallel, kv_unified=True)
    share = 1.0 if _mla(meta) else max(1.0 / k, 1.0 / max(1, meta.n_head_kv))
    static = math.ceil((_weights(meta) / k + kv * share) / _MB)
    dynamic = compute_mb(meta, PREFILL_TOKENS) + ENGINE_RUNTIME_MB
    return _scale(static), _scale(dynamic)


def need_mb(meta: ModelMeta, spec: ModelSpec, k: int) -> int:
    static, dynamic = need_parts_mb(meta, spec, k)
    return static + dynamic


def sizing(meta: ModelMeta, spec: ModelSpec, devs: Sequence[_Dev]) -> tuple[float, list[int]]:
    """(engine memory fraction, MB each device ends up holding) for tensor parallel len(devs)."""
    static, dynamic = need_parts_mb(meta, spec, len(devs))
    smallest = min(d.dev.total_mb for d in devs) or 1
    if spec.engine == "sglang":  # the fraction covers weights + KV; activations come on top
        frac = static / smallest
        return frac, [math.ceil(frac * d.dev.total_mb) + dynamic for d in devs]
    frac = (static + dynamic) / smallest  # vLLM: the fraction covers everything
    return frac, [math.ceil(frac * d.dev.total_mb) for d in devs]


def _fits(meta, spec, devs: Sequence[_Dev]) -> bool:
    frac, est = sizing(meta, spec, devs)
    return frac <= 0.98 and all(e <= d.dev.usable_mb for d, e in zip(devs, est))


def _device_sets(gpus: list[_Dev], k: int) -> list[list[_Dev]]:
    """The k roomiest GPUs of the server, and the k roomiest of each GPU model it has k of
    (tensor parallelism runs at the pace of the slowest GPU, so a matched set is worth trying)."""
    sets = []
    if len(gpus) >= k:
        sets.append(gpus[:k])
    by_name: dict[str, list[_Dev]] = {}
    for d in gpus:
        by_name.setdefault(d.dev.name, []).append(d)
    for same in by_name.values():
        if len(same) >= k and same[:k] not in sets:
            sets.append(same[:k])
    return sets


def candidates(meta: ModelMeta, spec: ModelSpec, nodes: Sequence[NodeReport],
               exclude_nodes=frozenset()) -> tuple[list[_Cand], list[_Dev]]:
    """Feasible candidates and the devices considered. Without spec.tensor_parallel each server
    offers its smallest k that fits: a bigger k is faster but holds GPUs other models need."""
    out: list[_Cand] = []
    pool: list[_Dev] = []
    sizes = tp_sizes(meta, spec)
    for n in nodes:
        if n.node_id in exclude_nodes or not has_engine(n, spec.engine):
            continue
        gpus = sorted((_Dev(n, d) for d in n.devices if d.kind == "cuda" and d.usable_mb > 0),
                      key=lambda d: (-d.dev.usable_mb, d.dev.device_id))
        pool += gpus
        for k in sizes:
            found = [s for s in _device_sets(gpus, k) if _fits(meta, spec, s)]
            for s in found:
                out.append(_Cand("single_gpu" if k == 1 else "single_node", s,
                                 [meta.n_layers] * k, n.node_id))
            if found and not spec.tensor_parallel:
                break
    return out, pool


def _speed(meta: ModelMeta) -> Callable[[_Cand, float], tuple[float, tuple[float, int, float]]]:
    read = sum(decode_bytes(meta)) + meta.output_bytes

    def speed(c: _Cand, cuda_bw: float) -> tuple[float, tuple[float, int, float]]:
        k = len(c.order)
        bw_s = max(read / k / (device_bw(d.dev, cuda_bw) * 1e9) for d in c.order)
        return current_tps(bw_s, 0, 0.0), (bw_s, 0, 0.0)

    return speed


def _ranked(meta, spec, nodes, occupants, exclude_nodes=frozenset(),
            extra_cands: Sequence[_Cand] = ()) -> list[_Scored]:
    cands, used = candidates(meta, spec, nodes, exclude_nodes)
    if extra_cands:
        keys = {_cand_key(c) for c in extra_cands}
        cands = [c for c in cands if _cand_key(c) not in keys] + list(extra_cands)
        used = used + [d for c in extra_cands for d in c.order]
    if not cands:
        return []
    return _score_all(meta, spec, cands, used, nodes, occupants, speed=_speed(meta),
                      split_word="tensor parallel over")


def _build(meta, spec, s: _Scored, replica_id: str, port_alloc) -> Placement:
    c = s.cand
    frac, est = sizing(meta, spec, c.order)
    assignments = [DeviceAssignment(node_id=d.node.node_id, device_id=d.dev.device_id,
                                    llama_device=d.dev.device_id, layers=meta.n_layers, est_mb=e,
                                    device_uuid=d.dev.uuid) for d, e in zip(c.order, est)]
    return Placement(
        model=spec.name, replica_id=replica_id, tier=c.tier, head_node=c.head_id,
        head_port=port_alloc(c.head_id), assignments=assignments,
        tensor_split=[1.0] * len(assignments), est_total_mb=sum(est),
        score=round(s.score, 1), est_decode_tps=round(s.tps, 1),
        est_bw_s=s.parts[0], est_hops=0, est_logits_s=0.0, reasons=s.reasons,
        mem_factor=_MEM_FACTOR.get(), engine=spec.engine, gpu_mem_fraction=round(min(frac, 0.98), 4))


def _no_fit(meta, spec, nodes, exclude_nodes) -> NoFit:
    sizes = tp_sizes(meta, spec)
    if not sizes:
        return NoFit(f"model {spec.name!r}: tensor parallel {spec.tensor_parallel} does not divide "
                     f"its {meta.n_head} attention heads")
    with_engine = [n for n in nodes if n.node_id not in exclude_nodes and has_engine(n, spec.engine)]
    if not with_engine:
        return NoFit(f"model {spec.name!r}: no server runs {spec.engine} (install it on a GPU server; "
                     f"the agent offers it when GPUPOOL_{spec.engine.upper()}_CMD resolves)")
    gpus = [f"{n.node_id}/{d.device_id} {d.usable_mb}" for n in with_engine
            for d in n.devices if d.kind == "cuda" and d.usable_mb > 0]
    need = ", ".join(f"{need_mb(meta, spec, k)} MB x {k}" for k in sizes)
    return NoFit(f"model {spec.name!r} ({spec.engine}) needs per GPU, by tensor parallel size: "
                 f"{need} at ctx {spec.ctx_size}, all on one server; usable MB: "
                 f"{', '.join(gpus) or 'none'}")


def plan(meta: ModelMeta, spec: ModelSpec, nodes: list[NodeReport], replica_id: str,
         port_alloc: Callable[[str], int], exclude_nodes=frozenset(),
         occupants: Sequence[Occupant] = ()) -> Placement:
    ranked = _ranked(meta, spec, nodes, occupants, exclude_nodes)
    if not ranked:
        raise _no_fit(meta, spec, nodes, exclude_nodes)
    return _build(meta, spec, ranked[0], replica_id, port_alloc)


def rank(meta: ModelMeta, spec: ModelSpec, nodes: list[NodeReport],
         occupants: Sequence[Occupant] = (), limit: int = 5,
         extra: Sequence[Placement] = ()) -> list[Placement]:
    """Same contract as scheduler.placement.rank."""
    dummy = lambda _nid: 0  # noqa: E731
    pairs = [(pl, _extra_cand(pl, nodes)) for pl in extra]
    pairs = [(pl, c) for pl, c in pairs if c is not None]
    owner = {id(c): pl for pl, c in pairs}
    out, n_cand = [], 0
    for s in _ranked(meta, spec, nodes, occupants, extra_cands=[c for _, c in pairs]):
        pl = owner.get(id(s.cand))
        if pl is not None:
            out.append(pl.model_copy(update={
                "score": round(s.score, 1), "est_decode_tps": round(s.tps, 1),
                "est_bw_s": s.parts[0], "est_hops": 0, "est_logits_s": 0.0,
                "reasons": s.reasons}, deep=True))
        elif n_cand < limit:
            n_cand += 1
            out.append(_build(meta, spec, s, "", dummy))
    return out
