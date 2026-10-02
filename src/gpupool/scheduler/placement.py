"""Placement: choose devices and a layer split for one replica (design section 6).

Pure function of (meta, spec, nodes, occupants): enumerate feasible candidates (single_gpu,
single_node, multi_node only when needed), score them, pick the best.
Ports are allocated only for the final plan.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from gpupool.common.models import (
    Device,
    DeviceAssignment,
    ModelMeta,
    ModelSpec,
    NodeReport,
    Occupant,
    Placement,
)
from gpupool.scheduler.estimate import device_need_mb, overhead_mb, total_need_mb
from gpupool.scheduler.scoring import default_cuda_bw, est_decode_tps


class NoFit(Exception):
    """The pool cannot hold the model."""


class _Dev(NamedTuple):
    node: NodeReport
    dev: Device


def _need(meta, ctx, d: _Dev, start: int, count: int, is_last: bool) -> int:
    return device_need_mb(meta, range(start, start + count), ctx, d.dev.kind, is_last)


def _check(meta, ctx, order: list[_Dev], counts: list[int]) -> list[int]:
    """Per-device slack (usable - need) for the given counts."""
    out, start = [], 0
    for i, (d, c) in enumerate(zip(order, counts)):
        out.append(d.dev.usable_mb - _need(meta, ctx, d, start, c, i == len(order) - 1))
        start += c
    return out


def _split(meta: ModelMeta, ctx: int, order: list[_Dev]) -> list[int] | None:
    """Layer counts per device in order, or None if no feasible split is found.

    Proportional to capacity (largest remainder), then a repair pass moving one
    layer at a time from the most overfull device to the one with most slack,
    verified with exact per-layer bytes.
    """
    n, L = len(order), meta.n_layers
    if n == 0 or n > L:
        return None
    w = [max(1, d.dev.usable_mb - overhead_mb(meta, d.dev.kind)) for d in order]
    # every device gets at least one layer; distribute the rest proportionally
    spare = L - n
    raw = [spare * x / sum(w) for x in w]
    counts = [1 + int(r) for r in raw]
    rest = L - sum(counts)
    for i in sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)[:rest]:
        counts[i] += 1
    for _ in range(4 * L + 4):
        slack = _check(meta, ctx, order, counts)
        worst = min(range(n), key=lambda i: slack[i])
        if slack[worst] >= 0:
            return counts
        if counts[worst] <= 1:
            return None
        best = max((i for i in range(n) if i != worst), key=lambda i: slack[i], default=None)
        if best is None or slack[best] <= 0:
            return None
        counts[worst] -= 1
        counts[best] += 1
    return None


def _order(devs: list[_Dev], head_id: str) -> list[_Dev]:
    """Head cuda devices first, then head cpu, then other nodes by usable desc."""
    by_usable = lambda d: -d.dev.usable_mb  # noqa: E731
    head = [d for d in devs if d.node.node_id == head_id]
    head_cuda = sorted((d for d in head if d.dev.kind == "cuda"), key=by_usable)
    head_cpu = sorted((d for d in head if d.dev.kind != "cuda"), key=by_usable)
    totals: dict[str, int] = {}
    for d in devs:
        totals[d.node.node_id] = totals.get(d.node.node_id, 0) + d.dev.usable_mb
    others = sorted(
        (d for d in devs if d.node.node_id != head_id),
        key=lambda d: (-totals[d.node.node_id], d.node.node_id, -d.dev.usable_mb),
    )
    return head_cuda + head_cpu + others


def _solve(meta, ctx, devs: list[_Dev]) -> tuple[list[_Dev], list[int], str] | None:
    """Order + split for a device set; head = node with most layers (fixed point)."""
    totals: dict[str, int] = {}
    for d in devs:
        totals[d.node.node_id] = totals.get(d.node.node_id, 0) + d.dev.usable_mb
    head = max(totals, key=lambda k: (totals[k], k))
    seen = set()
    first = None
    while head not in seen:
        seen.add(head)
        order = _order(devs, head)
        counts = _split(meta, ctx, order)
        if counts is None:
            break
        per: dict[str, int] = {}
        for d, c in zip(order, counts):
            per[d.node.node_id] = per.get(d.node.node_id, 0) + c
        first = first or (order, counts, head)
        top = max(per.values())
        if per[head] == top:
            return order, counts, head
        head = max((k for k in per if per[k] == top), key=lambda k: (totals[k], k))
    # no self-consistent head; keep a feasible plan, head = most layers actually held
    if first is None:
        return None
    order, counts, _ = first
    per = {}
    for d, c in zip(order, counts):
        per[d.node.node_id] = per.get(d.node.node_id, 0) + c
    return order, counts, max(per, key=lambda k: (per[k], k))


# Score weights (design section 4.3); higher score wins.
W_PERF = 100.0  # x tps / best tps among the candidates
W_SHARE = 10.0  # per engine already on a chosen GPU
W_BUSY = 30.0  # x busy (0..1) of that engine
W_SAME_GPU = 40.0  # per replica of the same model on a chosen GPU (spread gpu/node)
W_SAME_NODE = 20.0  # per replica of the same model on a chosen node (spread node)
W_WASTE = 15.0  # x mean(usable / biggest usable): keeps big GPUs free (best-fit among equals)
W_DEVICE = 5.0  # per extra device
W_RPC = 10.0  # per network hop
_TIER_ORDER = {"single_gpu": 0, "single_node": 1, "multi_node": 2}


class _Cand(NamedTuple):
    tier: str
    order: list[_Dev]
    counts: list[int]
    head_id: str


class _Scored(NamedTuple):
    score: float
    tps: float
    reasons: list[str]
    cand: _Cand


def _fits_single(meta, ctx, pool) -> list[_Dev]:
    return [d for d in pool
            if device_need_mb(meta, range(meta.n_layers), ctx, d.dev.kind, True) <= d.dev.usable_mb]


def _candidates(meta, ctx, pool, allow_single=True) -> list[_Cand]:
    """Every feasible placement; multi_node only when no single GPU / single node fits."""
    out = []
    if allow_single:
        out += [_Cand("single_gpu", [d], [meta.n_layers], d.node.node_id)
                for d in _fits_single(meta, ctx, pool)]
    out += [_Cand("single_node", *s) for s in _tier_single_node(meta, ctx, pool)]
    if not out:
        s = _tier_multi_node(meta, ctx, pool)
        if s:
            out.append(_Cand("multi_node", *s))
    return out


def _all_candidates(meta, ctx, pool: list[_Dev]) -> tuple[list[_Cand], list[_Dev]]:
    """GPU-only first. Only when the GPUs of the pool cannot hold the model do CPU devices
    join: within a tier they count as capacity like any GPU, so mixing them in from the
    start would put layers in host RAM (10x+ slower) even when spare VRAM exists elsewhere.
    Returns the candidates and the pool they were built from."""
    gpu_pool = [d for d in pool if d.dev.kind == "cuda"]
    if gpu_pool and len(gpu_pool) < len(pool):
        cands = _candidates(meta, ctx, gpu_pool)
        if cands:
            return cands, gpu_pool
        # CPU-inclusive pass: a lone CPU device must not win "single_gpu" (GPU+CPU split
        # beats CPU-only), so single-device candidates are skipped.
        return _candidates(meta, ctx, pool, allow_single=False), pool
    return _candidates(meta, ctx, pool), pool


def _is_local(d: _Dev, head_id: str) -> bool:
    return d.node.node_id == head_id and d.dev.kind == "cuda"


def _n_rpc(c: _Cand) -> int:
    return sum(1 for d in c.order if not _is_local(d, c.head_id))


def _score_all(meta, spec, cands: list[_Cand], pool: list[_Dev], nodes, occupants) -> list[_Scored]:
    cuda_bw = default_cuda_bw(nodes)
    tps_of = [
        est_decode_tps(meta, [(d.dev, k) for d, k in zip(c.order, c.counts)], _n_rpc(c), cuda_bw)
        for c in cands
    ]
    best_tps = max(tps_of, default=0.0) or 1.0
    biggest = max((d.dev.usable_mb for d in pool), default=1) or 1
    by_dev: dict[tuple[str, str], list[Occupant]] = {}
    by_node: dict[str, list[Occupant]] = {}
    for o in occupants:
        by_dev.setdefault((o.node_id, o.device_id), []).append(o)
        by_node.setdefault(o.node_id, []).append(o)
    out = []
    for c, tps in zip(cands, tps_of):
        n_dev, n_rpc = len(c.order), _n_rpc(c)
        perf = 100 * tps / best_tps
        score = W_PERF * tps / best_tps
        share_reasons, same_reasons = [], []
        for d in c.order:
            key = (d.node.node_id, d.dev.device_id)
            for o in by_dev.get(key, []):
                score -= W_SHARE + W_BUSY * o.busy
                share_reasons.append(
                    f"shares {key[0]}/{key[1]} with {o.model} (busy {round(o.busy * 100)}%)")
                if o.model == spec.name and spec.spread != "none":
                    score -= W_SAME_GPU
                    same_reasons.append(
                        f"another replica of this model is already on {key[0]}/{key[1]}")
        if spec.spread == "node":
            for nid in dict.fromkeys(d.node.node_id for d in c.order):
                for o in by_node.get(nid, []):
                    if o.model == spec.name:
                        score -= W_SAME_NODE
                        same_reasons.append(f"another replica of this model is on node {nid}")
        score -= W_WASTE * sum(d.dev.usable_mb / biggest for d in c.order) / n_dev
        score -= W_DEVICE * (n_dev - 1) + W_RPC * n_rpc
        if len(cands) == 1:
            speed = f"only feasible placement, ~{tps:.0f} tok/s"
        elif perf >= 99.95:
            speed = f"fastest option, ~{tps:.0f} tok/s"
        else:
            speed = f"~{tps:.0f} tok/s ({perf:.0f}% of the fastest)"
        reasons = [speed] + list(dict.fromkeys(same_reasons)) + share_reasons
        if n_dev > 1:
            reasons.append(f"split over {n_dev} GPUs")
        if n_rpc:
            reasons.append(f"{n_rpc} network hop{'s' if n_rpc > 1 else ''} (RPC)")
        out.append(_Scored(score, tps, reasons[:4], c))
    # deterministic: score, then smaller tier, then the first device's name
    out.sort(key=lambda s: (-round(s.score, 6), _TIER_ORDER[s.cand.tier],
                            s.cand.order[0].node.node_id, s.cand.order[0].dev.device_id))
    return out


def _ranked(meta, spec, nodes, occupants, exclude_nodes=frozenset()) -> list[_Scored]:
    pool = [
        _Dev(n, d)
        for n in nodes
        if n.node_id not in exclude_nodes
        for d in n.devices
        if d.usable_mb > 0
    ]
    cands, used = _all_candidates(meta, spec.ctx_size, pool)
    if not cands:
        return []
    return _score_all(meta, spec, cands, used, nodes, occupants)


def _finish(meta, spec, s: _Scored, replica_id, port_alloc) -> Placement:
    c = s.cand
    pl = _build(meta, spec, replica_id, port_alloc, c.tier, c.order, c.counts, c.head_id)
    pl.score = round(s.score, 1)
    pl.est_decode_tps = round(s.tps, 1)
    pl.reasons = s.reasons
    return pl


def plan(
    meta: ModelMeta,
    spec: ModelSpec,
    nodes: list[NodeReport],
    replica_id: str,
    port_alloc: Callable[[str], int],
    exclude_nodes: frozenset[str] | set[str] = frozenset(),
    occupants: Sequence[Occupant] = (),
) -> Placement:
    """The best-scored feasible placement; ports are allocated only for the winner."""
    ranked = _ranked(meta, spec, nodes, occupants, exclude_nodes)
    if not ranked:
        ctx = spec.ctx_size
        pool = [d for n in nodes if n.node_id not in exclude_nodes for d in n.devices
                if d.usable_mb > 0]
        raise NoFit(
            f"model {spec.name!r} needs about {total_need_mb(meta, ctx)} MB at ctx {ctx}, "
            f"pool has {sum(d.usable_mb for d in pool)} MB usable "
            f"across {len(pool)} devices (no feasible layer split)"
        )
    return _finish(meta, spec, ranked[0], replica_id, port_alloc)


def rank(
    meta: ModelMeta,
    spec: ModelSpec,
    nodes: list[NodeReport],
    occupants: Sequence[Occupant] = (),
    limit: int = 5,
) -> list[Placement]:
    """Feasible candidates, best first. Ports are dummies: these are for display/simulation."""
    dummy = lambda _nid: 0  # noqa: E731
    return [_finish(meta, spec, s, "", dummy) for s in _ranked(meta, spec, nodes, occupants)[:limit]]


def _tier_single_node(meta, ctx, pool):
    """One solved split per node: the first k devices (by usable) that admit a feasible one."""
    out = []
    for node_id in sorted({d.node.node_id for d in pool}):
        devs = sorted((d for d in pool if d.node.node_id == node_id), key=lambda d: -d.dev.usable_mb)
        for k in range(2, len(devs) + 1):
            solved = _solve(meta, ctx, devs[:k])
            if solved is not None:
                out.append(solved)
                break
    return out


def _tier_multi_node(meta, ctx, pool):
    totals: dict[str, int] = {}
    for d in pool:
        totals[d.node.node_id] = totals.get(d.node.node_id, 0) + d.dev.usable_mb
    ranked = sorted(totals, key=lambda k: (-totals[k], k))
    chosen: list[_Dev] = []
    solved = None
    for node_id in ranked:
        chosen += [d for d in pool if d.node.node_id == node_id]
        solved = _solve(meta, ctx, chosen)
        if solved:
            break
    if not solved:
        return None
    # drop devices that are not needed, smallest first, while still feasible
    for d in sorted(chosen, key=lambda d: d.dev.usable_mb):
        rest = [x for x in chosen if x is not d]
        if not rest:
            continue
        s = _solve(meta, ctx, rest)
        if s:
            chosen, solved = rest, s
    return solved


def _build(meta, spec, replica_id, port_alloc, tier, order, counts, head_id) -> Placement:
    ctx = spec.ctx_size
    head_port = port_alloc(head_id)
    assignments: list[DeviceAssignment] = []
    start, rpc_i = 0, 0
    for i, (d, c) in enumerate(zip(order, counts)):
        est = _need(meta, ctx, d, start, c, i == len(order) - 1)
        if d.node.node_id == head_id and d.dev.kind == "cuda":
            llama_dev, endpoint = d.dev.device_id, None
        else:
            llama_dev = f"RPC{rpc_i}"
            endpoint = f"{d.node.host}:{port_alloc(d.node.node_id)}"
            rpc_i += 1
        assignments.append(
            DeviceAssignment(
                node_id=d.node.node_id, device_id=d.dev.device_id, llama_device=llama_dev,
                rpc_endpoint=endpoint, layers=c, est_mb=est, device_uuid=d.dev.uuid,
            )
        )
        start += c
    return Placement(
        model=spec.name, replica_id=replica_id, tier=tier, head_node=head_id, head_port=head_port,
        assignments=assignments, tensor_split=[float(a.layers) for a in assignments],
        est_total_mb=sum(a.est_mb for a in assignments),
    )
