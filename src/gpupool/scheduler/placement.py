"""Placement: choose devices and a layer split for one replica (design section 6).

Pure function of (meta, spec, nodes). Tiers in order: single_gpu, single_node,
multi_node. Ports are allocated only for the final plan.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from gpupool.common.models import (
    Device,
    DeviceAssignment,
    ModelMeta,
    ModelSpec,
    NodeReport,
    Placement,
)
from gpupool.scheduler.estimate import OVERHEAD_MB, device_need_mb, total_need_mb


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
    w = [max(1, d.dev.usable_mb - OVERHEAD_MB[d.dev.kind]) for d in order]
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


def plan(
    meta: ModelMeta,
    spec: ModelSpec,
    nodes: list[NodeReport],
    replica_id: str,
    port_alloc: Callable[[str], int],
    exclude_nodes: frozenset[str] | set[str] = frozenset(),
) -> Placement:
    ctx = spec.ctx_size
    pool = [
        _Dev(n, d)
        for n in nodes
        if n.node_id not in exclude_nodes
        for d in n.devices
        if d.usable_mb > 0
    ]
    # GPU-only first, across all tiers. Only when the GPUs of the pool cannot hold the
    # model do CPU devices join: within a tier they count as capacity like any GPU,
    # so mixing them in from the start would put layers in host RAM (10x+ slower)
    # even when spare VRAM exists on another node.
    gpu_pool = [d for d in pool if d.dev.kind == "cuda"]
    solved, tier = None, "single_gpu"
    passes = [(gpu_pool, True), (pool, False)] if gpu_pool and len(gpu_pool) < len(pool) \
        else [(pool, True)]
    for candidates, allow_single in passes:
        # In the CPU-inclusive pass a lone CPU device must not win "single_gpu":
        # GPU+CPU split beats CPU-only. CPU-only is allowed when there is no GPU at all.
        if allow_single:
            solved, tier = _tier_single_gpu(meta, ctx, candidates), "single_gpu"
        if solved is None:
            solved, tier = _tier_single_node(meta, ctx, candidates), "single_node"
        if solved is None:
            solved, tier = _tier_multi_node(meta, ctx, candidates), "multi_node"
        if solved is not None:
            break
    if solved is None:
        raise NoFit(
            f"model {spec.name!r} needs about {total_need_mb(meta, ctx)} MB at ctx {ctx}, "
            f"pool has {sum(d.dev.usable_mb for d in pool)} MB usable "
            f"across {len(pool)} devices (no feasible layer split)"
        )
    order, counts, head_id = solved
    return _build(meta, spec, replica_id, port_alloc, tier, order, counts, head_id)


def _tier_single_gpu(meta, ctx, pool):
    cands = []
    for d in pool:
        need = device_need_mb(meta, range(meta.n_layers), ctx, d.dev.kind, True)
        if need <= d.dev.usable_mb:
            cands.append(d)
    if not cands:
        return None
    best = min(cands, key=lambda d: (d.dev.kind != "cuda", d.dev.usable_mb, d.node.node_id, d.dev.device_id))
    return [best], [meta.n_layers], best.node.node_id


def _tier_single_node(meta, ctx, pool):
    best = None  # (n_devices, -leftover, node_id, solved)
    for node_id in sorted({d.node.node_id for d in pool}):
        devs = sorted((d for d in pool if d.node.node_id == node_id), key=lambda d: -d.dev.usable_mb)
        for k in range(2, len(devs) + 1):
            solved = _solve(meta, ctx, devs[:k])
            if solved is None:
                continue
            order, counts, _ = solved
            left = sum(_check(meta, ctx, order, counts))
            key = (k, -left, node_id)
            if best is None or key < best[0]:
                best = (key, solved)
            break
    return best[1] if best else None


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
                rpc_endpoint=endpoint, layers=c, est_mb=est,
            )
        )
        start += c
    return Placement(
        model=spec.name, replica_id=replica_id, tier=tier, head_node=head_id, head_port=head_port,
        assignments=assignments, tensor_split=[float(a.layers) for a in assignments],
        est_total_mb=sum(a.est_mb for a in assignments),
    )
