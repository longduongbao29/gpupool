"""Preemption and cluster simulation helpers (design sections 4.4 and 6.4).

Pure functions over copies of node reports: no store, no I/O. The reconciler uses them to pick
which lower-priority replicas to stop when a higher-priority model has no room, and the
/api/simulate endpoint uses the same functions, so a preview shows what the reconciler would do.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gpupool.common.models import (
    Device, DeviceAssignment, ModelMeta, ModelSpec, NodeReport, Occupant, Placement, ReplicaRecord,
)

# rank_fn has the signature of scheduler.placement.rank:
#   rank_fn(meta, spec, nodes, occupants=..., limit=...) -> list[Placement]
RankFn = Callable[..., list[Placement]]


@dataclass
class Candidate:
    """A live replica that a higher-priority model might stop."""

    replica: ReplicaRecord
    priority: int  # its model's priority
    preemptible: bool  # its model's spec.preemptible
    above_min: bool  # its model has more active replicas than its running minimum
    busy: float  # 0..1


def _find_device(devices: list[Device], a: DeviceAssignment) -> Device | None:
    """Mirror of reconciler._find_device: by uuid when both sides have one (device_id is a position
    that shifts when a GPU drops off the bus), else by device_id."""
    if a.device_uuid and any(d.uuid for d in devices):
        return next((d for d in devices if d.uuid == a.device_uuid), None)
    return next((d for d in devices if d.device_id == a.device_id), None)


def _copy(reports: Sequence[NodeReport]) -> dict[str, NodeReport]:
    return {r.node_id: r.model_copy(deep=True) for r in reports}


def free_replicas(reports: Sequence[NodeReport], replicas: Sequence[ReplicaRecord],
                  disabled: set[tuple[str, str]] | frozenset = frozenset()) -> list[NodeReport]:
    """Deep copies of `reports` with every assignment's est_mb of `replicas` given back to the
    usable_mb of its device (matched by uuid when both sides have one, else by device_id),
    capped at the device's total_mb.

    `disabled` is a set of (node_id, device_id) the operator turned off: they arrive with usable 0
    and stay there, so preemption never makes a disabled GPU usable. Assignments on nodes absent
    from `reports` or on devices that no longer match are ignored."""
    copies = _copy(reports)
    for rec in replicas:
        for a in rec.placement.assignments:
            node = copies.get(a.node_id)
            d = _find_device(node.devices, a) if node else None
            if d is None or (a.node_id, d.device_id) in disabled:
                continue
            d.usable_mb = min(d.total_mb, d.usable_mb + a.est_mb)
    return [copies[r.node_id] for r in reports]


def apply_placement(reports: Sequence[NodeReport], placement: Placement) -> list[NodeReport]:
    """Deep copies of `reports` with the placement's est_mb taken from its devices (floor 0)."""
    copies = _copy(reports)
    for a in placement.assignments:
        node = copies.get(a.node_id)
        d = _find_device(node.devices, a) if node else None
        if d is not None:
            d.usable_mb = max(0, d.usable_mb - a.est_mb)
    return [copies[r.node_id] for r in reports]


def without_occupants(occupants: Sequence[Occupant], replica_ids: set[str]) -> list[Occupant]:
    """Occupants minus those of the given replicas."""
    return [o for o in occupants if o.replica_id not in replica_ids]


def placement_occupants(placement: Placement, busy: float = 0.0) -> list[Occupant]:
    """Occupants a new placement adds (one per assignment)."""
    return [Occupant(node_id=a.node_id, device_id=a.device_id, model=placement.model,
                     replica_id=placement.replica_id, est_mb=a.est_mb, busy=busy)
            for a in placement.assignments]


def find_victims(meta: ModelMeta, spec: ModelSpec, reports: Sequence[NodeReport],
                 occupants: Sequence[Occupant], candidates: Sequence[Candidate],
                 rank_fn: RankFn,
                 disabled: set[tuple[str, str]] | frozenset = frozenset()) -> list[ReplicaRecord] | None:
    """Smallest set of eligible replicas whose removal lets `spec` place one replica, or None.

    Eligible: preemptible and priority < spec.priority (equal priority never preempts). Taken in
    order: above_min first, then least busy, then newest. Greedy until rank_fn finds a placement
    on the freed reports, then each chosen victim is dropped again if the rest still suffices.

    `disabled` is passed through to free_replicas. At most n greedy + n minimisation rank_fn calls.
    Returned in eviction order (the greedy order, minus dropped ones). The result is empty only
    when spec already fits without evicting anything.
    """
    eligible = [c for c in candidates if c.preemptible and c.priority < spec.priority]
    # replica_id as the last key makes the order total, hence deterministic.
    eligible.sort(key=lambda c: (not c.above_min, c.busy, -c.replica.created_at, c.replica.replica_id))

    def fits(chosen: list[ReplicaRecord]) -> bool:
        ids = {r.replica_id for r in chosen}
        return bool(rank_fn(meta, spec, free_replicas(reports, chosen, disabled),
                            occupants=without_occupants(occupants, ids), limit=1))

    chosen: list[ReplicaRecord] = []
    for c in eligible:
        chosen.append(c.replica)
        if fits(chosen):
            break
    else:
        return None
    # Greedy order is preference, not need: a big late victim can make early ones redundant.
    for r in reversed(list(chosen)):
        rest = [x for x in chosen if x is not r]
        if fits(rest):
            chosen = rest
    return chosen
