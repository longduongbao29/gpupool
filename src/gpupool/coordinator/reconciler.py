"""Reconcile loop: failure detection, desired-count enforcement, launch, drain."""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import (
    ACTIVE_STATES,
    LIVE_STATES,
    Device,
    DeviceAssignment,
    EngineSpec,
    ModelMeta,
    ModelSpec,
    NodeReport,
    Occupant,
    Placement,
    ReplicaRecord,
)
from gpupool.common.net import internal_client
from gpupool.coordinator import preemption
from gpupool.coordinator.agent_client import AgentClient
from gpupool.coordinator.autoscaler import bounds
from gpupool.coordinator.events import Notifier
from gpupool.coordinator.store import NodeRecord, Store, gpu_key, planning_factor
from gpupool.scheduler.estimate import CONTEXT_MB
from gpupool.scheduler.placement import NoFit, plan, rank
from gpupool.scheduler.scoring import (
    ETA_RANGE, bandwidth_seconds, default_cuda_bw, logits_seconds, set_speed_model, speed_model,
)

log = logging.getLogger("gpupool.reconciler")


class LaunchError(Exception):
    pass


class _Superseded(Exception):
    """The replica was drained/failed while launching: roll back quietly, not a launch failure."""


@dataclass
class _Realloc:
    """A model's pending re-allocation (lost replica -> new placement -> serving again)."""
    since: float
    lost: tuple[str, str] | None  # (replica_id, reason)
    started: bool = False


def _port_of(endpoint: str) -> int:
    return int(endpoint.rsplit(":", 1)[1])


def _find_device(devices: list[Device], a: DeviceAssignment) -> Device | None:
    """The card an assignment was planned on. By uuid when both sides have one: device_id is a
    position that shifts when a GPU drops off the bus, so matching it would blame a healthy replica
    for the lost card's neighbour, or miss the loss when another card took the old id. Without uuids
    in the report (agent downgrade) or on the assignment, fall back to device_id."""
    if a.device_uuid and any(d.uuid for d in devices):
        return next((d for d in devices if d.uuid == a.device_uuid), None)
    return next((d for d in devices if d.device_id == a.device_id), None)


def _draft_device(p: Placement) -> DeviceAssignment | None:
    """The head's first local CUDA device: where the scheduler reserved the draft model."""
    return next((a for a in p.assignments if a.node_id == p.head_node and not a.rpc_endpoint
                 and a.llama_device.startswith("CUDA")), None)


def is_head_engine(engine_id: str) -> bool:
    return engine_id.endswith("-head")


async def stop_head_first(pairs, stop) -> None:
    """Stop the head engines (llama-server) and wait for them to exit, then the RPC servers.

    A head frees its buffers on the RPC servers while it shuts down; if an RPC server is already
    gone that send fails and llama.cpp aborts (ggml-rpc.cpp "Remote RPC server crashed", SIGABRT,
    a core dump each time). Stopping everything at once lost that race on every stop, because
    ggml-rpc-server exits immediately while llama-server takes a moment to clean up."""
    pairs = list(pairs)
    heads = [p for p in pairs if is_head_engine(p[1])]
    rest = [p for p in pairs if not is_head_engine(p[1])]
    for group in (heads, rest):
        if group:
            await asyncio.gather(*(stop(a, b) for a, b in group))


def rpc_servers(p: Placement) -> list[tuple[str, list[DeviceAssignment]]]:
    """(endpoint, its assignments in device order) per RPC server of a placement, in --rpc order.
    Several devices share an endpoint when one server process serves them all."""
    out: dict[str, list[DeviceAssignment]] = {}
    for a in p.assignments:
        if a.rpc_endpoint:
            out.setdefault(a.rpc_endpoint, []).append(a)
    return list(out.items())


def rpc_engine_id(replica_id: str, devices: list[DeviceAssignment]) -> str:
    return f"{replica_id}-rpc-{devices[0].device_id}"


def engine_ids(rec: ReplicaRecord) -> list[tuple[str, str]]:
    """(node_id, engine_id) for every engine this replica owns."""
    p = rec.placement
    out = [(p.head_node, f"{rec.replica_id}-head")]
    for _, devs in rpc_servers(p):
        out.append((devs[0].node_id, rpc_engine_id(rec.replica_id, devs)))
    return out


class Reconciler:
    READY_REPORT_GRACE_S = 5.0
    DEAD_AFTER_FAILED_POLLS = 2  # stale report AND this many failed polls in a row = node dead
    STABLE_S = 300.0  # a replica ready this long counts as healthy: crashes before it feed the backoff
    KEEP_TERMINAL_PER_MODEL = 10  # stopped/failed history rows kept per model
    # A model that evicted others may not do it again for this long, so two models whose needs
    # overlap cannot keep stopping each other's replicas.
    PREEMPT_COOLDOWN_S = 600.0
    PREEMPT_CLAIM_GRACE_S = 120.0  # after the drain timeout, time for the preemptor to be placed
    # A move reloads a whole model, so it must be clearly better, not just better (score points).
    REBALANCE_MIN_GAIN = 25.0
    REBALANCE_EXTRA_TIMEOUT_S = 60.0  # slack on top of launch_timeout_s before a move is abandoned
    CAL_ALPHA = 0.5  # EMA weight of a new calibration sample
    SPEED_SAMPLE_S = 60.0  # how often ready replicas feed the speed model
    SPEED_ALPHA = 0.2  # EMA weight of one speed sample (one per replica per SPEED_SAMPLE_S)
    CAL_EVENT_DELTA = 0.05  # emit `calibrated` when the planning factor moves by more than this

    def __init__(
        self,
        store: Store,
        cfg: CoordinatorConfig,
        client: AgentClient,
        meta_for: Callable[[ModelSpec], Awaitable[ModelMeta]],
        outstanding: Callable[[str], int],
        clock: Callable[[], float] = time.time,
        notifier: Notifier | None = None,
    ):
        self.store = store
        self.cfg = cfg
        self.client = client
        self.meta_for = meta_for
        self.outstanding = outstanding
        self.clock = clock
        self.poller = None  # set by the app; None = pure last_seen time rule (tests)
        self.autoscaler = None  # set by the app; None keeps the fixed `replicas` behaviour
        self.notifier = notifier or Notifier(store, getattr(cfg, "webhook_url", ""), clock=clock)
        self.poll_s = 0.5  # engine/health poll interval during launch
        self.planner: Callable | None = None  # tests inject; default is scheduler.placement.plan
        self.ranker: Callable | None = None  # tests inject; default is scheduler.placement.rank
        self._http: httpx.AsyncClient | None = None
        self._launches: dict[str, asyncio.Task] = {}
        self._suspect: set[str] = set()
        self._tick_lock = asyncio.Lock()
        self._nofit: dict[str, str] = {}
        self._backoff: dict[str, tuple[int, float]] = {}  # model -> (consecutive failures, retry not before)
        self._node_up: dict[str, bool] = {}  # node_id -> last observed liveness (for transition events)
        self._versions_seen: tuple[str, ...] = ()  # llama.cpp builds of live servers at the last check
        self._last_speed_sample = float("-inf")
        self._realloc: dict[str, _Realloc] = {}
        self._preempted: dict[str, tuple[float, set[str]]] = {}  # model -> (when, victim replica ids)
        self._wake = asyncio.Event()
        # The one replica move in flight: {model, old, new, since}. Cluster-wide, so at most one.
        self._move: dict | None = None
        self._move_text: tuple[str, str, float] = ("", "", 0.0)  # (from, to, gain) for the events
        # Starts at boot, not 0: right after a restart the reports are stale and a move would be a guess.
        self._last_rebalance = clock()
        self._load_state()

    # ------------------------------------------------------------------ persistence
    # Control state is written through to the store on every change (rare events) and loaded here, so a
    # coordinator restart keeps preemption cooldowns, the move in flight and crash-loop backoffs.
    # Times are wall-clock (clock()), so they mean the same after a restart. _last_rebalance is NOT
    # persisted: it deliberately restarts at boot, because right after a restart the reports are stale.
    def _load_state(self) -> None:
        try:
            pre = self.store.get_state("preempted") or {}
            self._preempted = {m: (float(v[0]), set(v[1])) for m, v in pre.items()}
            bo = self.store.get_state("backoff") or {}
            self._backoff = {m: (int(v[0]), float(v[1])) for m, v in bo.items()}
            sm = self.store.get_state("speed_model")
            if sm:
                set_speed_model(float(sm["eta"]), float(sm["hop_s"]))
            saved = self.store.get_state("move")
            if saved:
                mv = saved["move"]
                # A move whose replicas are gone has nothing to resume: clear it quietly.
                if self.store.get_replica(mv["old"]) is None or self.store.get_replica(mv["new"]) is None:
                    self.store.delete_state("move")
                else:
                    self._move = dict(mv)
                    self._move_text = (str(saved["text"][0]), str(saved["text"][1]), float(saved["text"][2]))
        except Exception:
            log.exception("loading saved coordinator state failed; starting clean")
            self._preempted, self._backoff, self._move = {}, {}, None

    def _save(self, key: str, value: dict | None) -> None:
        """Write-through; a store problem must not break reconciliation (state stays in memory)."""
        try:
            if value:
                self.store.put_state(key, value, now=self.clock())
            else:
                self.store.delete_state(key)
        except Exception:
            log.exception("saving coordinator state %r failed", key)

    def _save_preempted(self) -> None:
        self._save("preempted", {m: [w, sorted(ids)] for m, (w, ids) in self._preempted.items()})

    def _save_backoff(self) -> None:
        self._save("backoff", {m: [n, t] for m, (n, t) in self._backoff.items()})

    def _set_move(self, move: dict | None, text: tuple[str, str, float] | None = None) -> None:
        self._move = move
        if text is not None:
            self._move_text = text
        self._save("move", {"move": move, "text": list(self._move_text)} if move else None)

    # ------------------------------------------------------------------ helpers
    def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = internal_client(timeout=httpx.Timeout(5.0))
        return self._http

    def _alive(self, rec: NodeRecord, now: float) -> bool:
        if rec.alive(now, self.cfg.heartbeat_timeout_s):
            return True
        # A stale report may be OUR event loop stalling, not the agent dying. With a poller,
        # dead needs evidence: DEAD_AFTER_FAILED_POLLS polls that really failed.
        if self.poller is None:
            return False
        return self.poller.failed_polls(rec.report.node_id) < self.DEAD_AFTER_FAILED_POLLS

    def node_alive(self, rec: NodeRecord, now: float) -> bool:
        """Public liveness rule shared by the router, autoscaler and API views."""
        return self._alive(rec, now)

    def _node_map(self) -> dict[str, NodeRecord]:
        # Only registered servers count: a stray heartbeat must not receive placements.
        registered = {s.node_id for s in self.store.list_servers()}
        return {n.report.node_id: n for n in self.store.list_nodes() if n.report.node_id in registered}

    def _planner(self) -> Callable:
        if self.planner is not None:
            return self.planner
        return plan

    def _ranker(self) -> Callable:
        return self.ranker if self.ranker is not None else rank

    def note_error(self, replica_id: str) -> None:
        self._suspect.add(replica_id)

    def nofit_reason(self, model: str) -> str | None:
        """Why the last attempt to plan `model` failed, or None if it was planned since."""
        return self._nofit.get(model)

    # ------------------------------------------------------------------ planning
    def _used_ports(self, node_id: str) -> set[int]:
        used: set[int] = set()
        for r in self.store.list_replicas(states=set(LIVE_STATES)):
            p = r.placement
            if p.head_node == node_id:
                used.add(p.head_port)
            for a in p.assignments:
                if a.rpc_endpoint and a.node_id == node_id:
                    used.add(_port_of(a.rpc_endpoint))
        return used

    def _reserved_nodes(self, nodes: dict[str, NodeRecord], now: float) -> list[NodeReport]:
        """Alive reports with usable_mb reduced by what launching replicas will still claim."""
        reserved: dict[tuple[str, str], int] = {}
        for r in self.store.list_replicas(states={"launching", "ready"}):
            for a in r.placement.assignments:
                # A ready replica keeps its reservation until its node has reported again
                # well after it became ready: until then free_mb may predate the model load,
                # and the next plan would hand the same memory out twice.
                if r.state == "ready":
                    n = nodes.get(a.node_id)
                    if n is not None and n.last_seen > r.updated_at + self.READY_REPORT_GRACE_S:
                        continue
                key = (a.node_id, a.device_id)
                reserved[key] = reserved.get(key, 0) + a.est_mb
        flags = self.store.gpu_flags()
        live = list(self.store.list_replicas(states=set(LIVE_STATES)))
        out = []
        for n in nodes.values():
            if not self._alive(n, now):
                continue
            rep = n.report.model_copy(deep=True)
            for d in rep.devices:
                if not flags.get((rep.node_id, gpu_key(d)), True):
                    # Disabled in the pool: usable_mb 0 makes the unchanged planner skip it.
                    d.usable_mb = 0
                    continue
                d.usable_mb = max(0, d.usable_mb - reserved.get((rep.node_id, d.device_id), 0))
                if d.budget_mb is not None:
                    # The budget caps everything we hold, however long ago it loaded: free memory
                    # stops showing our own share once the grace above has passed.
                    own = sum(a.est_mb for r in live for a in r.placement.assignments
                              if a.node_id == rep.node_id and _find_device(rep.devices, a) is d)
                    d.usable_mb = min(d.usable_mb, max(0, d.budget_mb - own))
            out.append(rep)
        return out

    def available_reports(self) -> list[NodeReport]:
        """What a new replica could use: alive registered nodes, disabled GPUs at usable 0,
        launching replicas' reservations subtracted. Private copies, safe to edit."""
        return self._reserved_nodes(self._node_map(), self.clock())

    def occupants(self) -> list[Occupant]:
        """Every assignment of every live replica (draining still holds memory), on the device's
        current id: stored device_ids are positions that shift, so they are resolved by uuid."""
        nodes = self._node_map()
        specs = {s.name: s for s in self.store.list_models()}
        out: list[Occupant] = []
        for r in self.store.list_replicas(states=set(LIVE_STATES)):
            spec = specs.get(r.model)
            busy = min(1.0, self.outstanding(r.replica_id) / max(1, spec.parallel if spec else 1))
            for a in r.placement.assignments:
                n = nodes.get(a.node_id)
                d = _find_device(n.report.devices, a) if n is not None else None
                if d is None:
                    continue
                out.append(Occupant(node_id=a.node_id, device_id=d.device_id, model=r.model,
                                    replica_id=r.replica_id, est_mb=a.est_mb, busy=busy))
        return out

    @staticmethod
    def _apply_pins(spec: ModelSpec, reports: list[NodeReport]) -> None:
        """Hide every device outside the model's allowed set from the planner; the scheduler still
        picks the best placement among the rest. "<node>/*" allows all of that server's devices,
        including GPUs added to it later."""
        if spec.pin_devices:
            pins = set(spec.pin_devices)
            for rep in reports:
                whole = f"{rep.node_id}/*" in pins
                for d in rep.devices:
                    if not whole and f"{rep.node_id}/{d.device_id}" not in pins:
                        d.usable_mb = 0

    async def draft_meta_for(self, spec: ModelSpec) -> ModelMeta | None:
        """Meta of the draft model, or None when `spec` does not use one."""
        if spec.speculative != "draft" or not spec.draft:
            return None
        return await self.meta_for(spec.model_copy(update={"source": spec.draft}))

    @staticmethod
    def _dkw(draft_meta: ModelMeta | None) -> dict:
        # Only passed when there is a draft, so planners/rankers without the parameter keep working.
        return {} if draft_meta is None else {"draft_meta": draft_meta}

    def _pkw(self, model: str, draft_meta: ModelMeta | None = None) -> dict:
        """Extra planner/ranker keywords for `model`: the draft and the calibrated memory factor.
        Each only when it applies (factor != 1.0), so fakes without the parameters keep working."""
        kw = self._dkw(draft_meta)
        try:
            f = self.store.mem_factor(model)
        except Exception:
            log.exception("reading the memory factor of %s failed", model)
            f = 1.0
        if f != 1.0:
            kw["mem_factor"] = f
        return kw

    async def rank_for(self, spec: ModelSpec, limit: int) -> list[Placement]:
        """Best placements for `spec` right now, without ports or a replica id. No side effects."""
        meta = await self.meta_for(spec)
        draft = await self.draft_meta_for(spec)
        reports = self.available_reports()
        self._apply_pins(spec, reports)
        occupants, kw = self.occupants(), self._pkw(spec.name, draft)  # store reads stay here
        # Ranking is pure CPU (hundreds of ms on a large pool) and Recommend ranks a dozen
        # variants: on the event loop that would stall every streamed response for as long.
        # Its inputs are private copies, so a worker thread can have them.
        return await asyncio.to_thread(self._ranker(), meta, spec, reports, occupants=occupants,
                                       limit=limit, **kw)

    async def plan_for(self, spec: ModelSpec, replica_id: str | None = None) -> Placement:
        """Plan one replica. No side effects (ports are only 'handed out' within this call)."""
        reports = self.available_reports()
        self._apply_pins(spec, reports)
        # Only live processes hold a port; exited/failed engines linger in reports until stopped.
        reported_ports = {r.node_id: {e.port for e in r.engines if e.state in ("starting", "running")}
                          for r in reports}
        handed: dict[str, set[int]] = {}
        lo, hi = self.cfg.port_range
        used_cache: dict[str, set[int]] = {}

        def port_alloc(node_id: str) -> int:
            if node_id not in used_cache:
                used_cache[node_id] = self._used_ports(node_id) | reported_ports.get(node_id, set())
            taken = used_cache[node_id] | handed.setdefault(node_id, set())
            for p in range(lo, hi + 1):
                if p not in taken:
                    handed[node_id].add(p)
                    return p
            raise RuntimeError(f"no free port in {lo}-{hi} on node {node_id}")

        rid = replica_id or self._new_replica_id(spec.name)
        meta = await self.meta_for(spec)
        draft = await self.draft_meta_for(spec)
        return self._planner()(meta, spec, reports, rid, port_alloc, occupants=self.occupants(),
                               **self._pkw(spec.name, draft))

    @staticmethod
    def _new_replica_id(model: str) -> str:
        return f"{re.sub(r'[^A-Za-z0-9_.-]', '-', model)}-{uuid.uuid4().hex[:6]}"

    # ------------------------------------------------------------------ preemption
    @staticmethod
    def _floor(spec: ModelSpec) -> int:
        """The replica count below which a model counts as under its running minimum."""
        return max(bounds(spec)[0], 1)

    def _wanted(self, spec: ModelSpec) -> int:
        return self.autoscaler.desired(spec) if self.autoscaler is not None else spec.replicas

    def _constrain(self, spec: ModelSpec, nodes: Sequence[NodeReport]) -> list[NodeReport]:
        """Copies of `nodes` with GPUs switched off in the pool and GPUs the model is not pinned
        to at usable 0. preemption.free_replicas hands memory back to every device, so this is
        re-applied after freeing: a preemption must never place a model where a normal launch
        could not go."""
        flags = self.store.gpu_flags()
        reps = [n.model_copy(deep=True) for n in nodes]
        for rep in reps:
            for d in rep.devices:
                if not flags.get((rep.node_id, gpu_key(d)), True):
                    d.usable_mb = 0
        self._apply_pins(spec, reps)
        return reps

    def _disabled(self, nodes: Sequence[NodeReport]) -> frozenset[tuple[str, str]]:
        """(node_id, device_id) of GPUs switched off in the pool, for preemption.free_replicas."""
        flags = self.store.gpu_flags()
        return frozenset((n.node_id, d.device_id) for n in nodes for d in n.devices
                         if not flags.get((n.node_id, gpu_key(d)), True))

    def _ranker_for(self, spec: ModelSpec, draft_meta: ModelMeta | None = None) -> Callable:
        """The ranker, with `spec`'s constraints re-applied to whatever reports it is handed and
        the draft model (if any) accounted for, so preemption and simulation size it like a launch."""
        base = self._ranker()
        extra = self._pkw(spec.name, draft_meta)

        def ranked(meta, spec_, nodes, occupants=(), limit=5):
            return base(meta, spec_, self._constrain(spec, nodes), occupants=occupants, limit=limit, **extra)

        return ranked

    def _candidates(self, specs: dict[str, ModelSpec], reps: Sequence[ReplicaRecord],
                    exclude: str | None) -> list[preemption.Candidate]:
        """Active replicas of other models, annotated for preemption.find_victims."""
        count = Counter(r.model for r in reps)
        out = []
        for r in reps:
            s = specs.get(r.model)
            if s is None or r.model == exclude:
                continue
            busy = min(1.0, self.outstanding(r.replica_id) / max(1, s.parallel))
            out.append(preemption.Candidate(replica=r, priority=s.priority, preemptible=s.preemptible,
                                            above_min=count[r.model] > self._floor(s), busy=busy))
        return out

    async def _find_victims(self, spec: ModelSpec, exclude: str | None) -> list[ReplicaRecord] | None:
        specs = {s.name: s for s in self.store.list_models()}
        cands = self._candidates(specs, self.store.list_replicas(states=set(ACTIVE_STATES)), exclude)
        if not cands:
            return None
        meta = await self.meta_for(spec)
        draft = await self.draft_meta_for(spec)
        reports = self.available_reports()
        self._apply_pins(spec, reports)
        return preemption.find_victims(meta, spec, reports, self.occupants(), cands,
                                        self._ranker_for(spec, draft),
                                        disabled=self._disabled(reports))

    async def rank_with_preemption(self, spec: ModelSpec, limit: int, exclude: str | None = None,
                                   ) -> tuple[list[ReplicaRecord], list[Placement]] | None:
        """(victims, placements once they are gone), or None when evicting lower-priority replicas
        would not help. No side effects. `exclude`: a model whose replicas are never victims
        (the one being placed)."""
        victims = await self._find_victims(spec, exclude)
        if not victims:
            return None
        reports = self.available_reports()
        reports = preemption.free_replicas(reports, victims, self._disabled(reports))
        occ = preemption.without_occupants(self.occupants(), {v.replica_id for v in victims})
        ranked = self._ranker_for(spec, await self.draft_meta_for(spec))(
            await self.meta_for(spec), spec, reports, occupants=occ, limit=limit)
        return (victims, ranked) if ranked else None

    def _claim_priority(self, specs: dict[str, ModelSpec], wanted: dict[str, int], now: float) -> int:
        """Priority of the highest model still waiting for room it preempted, else -1.

        While victims drain, a draining replica is no longer active, so its own model would see a
        deficit and relaunch into the memory being freed (seen on real hardware: the evicted model
        came straight back and the preemptor stayed stuck behind the cooldown). Lower priorities
        therefore do not launch until the preemptor has its replica or the claim expires."""
        hold = self.cfg.drain_timeout_s + self.PREEMPT_CLAIM_GRACE_S
        best = -1
        for name, (when, _ids) in self._preempted.items():
            spec = specs.get(name)
            if spec is None or now - when > hold:
                continue
            active = len(self.store.list_replicas(model=name, states=set(ACTIVE_STATES)))
            if active < min(wanted.get(name, 0), self._floor(spec)):
                best = max(best, spec.priority)
        return best

    async def _preempt_for(self, spec: ModelSpec, now: float, wanted: int) -> None:
        """Stop lower-priority replicas so `spec` can place one, when it has no room.

        Only a model below its running minimum may evict, so autoscale extras never do. The
        model is not placed here: draining replicas still hold their memory, so a later tick
        places it once they are stopped. Pins naming devices that do not exist are not special-
        cased: they simply find no victims."""
        active = len(self.store.list_replicas(model=spec.name, states=set(ACTIVE_STATES)))
        if active >= min(wanted, self._floor(spec)):
            return
        prev = self._preempted.get(spec.name)
        if prev is not None:
            when, ids = prev
            if now - when < self.PREEMPT_COOLDOWN_S:
                return
            if any((r := self.store.get_replica(i)) is not None and r.state in LIVE_STATES for i in ids):
                return  # earlier victims still drain: wait for their memory instead of evicting more
        try:
            victims = await self._find_victims(spec, exclude=spec.name)
        except Exception:
            log.exception("looking for preemption victims for %s failed", spec.name)
            return
        if not victims:
            return
        specs = {s.name: s for s in self.store.list_models()}
        for v in victims:
            vs = specs.get(v.model)
            await self.drain(v.replica_id)
            self._emit("warning", "preempted",
                       f"Stopped replica {v.replica_id} of {v.model} (priority {vs.priority if vs else 0}) "
                       f"to make room for {spec.name} (priority {spec.priority})",
                       node_id=v.placement.head_node, model=v.model)
        self._preempted[spec.name] = (now, {v.replica_id for v in victims})
        self._save_preempted()

    # ------------------------------------------------------------------ simulation
    async def simulate(self, specs: list[ModelSpec]) -> dict:
        """What the next ticks would do if the models were `specs`. No side effects.

        Same order and rules as _enforce_counts/_preempt_for, except: the preemption cooldown is
        ignored, memory freed by a stop is available to every model (the real cluster frees it a
        tick later), and busyness of replicas the simulation starts is 0."""
        stored = {s.name for s in self.store.list_models()}
        by_name = {s.name: s for s in specs}
        live = [r for r in self.store.list_replicas(states=set(ACTIVE_STATES)) if r.model in by_name]
        reports, occ = self.available_reports(), self.occupants()
        gone: set[str] = set()
        out: dict = {"start": [], "stop": [], "preempt": [], "unplaced": []}

        def wanted_of(s: ModelSpec) -> int:
            if s.name in stored:
                return s.replicas if self.autoscaler is None else self.autoscaler.peek(s)
            return self._floor(s) if s.replicas > 0 else 0  # a model that does not exist yet starts at its floor

        def active_of(name: str) -> list[ReplicaRecord]:
            return [r for r in live if r.model == name and r.replica_id not in gone]

        wanted = {n: wanted_of(s) for n, s in by_name.items()}
        order = sorted(specs, key=lambda s: (-s.priority, bool(active_of(s.name)), s.name))

        def release(recs: list[ReplicaRecord]) -> None:
            nonlocal reports, occ
            gone.update(r.replica_id for r in recs)
            reports = preemption.free_replicas(reports, recs, self._disabled(reports))
            occ = preemption.without_occupants(occ, {r.replica_id for r in recs})

        # Stops first: their memory is what the deficits below can use.
        for s in order:
            act = sorted(active_of(s.name), key=lambda r: r.created_at, reverse=True)
            for r in act[: max(0, len(act) - wanted[s.name])]:
                out["stop"].append({"replica_id": r.replica_id, "model": s.name,
                                    "reason": f"{len(act)} running, {wanted[s.name]} wanted"})
                release([r])

        for s in order:
            missing = wanted[s.name] - len(active_of(s.name))
            started = 0
            try:
                meta = await self.meta_for(s)
                draft = await self.draft_meta_for(s)
            except Exception as e:
                if missing > 0:
                    out["unplaced"].append({"model": s.name, "missing": missing,
                                            "why": f"{type(e).__name__}: {e}"})
                continue
            ranker = self._ranker_for(s, draft)
            for _ in range(max(0, missing)):
                ranked = ranker(meta, s, reports, occupants=occ, limit=1)
                if not ranked and len(active_of(s.name)) + started < min(wanted[s.name], self._floor(s)):
                    cands = self._candidates(by_name, [r for r in live if r.replica_id not in gone], s.name)
                    victims = (preemption.find_victims(meta, s, reports, occ, cands, ranker,
                                                         disabled=self._disabled(reports))
                               if cands else None)
                    if victims:
                        for v in victims:
                            out["preempt"].append({"replica_id": v.replica_id, "model": v.model,
                                                   "priority": by_name[v.model].priority, "for_model": s.name})
                        release(victims)
                        ranked = ranker(meta, s, reports, occupants=occ, limit=1)
                if not ranked:
                    out["unplaced"].append({"model": s.name, "missing": missing - started,
                                            "why": self._why_unplaced(meta, s, reports, occ, draft)})
                    break
                p = ranked[0]
                out["start"].append({"model": s.name, "tier": p.tier, "est_decode_tps": p.est_decode_tps,
                                     "assignments": [{"node_id": a.node_id, "device_id": a.device_id,
                                                      "layers": a.layers, "est_mb": a.est_mb}
                                                     for a in p.assignments]})
                reports = preemption.apply_placement(reports, p)
                occ = occ + preemption.placement_occupants(p)
                started += 1
        return out

    def _why_unplaced(self, meta: ModelMeta, spec: ModelSpec, reports: list[NodeReport],
                      occ: list[Occupant], draft: ModelMeta | None = None) -> str:
        """The planner's own NoFit message for the simulated cluster state."""
        try:
            self._planner()(meta, spec, self._constrain(spec, reports), "sim", lambda _n: 0, occupants=occ,
                           **self._pkw(spec.name, draft))
        except Exception as e:
            return f"{type(e).__name__}: {e}"
        return "no placement found"

    # ------------------------------------------------------------------ rebalancing
    @staticmethod
    def _where(assignments) -> list[dict]:
        return [{"node_id": a.node_id, "device_id": a.device_id} for a in assignments]

    @staticmethod
    def _where_text(where: list[dict]) -> str:
        return ", ".join(f"{w['node_id']}/{w['device_id']}" for w in where)

    async def rebalance_candidates(self) -> list[dict]:
        """Ready replicas that would score REBALANCE_MIN_GAIN or more somewhere else, best first.
        No side effects.

        Each replica is scored in the same pass as its alternatives (`extra`), with its own memory
        removed from the occupants but still subtracted from the reports: make-before-break needs
        the new place to fit while the old replica is still running."""
        specs = {s.name: s for s in self.store.list_models()}
        occ_all = self.occupants()
        out: list[dict] = []
        for r in self.store.list_replicas(states={"ready"}):
            spec = specs.get(r.model)
            if spec is None:
                continue
            try:
                meta = await self.meta_for(spec)
                draft = await self.draft_meta_for(spec)
                reports = self.available_reports()
                self._apply_pins(spec, reports)  # a move stays within the model's pins
                ranked = self._ranker()(meta, spec, reports, limit=5, extra=[r.placement],
                                        occupants=[o for o in occ_all if o.replica_id != r.replica_id],
                                        **self._pkw(spec.name, draft))
            except Exception:
                log.exception("scoring a rebalance for %s failed", r.replica_id)
                continue
            cur = next((p for p in ranked if p.replica_id == r.replica_id), None)
            cands = [p for p in ranked if p.replica_id == "" and p.score is not None]
            if cur is None or cur.score is None or not cands:
                continue
            best = max(cands, key=lambda p: p.score)
            gain = best.score - cur.score
            if gain < self.REBALANCE_MIN_GAIN:
                continue
            out.append({"replica_id": r.replica_id, "model": r.model,
                        "from": self._where(r.placement.assignments), "to": self._where(best.assignments),
                        "current_score": cur.score, "new_score": best.score, "gain": gain,
                        "reasons": list(best.reasons)})
        out.sort(key=lambda m: m["gain"], reverse=True)
        return out

    def _not_quiet(self, now: float) -> str | None:
        """Why the cluster is not in a state to start a move, or None."""
        if self._move is not None:
            return "a move is already in progress"
        if self.store.list_replicas(states={"pending", "launching"}):
            return "a replica is launching"
        specs = {s.name: s for s in self.store.list_models()}
        if self._claim_priority(specs, {n: self._wanted(s) for n, s in specs.items()}, now) >= 0:
            return "a preemption is waiting for its memory"
        return None

    async def start_move(self, move: dict) -> bool:
        """Start the make-before-break move `move` (an entry of rebalance_candidates()). False when
        refused: nothing was changed. Takes the tick lock so it cannot interleave with a tick."""
        async with self._tick_lock:
            return await self._start_move(move, self.clock())

    async def _start_move(self, move: dict, now: float) -> bool:
        why = self._not_quiet(now)
        old = self.store.get_replica(move["replica_id"])
        spec = next((s for s in self.store.list_models() if s.name == move["model"]), None)
        if why is None and (old is None or old.state != "ready" or spec is None):
            why = "the replica is no longer ready"
        if why is not None:
            log.info("rebalance of %s refused: %s", move["replica_id"], why)
            return False
        pins = [f"{w['node_id']}/{w['device_id']}" for w in move["to"]]
        try:
            # Pinned to the move's target so the planner (real ports) cannot choose elsewhere.
            placement = await self.plan_for(spec.model_copy(update={"pin_devices": pins}))
        except Exception as e:
            log.warning("rebalance of %s: cannot plan the new replica: %s", old.replica_id, e)
            return False
        rec = self._spawn_launch(spec, placement, now)  # real spec: the pins were for planning only
        self._set_move({"model": spec.name, "old": old.replica_id, "new": rec.replica_id, "since": now},
                       (self._where_text(move["from"]), self._where_text(move["to"]), float(move["gain"])))
        self._emit("info", "rebalance_started",
                   f"Moving replica {old.replica_id} of {spec.name} from {self._move_text[0]} to "
                   f"{self._move_text[1]} (score {move['gain']:+.0f})",
                   node_id=old.placement.head_node, model=spec.name)
        return True

    async def _advance_move(self, now: float) -> None:
        """Finish or abandon the move in flight. Runs before _enforce_counts every tick."""
        mv = self._move
        if mv is None:
            return
        old, new = self.store.get_replica(mv["old"]), self.store.get_replica(mv["new"])
        frm, to, gain = self._move_text
        if new is not None and new.state == "ready" and old is not None and old.state == "ready":
            await self.drain(old.replica_id)
            self._set_move(None)
            self._emit("info", "rebalanced",
                       f"Moved replica {old.replica_id} of {mv['model']} to {to} (score {gain:+.0f})",
                       node_id=new.placement.head_node, model=mv["model"])
            return
        reason = None
        if new is None or new.state not in ("pending", "launching", "ready"):
            reason = f"the new replica {'vanished' if new is None else new.state}"
            if new is not None and new.error:
                reason += f": {new.error[:200]}"
        elif old is None or old.state != "ready":
            reason = f"the old replica is {'gone' if old is None else old.state}"
        elif now - mv["since"] > self.cfg.launch_timeout_s + self.REBALANCE_EXTRA_TIMEOUT_S:
            reason = "the new replica did not become ready in time"
        if reason is None:
            return
        # The old replica keeps serving; a new one still alive is surplus and drains normally.
        self._set_move(None)
        self._emit("warning", "rebalance_failed",
                   f"Move of replica {mv['old']} of {mv['model']} from {frm} to {to} abandoned: {reason}",
                   model=mv["model"])

    async def _rebalance_if_due(self, now: float) -> None:
        every = self.cfg.rebalance_s
        if every <= 0 or now - self._last_rebalance < every or self._not_quiet(now) is not None:
            return  # not quiet: stay due and try again next tick
        self._last_rebalance = now
        try:
            moves = await self.rebalance_candidates()
        except Exception:
            log.exception("looking for rebalance moves failed")
            return
        if moves:
            await self._start_move(moves[0], now)

    def rebalance_state(self) -> dict:
        """For /api/state: the move in flight and when the periodic run is next due."""
        due = self._last_rebalance + self.cfg.rebalance_s if self.cfg.rebalance_s > 0 else None
        return {"in_progress": dict(self._move) if self._move else None, "next_run_ts": due}

    # ------------------------------------------------------------------ server removal
    async def remove_node(self, node_id: str) -> None:
        """Stop every replica touching `node_id`, then forget the server.

        Replicas are marked stopped (not failed) so the desired count is simply re-placed
        elsewhere by the next tick.
        """
        async with self._tick_lock:  # do not race a tick that is launching onto this node
            now = self.clock()
            nodes = self._node_map()
            for rec in self.store.list_replicas(states=set(LIVE_STATES)):
                p = rec.placement
                if node_id != p.head_node and node_id not in {a.node_id for a in p.assignments}:
                    continue
                self.store.set_replica_state(rec.replica_id, "stopped", None, now=now)
                t = self._launches.get(rec.replica_id)
                if t is not None:
                    t.cancel()
                    await asyncio.gather(t, return_exceptions=True)  # its rollback stops created engines
                await self._stop_engines(rec, nodes, now)
            self.store.delete_server(node_id)

    # ------------------------------------------------------------------ tick
    async def tick(self) -> None:
        async with self._tick_lock:
            now = self.clock()
            nodes = self._node_map()
            self._track_nodes(nodes, now)
            await self._fail_orphaned_launches(nodes, now)
            await self._detect_failures(nodes, now)
            await self._check_suspects(nodes)
            await self._process_drains(nodes, now)
            self._clear_stable_backoff(now)
            await self._advance_move(now)
            await self._enforce_counts(nodes, now)
            await self._rebalance_if_due(now)
            await self._learn_speed(nodes, now)
            self.store.prune_replicas(self.KEEP_TERMINAL_PER_MODEL)

    async def run(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("reconcile tick failed")
            try:  # sleep until the next period, or earlier when wake() is called
                await asyncio.wait_for(self._wake.wait(), self.cfg.reconcile_s)
            except TimeoutError:
                pass
            self._wake.clear()

    def wake(self) -> None:
        """Run the next tick now instead of at the end of the period (desired state changed).

        API handlers call this instead of awaiting tick(): a tick waits on the tick lock and on
        HTTP calls to agents, which would hold the request for seconds. Call it on the event loop
        (from async routes): asyncio.Event is not thread-safe, so sync threadpool routes must not."""
        self._wake.set()

    async def shutdown(self) -> None:
        tasks = list(self._launches.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------ events
    def _emit(self, *a, **kw) -> None:
        try:
            self.notifier.emit(*a, **kw)
        except Exception:  # an event problem must never break reconciliation
            log.exception("emitting event failed")

    def _track_nodes(self, nodes: dict[str, NodeRecord], now: float) -> None:
        """node_offline / node_online, once per liveness transition."""
        for gone in set(self._node_up) - set(nodes):
            self._node_up.pop(gone)  # server removed
        for node_id, n in nodes.items():
            alive = self._alive(n, now)
            prev = self._node_up.get(node_id)
            self._node_up[node_id] = alive
            if prev is None or prev == alive:
                continue  # first sighting: no event, only transitions are news
            if alive:
                self._emit("info", "node_online", f"Server {node_id} is back online", node_id=node_id)
                continue
            models = sorted({r.model for r in self.store.list_replicas(states=set(LIVE_STATES))
                             if node_id == r.placement.head_node
                             or node_id in {a.node_id for a in r.placement.assignments}})
            tail = f"; {len(models)} model(s) affected: {', '.join(models)}" if models else "; no model affected"
            self._emit("warning", "node_offline",
                       f"Server {node_id} went offline (no report for {now - n.last_seen:.0f} s){tail}",
                       node_id=node_id)
        self._track_versions(nodes, now)

    def _track_versions(self, nodes: dict[str, NodeRecord], now: float) -> None:
        """llama_version_mismatch when registered servers run different llama.cpp builds, once per
        change of the set of builds.

        A split replica needs one RPC protocol on its head and every RPC server (ggml-rpc refuses a
        different major version at HELLO, and the head aborts). Builds do not say which protocol
        they speak, so the warning names the builds and the risk. Every registered server's last
        report counts, alive or not, so a server that flaps does not repeat the warning; "unknown"
        builds are left out."""
        by_version: dict[str, list[str]] = {}
        for node_id, n in nodes.items():
            v = n.report.llama_version
            if v and v != "unknown":
                by_version.setdefault(v, []).append(node_id)
        key = tuple(sorted(by_version))
        if key == self._versions_seen:
            return
        self._versions_seen = key
        if len(by_version) > 1:
            parts = "; ".join(f"{v}: {', '.join(sorted(ids))}" for v, ids in sorted(by_version.items()))
            self._emit("warning", "llama_version_mismatch",
                       f"Servers run different llama.cpp builds ({parts}). A model split over servers needs "
                       "the same RPC protocol on all of them; builds of one gpupool release always match. "
                       "If launches of split models fail, upgrade every agent to the same image.")

    # ------------------------------------------------------------------ failure detection
    def _bump_backoff(self, model: str, now: float) -> tuple[int, float]:
        n, _ = self._backoff.get(model, (0, 0.0))
        n += 1
        delay = min(300.0, 5.0 * 2 ** (n - 1))
        self._backoff[model] = (n, now + delay)
        self._save_backoff()
        return n, delay

    def _clear_stable_backoff(self, now: float) -> None:
        """A model whose replica has stayed ready for STABLE_S is healthy again: forget its backoff."""
        cleared = False
        for model in list(self._backoff):
            if any(now - r.updated_at >= self.STABLE_S
                   for r in self.store.list_replicas(model=model, states={"ready"})):
                self._backoff.pop(model, None)
                cleared = True
        if cleared:
            self._save_backoff()

    async def _fail(self, rec: ReplicaRecord, reason: str, nodes: dict[str, NodeRecord], now: float,
                    model_fault: bool = False) -> None:
        """`model_fault`: the model/engine caused it (not a dead node or missing GPU), so a
        replica that dies soon after becoming ready backs the model off instead of reloading every tick."""
        log.warning("replica %s failed: %s", rec.replica_id, reason)
        if model_fault and rec.state == "ready" and now - rec.updated_at < self.STABLE_S:
            n, delay = self._bump_backoff(rec.model, now)
            if n >= 2:
                self._emit("warning", "crash_loop",
                           f"{rec.model} crashed {n} times shortly after start; next attempt in {delay:.0f} s",
                           model=rec.model)
        entry = self._realloc.get(rec.model)
        if entry is None:
            self._realloc[rec.model] = _Realloc(since=now, lost=(rec.replica_id, reason))
        elif not entry.started:
            entry.lost = (rec.replica_id, reason)
        t = self._launches.get(rec.replica_id)
        if t is not None:
            t.cancel()
        self.store.set_replica_state(rec.replica_id, "failed", reason, now=now)
        await self._stop_engines(rec, nodes, now)

    async def _stop_engines(self, rec: ReplicaRecord, nodes: dict[str, NodeRecord], now: float) -> None:
        async def one(node_id: str, engine_id: str) -> None:
            n = nodes.get(node_id)
            if n is None or not self._alive(n, now):
                return  # dead node: nothing reachable to stop
            try:
                await self.client.stop_engine(n.report.agent_url, engine_id)
            except Exception as e:  # best effort; agent GC / next report will show leftovers
                log.warning("stop %s on %s failed: %s", engine_id, node_id, e)

        await stop_head_first(engine_ids(rec), one)

    async def _fail_orphaned_launches(self, nodes: dict[str, NodeRecord], now: float) -> None:
        """Fail pending/launching replicas that no launch task of this process owns.

        A replica record and its launch task are created together (_spawn_launch), so a
        launching record without a task means the coordinator died mid-launch (kill -9, OOM,
        power loss; a clean shutdown cancels the task and marks it failed itself). Seen on a real
        cluster: after `docker kill` the replica stayed "launching" forever and, being counted as
        active, kept its model from ever being launched again. Its engines may be half-started on
        the agents, so _fail stops them; the next tick re-plans the model."""
        for rec in self.store.list_replicas(states={"pending", "launching"}):
            if rec.replica_id not in self._launches:
                await self._fail(rec, "coordinator restarted during launch", nodes, now)

    async def _detect_failures(self, nodes: dict[str, NodeRecord], now: float) -> None:
        for rec in self.store.list_replicas(states={"launching", "ready"}):
            p = rec.placement
            node_ids = {p.head_node} | {a.node_id for a in p.assignments}
            dead = sorted(n for n in node_ids if n not in nodes or not self._alive(nodes[n], now))
            if dead:
                await self._fail(rec, f"node(s) dead: {', '.join(dead)}", nodes, now)
                continue
            gone = sorted(f"{a.node_id}/{a.device_id}" for a in p.assignments
                          if _find_device(nodes[a.node_id].report.devices, a) is None)
            if gone:
                reason = f"GPU {', '.join(gone)} no longer reported by its (live) server"
                self._emit("error", "gpu_missing", f"{reason}; replica {rec.replica_id} of {rec.model} failed",
                           node_id=gone[0].split("/")[0], model=rec.model)
                await self._fail(rec, reason, nodes, now)
                continue
            if rec.state != "ready":
                continue
            for node_id, eid in engine_ids(rec):
                nr = nodes[node_id]
                st = next((e for e in nr.report.engines if e.engine_id == eid), None)
                if st is not None and st.state in ("exited", "failed"):
                    tail = " | ".join(st.log_tail[-5:])
                    self._emit("error", "engine_crashed",
                               f"Engine {eid} on {node_id} {st.state} (exit code {st.exit_code}); "
                               f"last log: {tail or 'n/a'}", node_id=node_id, model=rec.model)
                    await self._fail(rec, f"engine {eid} {st.state} (exit={st.exit_code}) {tail}".strip(), nodes, now,
                                     model_fault=True)
                    break
                if st is None and nr.last_seen > rec.updated_at:
                    self._emit("error", "engine_crashed",
                               f"Engine {eid} vanished from the report of {node_id}", node_id=node_id,
                               model=rec.model)
                    await self._fail(rec, f"engine {eid} missing from {node_id} report", nodes, now,
                                     model_fault=True)
                    break

    async def _check_suspects(self, nodes: dict[str, NodeRecord]) -> None:
        suspects, self._suspect = self._suspect, set()
        for rid in suspects:
            rec = self.store.get_replica(rid)
            if rec is None or rec.state != "ready":
                continue
            head = nodes.get(rec.placement.head_node)
            if head is None:
                continue
            url = f"http://{head.report.host}:{rec.placement.head_port}/health"
            try:
                r = await self._http_client().get(url)
                ok = r.status_code == 200
                why = f"health returned {r.status_code}"
            except httpx.HTTPError as e:
                ok, why = False, f"health unreachable: {type(e).__name__}"
            if not ok:
                await self._fail(rec, f"router reported errors and {why}", nodes, self.clock(),
                                 model_fault=True)

    # ------------------------------------------------------------------ drain
    async def drain(self, replica_id: str) -> None:
        rec = self.store.get_replica(replica_id)
        if rec is None or rec.state in ("draining", "stopped", "failed"):
            return
        t = self._launches.get(replica_id)
        if t is not None:
            t.cancel()  # launch rolls back its engines; the tick then marks it stopped
        self.store.set_replica_state(replica_id, "draining", None, now=self.clock())

    async def _process_drains(self, nodes: dict[str, NodeRecord], now: float) -> None:
        for rec in self.store.list_replicas(states={"draining"}):
            waited = now - rec.updated_at
            if self.outstanding(rec.replica_id) > 0 and waited < self.cfg.drain_timeout_s:
                continue
            await self._stop_engines(rec, nodes, now)
            self.store.set_replica_state(rec.replica_id, "stopped", None, now=now)

    # ------------------------------------------------------------------ desired count
    def _low_free(self, rec: ReplicaRecord, nodes: dict[str, NodeRecord]) -> bool:
        for a in rec.placement.assignments:
            n = nodes.get(a.node_id)
            if n is None:
                continue
            d = _find_device(n.report.devices, a)
            if d is not None and d.free_mb < self.cfg.low_free_mb:
                return True
        return False

    async def _enforce_counts(self, nodes: dict[str, NodeRecord], now: float) -> None:
        specs = {s.name: s for s in self.store.list_models()}
        # replicas of models no longer registered: drain
        for rec in self.store.list_replicas(states=set(ACTIVE_STATES)):
            if rec.model not in specs:
                await self.drain(rec.replica_id)
        # Without an autoscaler the count is the spec's fixed `replicas`, as before.
        wanted = {n: self._wanted(s) for n, s in specs.items()}
        for m in [m for m in self._realloc if m not in specs or wanted[m] == 0]:
            self._realloc.pop(m)  # stopped or unloaded on purpose: nothing left to re-allocate

        def has_active(name: str) -> bool:
            return bool(self.store.list_replicas(model=name, states=set(ACTIVE_STATES)))

        # Scarce VRAM goes to high priority first; among equals every model gets a first replica
        # before any gets a second, then name order keeps ticks deterministic.
        for spec in sorted(specs.values(), key=lambda s: (-s.priority, has_active(s.name), s.name)):
            active = self.store.list_replicas(model=spec.name, states=set(ACTIVE_STATES))
            # Re-read per model: a preemption made earlier in this same tick must already hold.
            if (len(active) < wanted[spec.name]
                    and spec.priority < self._claim_priority(specs, wanted, now)):
                continue  # memory being freed by a preemption belongs to the higher-priority model
            moving = [r for r in active if r.state == "ready" and self._low_free(r, nodes)]
            moving_ids = {r.replica_id for r in moving}
            normal = [r for r in active if r.replica_id not in moving_ids]
            need = wanted[spec.name]
            if (need > 0 and self._move is not None and self._move["model"] == spec.name
                    and any(r.replica_id == self._move["new"] and r.state in ("pending", "launching")
                            for r in active)):
                need += 1  # make-before-break: the old replica serves until the new one is ready

            if len(normal) < need:
                await self._maybe_launch(spec, now, need)
            elif len(normal) > need:
                for r in sorted(normal, key=lambda r: r.created_at, reverse=True)[: len(normal) - need]:
                    await self.drain(r.replica_id)
            elif moving and sum(1 for r in normal if r.state == "ready") >= need:
                for r in moving:
                    log.info("replica %s: device low on free memory, replacement ready; draining", r.replica_id)
                    await self.drain(r.replica_id)

    async def _maybe_launch(self, spec: ModelSpec, now: float, wanted: int | None = None) -> None:
        fails, not_before = self._backoff.get(spec.name, (0, 0.0))
        if now < not_before:
            return
        try:
            placement = await self.plan_for(spec)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if self._nofit.get(spec.name) != msg:
                self._nofit[spec.name] = msg
                if isinstance(e, NoFit):
                    self._realloc.setdefault(spec.name, _Realloc(since=now, lost=None))
                    self._emit("error", "realloc_failed",
                               f"Cannot place {spec.name}: {e}", model=spec.name)
                else:
                    self._emit("error", "launch_failed", f"Cannot plan {spec.name}: {msg}", model=spec.name)
                (log.warning if isinstance(e, NoFit) else log.error)(
                    "cannot place %s: %s", spec.name, msg)
            if isinstance(e, NoFit):
                await self._preempt_for(spec, now, self._wanted(spec) if wanted is None else wanted)
            return
        self._nofit.pop(spec.name, None)
        rec = self._spawn_launch(spec, placement, now)
        entry = self._realloc.get(spec.name)
        if entry is not None and entry.lost is not None and not entry.started:
            entry.started = True
            lost_id, why = entry.lost
            where = ", ".join(f"{a.node_id}/{a.device_id} {a.layers} layers" for a in placement.assignments)
            self._emit("warning", "realloc_started",
                       f"Re-allocating {spec.name}: replica {lost_id} lost ({why}); new placement: {where}",
                       model=spec.name)

    def _spawn_launch(self, spec: ModelSpec, placement: Placement, now: float) -> ReplicaRecord:
        """Record `placement` as a launching replica and start its launch task."""
        rec = ReplicaRecord(replica_id=placement.replica_id, model=spec.name, placement=placement,
                            state="launching", created_at=now, updated_at=now)
        self.store.put_replica(rec)
        self._launches[rec.replica_id] = asyncio.create_task(self._launch(rec, spec), name=f"launch-{rec.replica_id}")
        return rec

    # ------------------------------------------------------------------ launch
    def _model_source(self, spec: ModelSpec, source: str | None = None) -> tuple[str, str]:
        """(cache name, source to send to the agent). `source` overrides spec.source (the draft)."""
        src = source if source is not None else spec.source
        if not src.startswith("coordinator://"):
            try:
                p = Path(src)
                if p.is_absolute():
                    rel = p.resolve().relative_to(self.cfg.models_dir.resolve())
                    src = "coordinator://" + rel.as_posix()
            except ValueError:
                pass  # not inside models_dir: send as-is
        if src.startswith("coordinator://"):
            name = Path(src[len("coordinator://"):]).name
        else:
            name = Path(urlparse(src).path).name
        name = name or spec.name
        return name, src

    async def _launch(self, rec: ReplicaRecord, spec: ModelSpec) -> None:
        rid, p = rec.replica_id, rec.placement
        created: list[tuple[str, str]] = []  # (agent_url, engine_id)
        head_url = ""
        became_ready = False
        try:
            nodes = self._node_map()

            def agent(node_id: str) -> str:
                n = nodes.get(node_id)
                if n is None:
                    raise LaunchError(f"node {node_id} unknown")
                return n.report.agent_url

            head_host = nodes[p.head_node].report.host if p.head_node in nodes else None
            if head_host is None:
                raise LaunchError(f"head node {p.head_node} unknown")
            head_url = agent(p.head_node)

            extra: dict = {"cache_type": spec.kv_cache_type, "spec_type": spec.speculative,
                           "flash_attn": spec.flash_attn, "batch": spec.batch, "ubatch": spec.ubatch,
                           "kv_unified": spec.kv_unified}
            if spec.speculative == "draft":
                if not spec.draft:
                    raise LaunchError("speculative 'draft' without a draft model")
                # The draft runs inside the head's llama-server: an RPC or remote device cannot host it.
                if _draft_device(p) is None:
                    raise LaunchError("the draft model needs a local CUDA device on the head node, but the "
                                      "placement has none")

            # The head's model files are fetched while the RPC engines start: on a cold start the
            # download, not the engines, is the long pole, and the two do not depend on each other.
            models = asyncio.create_task(self._ensure_head_models(spec, head_url))
            try:
                await self._start_rpc_engines(rid, p, agent, head_host, created)
                path, draft_path = await models
            finally:  # always reap: a download that failed meanwhile must not go unretrieved
                if not models.done():
                    models.cancel()
                await asyncio.gather(models, return_exceptions=True)
            if draft_path is not None:
                extra.update(draft_model_path=draft_path, draft_device=_draft_device(p).llama_device,
                             draft_n_max=spec.draft_n_max)
            elif spec.speculative == "mtp":
                if "spec_mtp" in nodes[p.head_node].report.features:
                    extra.update(draft_n_max=spec.draft_n_max)
                else:
                    # The head's llama.cpp cannot draft with MTP layers (older build or agent): serve
                    # without speculation rather than fail the launch over an optimisation.
                    extra["spec_type"] = "none"
                    self._emit("warning", "mtp_unavailable",
                               f"{spec.name}: the head {p.head_node} runs a llama.cpp build without "
                               "--spec-type draft-mtp; serving without speculative decoding. Upgrade "
                               "the agent to enable it.", node_id=p.head_node, model=spec.name)

            head_id = f"{rid}-head"
            created.append((head_url, head_id))
            await self.client.start_engine(head_url, EngineSpec(
                engine_id=head_id, kind="server", port=p.head_port,
                devices=[a.llama_device for a in p.assignments],
                rpc_endpoints=[ep for ep, _ in rpc_servers(p)],
                tensor_split=p.tensor_split, model=spec.name, model_path=path,
                ctx_size=spec.ctx_size, parallel=spec.parallel, **extra))
            await self._wait_health(head_url, head_id, f"http://{head_host}:{p.head_port}/health")

            cur = self.store.get_replica(rid)
            if cur is None or cur.state != "launching":
                raise _Superseded
            self.store.set_replica_state(rid, "ready", None, now=self.clock())
            became_ready = True
            log.info("replica %s ready", rid)
            entry = self._realloc.pop(spec.name, None)
            if entry is not None:
                self._emit("info", "realloc_done",
                           f"{spec.name} is serving again after {self.clock() - entry.since:.0f} s "
                           f"(replica {rid})", model=spec.name)
        except _Superseded:
            await asyncio.shield(self._rollback(created))  # state was already changed by whoever superseded us
        except asyncio.CancelledError:
            await asyncio.shield(self._rollback(created))
            cur = self.store.get_replica(rid)
            if cur is not None and cur.state == "launching":
                self.store.set_replica_state(rid, "failed", "launch cancelled", now=self.clock())
            raise
        except Exception as e:  # not BaseException: KeyboardInterrupt/SystemExit must propagate
            err = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            if head_url:
                try:
                    st = await self.client.get_engine(head_url, f"{rid}-head")
                    if st and st.log_tail:
                        err += " | head log: " + " / ".join(st.log_tail[-8:])
                except Exception:
                    pass
            log.warning("launch of %s failed: %s", rid, err)
            self._emit("error", "launch_failed", f"Launch of {spec.name} (replica {rid}) failed: {err[:500]}",
                       node_id=p.head_node, model=spec.name)
            await asyncio.shield(self._rollback(created))
            cur = self.store.get_replica(rid)
            if cur is not None and cur.state == "launching":
                self.store.set_replica_state(rid, "failed", err[:2000], now=self.clock())
            self._bump_backoff(spec.name, self.clock())
        finally:
            try:
                if became_ready:
                    # Outside the try above on purpose: a cancel (drain) here must not roll back a
                    # replica that is already serving.
                    await self._calibrate(rec, spec, head_url)
            finally:
                self._launches.pop(rid, None)

    async def _start_rpc_engines(self, rid: str, p: Placement, agent: Callable[[str], str],
                                 head_host: str, created: list[tuple[str, str]]) -> None:
        """Start every RPC engine of `p`, then wait until all of them listen. Each one is added to
        `created` before its start call, so a failure anywhere rolls back what may already run."""
        rpc_ids: list[tuple[str, str]] = []
        for endpoint, devs in rpc_servers(p):
            eid = rpc_engine_id(rid, devs)
            url = agent(devs[0].node_id)
            created.append((url, eid))
            await self.client.start_engine(url, EngineSpec(
                engine_id=eid, kind="rpc", port=_port_of(endpoint), devices=[a.device_id for a in devs],
                allowed_peers=[head_host]))  # only the head connects to an RPC engine
            rpc_ids.append((url, eid))
        for url, eid in rpc_ids:
            await self._wait_running(url, eid)

    async def _ensure_head_models(self, spec: ModelSpec, head_url: str) -> tuple[str, str | None]:
        """(model path, draft path or None) on the head node; both files are fetched at once."""
        name, src = self._model_source(spec)
        if spec.speculative != "draft":
            return await self.client.ensure_model(head_url, name, src), None
        dname, dsrc = self._model_source(spec, spec.draft)
        tasks = [asyncio.create_task(self.client.ensure_model(head_url, name, src)),
                 asyncio.create_task(self.client.ensure_model(head_url, dname, dsrc))]
        try:
            return await tasks[0], await tasks[1]
        finally:  # one failed or we were cancelled: do not leave the other request running
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------ speed model
    async def _learn_speed(self, nodes: dict[str, NodeRecord], now: float) -> None:
        """Refine the decode-speed model (eta, seconds per RPC hop) from what replicas measure.

        Only replicas whose measured speed is one plain decode stream count: parallel 1 and no
        speculative decoding (several streams share the bandwidth; drafting multiplies tokens).
        A single-server replica measures eta: tok/s = eta / bandwidth_seconds. A split one, with
        eta known, measures the time its RPC servers add per token. Each is an EMA, clamped."""
        if self.autoscaler is None or now - self._last_speed_sample < self.SPEED_SAMPLE_S:
            return
        self._last_speed_sample = now
        reports = [n.report for n in nodes.values()]
        cuda_default = default_cuda_bw(reports)
        sm = speed_model()
        eta, hop = sm.eta, sm.hop_s
        n_eta = n_hop = 0
        for rec in self.store.list_replicas(states={"ready"}):
            spec = self.store.get_model(rec.model)
            measured = self.autoscaler.measured_tps(rec.replica_id)
            if spec is None or not measured or spec.parallel != 1 or spec.speculative != "none":
                continue
            p = rec.placement
            devs = []
            for a in p.assignments:
                n = nodes.get(a.node_id)
                d = _find_device(n.report.devices, a) if n is not None else None
                if d is None:
                    break
                devs.append((d, a.layers))
            else:
                try:
                    meta = await self.meta_for(spec)
                except Exception:
                    continue
                t_bw = bandwidth_seconds(meta, devs, cuda_default)
                if t_bw <= 0:
                    continue
                n_rpc = len(rpc_servers(p))
                if n_rpc == 0:
                    eta += self.SPEED_ALPHA * (t_bw * measured - eta)
                    eta = min(max(eta, ETA_RANGE[0]), ETA_RANGE[1])  # hop samples below use it
                    n_eta += 1
                    continue
                rest = 1.0 / measured - t_bw / eta
                if p.assignments and p.assignments[-1].node_id != p.head_node:
                    rest -= logits_seconds(meta)
                hop += self.SPEED_ALPHA * (rest / n_rpc - hop)
                n_hop += 1
        if not (n_eta or n_hop):
            return
        set_speed_model(eta, hop)
        sm = speed_model()
        self._save("speed_model", {"eta": sm.eta, "hop_s": sm.hop_s})
        log.info("speed model: eta %.3f, %.2f ms per RPC hop (%d single-server, %d split samples)",
                 sm.eta, sm.hop_s * 1000, n_eta, n_hop)

    # ------------------------------------------------------------------ calibration
    @staticmethod
    def calibration_sample(placement: Placement, memory: dict) -> tuple[float, float] | None:
        """(measured MB, estimated MB) of a ready replica, or None when the data does not cover it.

        Measured: the head's per-device buffers summed over the placement's devices (the draft model
        sits in the head device's buffers). Estimated: est_mb, unscaled by the factor the placement
        was planned with, minus the per-device runtime context, which llama.cpp does not report as
        buffers (a draft brings its own context: one more)."""
        devices = memory.get("devices") if isinstance(memory, dict) else None
        if not isinstance(devices, dict) or not devices:
            return None
        measured = estimated = 0.0
        f = placement.mem_factor if placement.mem_factor > 0 else 1.0
        for a in placement.assignments:
            d = devices.get(a.llama_device)
            if not isinstance(d, dict) or not d.get("total_mb"):
                return None  # partial data would bias the ratio low: skip the sample
            measured += float(d["total_mb"])
            estimated += a.est_mb / f - CONTEXT_MB["cuda"]
        if placement.draft_est_mb:
            estimated -= CONTEXT_MB["cuda"]
        if measured <= 0 or estimated <= 0:
            return None
        return measured, estimated

    async def _calibrate(self, rec: ReplicaRecord, spec: ModelSpec, head_url: str) -> None:
        """Compare what the head engine really loaded with the estimate and fold it into the model's
        memory factor. Never fails the launch: this is bookkeeping about a replica already serving."""
        try:
            memory = await self.client.engine_memory(head_url, f"{rec.replica_id}-head")
            sample = self.calibration_sample(rec.placement, memory) if memory else None
            if sample is None:
                return  # old agent, logs without -lv 4, or devices missing: leave the factor alone
            measured, estimated = sample
            ratio = measured / estimated
            old = self.store.get_calibration(spec.name)
            raw = ratio if old is None else self.CAL_ALPHA * ratio + (1 - self.CAL_ALPHA) * old["factor"]
            self.store.put_calibration(spec.name, raw, (old["samples"] if old else 0) + 1, now=self.clock())
            before, after = planning_factor(old["factor"] if old else None), planning_factor(raw)
            if abs(after - before) / before > self.CAL_EVENT_DELTA:
                msg = (f"{spec.name}: measured {measured:.0f} MB vs estimated {estimated:.0f} MB, "
                       f"planning factor {after:.2f}")
                log.info(msg)
                self._emit("info", "calibrated", msg, node_id=rec.placement.head_node, model=spec.name)
        except Exception:
            log.exception("memory calibration of %s failed", rec.replica_id)

    async def _rollback(self, created: list[tuple[str, str]]) -> None:
        """Stop every engine created so far: a half-launched replica would pin VRAM on shared GPUs."""
        async def one(url: str, eid: str) -> None:
            try:
                await self.client.stop_engine(url, eid)
            except Exception as e:
                log.warning("rollback: stop %s failed: %s", eid, e)

        await stop_head_first(created, one)  # the head may already be up: same race as a stop

    async def _wait_running(self, url: str, eid: str) -> None:
        deadline = time.monotonic() + self.cfg.launch_timeout_s
        while True:
            st = await self.client.get_engine(url, eid)
            if st is not None and st.state == "running":
                return
            if st is not None and st.state in ("exited", "failed"):
                raise LaunchError(f"engine {eid} {st.state}: {' / '.join(st.log_tail[-8:])}")
            if time.monotonic() > deadline:
                raise LaunchError(f"engine {eid} not running after {self.cfg.launch_timeout_s}s")
            await asyncio.sleep(self.poll_s)

    async def _wait_health(self, agent_url: str, head_id: str, health_url: str) -> None:
        deadline = time.monotonic() + self.cfg.launch_timeout_s
        while True:
            try:
                r = await self._http_client().get(health_url)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass  # llama-server not listening yet
            st = await self.client.get_engine(agent_url, head_id)
            if st is not None and st.state in ("exited", "failed"):
                raise LaunchError(f"head engine {st.state} (exit={st.exit_code})")
            if time.monotonic() > deadline:
                raise LaunchError(f"{health_url} not healthy after {self.cfg.launch_timeout_s}s")
            await asyncio.sleep(self.poll_s)
