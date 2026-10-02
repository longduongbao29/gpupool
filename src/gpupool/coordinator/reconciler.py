"""Reconcile loop: failure detection, desired-count enforcement, launch, drain."""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
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
    Placement,
    ReplicaRecord,
)
from gpupool.common.net import internal_client
from gpupool.coordinator.agent_client import AgentClient
from gpupool.coordinator.events import Notifier
from gpupool.coordinator.store import NodeRecord, Store, gpu_key
from gpupool.scheduler.placement import NoFit, plan

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


def engine_ids(rec: ReplicaRecord) -> list[tuple[str, str]]:
    """(node_id, engine_id) for every engine this replica owns."""
    p = rec.placement
    out = [(p.head_node, f"{rec.replica_id}-head")]
    for a in p.assignments:
        if a.rpc_endpoint:
            out.append((a.node_id, f"{rec.replica_id}-rpc-{a.device_id}"))
    return out


class Reconciler:
    READY_REPORT_GRACE_S = 5.0
    STABLE_S = 300.0  # a replica ready this long counts as healthy: crashes before it feed the backoff
    KEEP_TERMINAL_PER_MODEL = 10  # stopped/failed history rows kept per model

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
        self.notifier = notifier or Notifier(store, getattr(cfg, "webhook_url", ""), clock=clock)
        self.poll_s = 0.5  # engine/health poll interval during launch
        self.planner: Callable | None = None  # tests inject; default is scheduler.placement.plan
        self._http: httpx.AsyncClient | None = None
        self._launches: dict[str, asyncio.Task] = {}
        self._suspect: set[str] = set()
        self._tick_lock = asyncio.Lock()
        self._nofit: dict[str, str] = {}
        self._backoff: dict[str, tuple[int, float]] = {}  # model -> (consecutive failures, retry not before)
        self._node_up: dict[str, bool] = {}  # node_id -> last observed liveness (for transition events)
        self._realloc: dict[str, _Realloc] = {}
        self._wake = asyncio.Event()

    # ------------------------------------------------------------------ helpers
    def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = internal_client(timeout=httpx.Timeout(5.0))
        return self._http

    def _alive(self, rec: NodeRecord, now: float) -> bool:
        return rec.alive(now, self.cfg.heartbeat_timeout_s)

    def _node_map(self) -> dict[str, NodeRecord]:
        # Only registered servers count: a stray heartbeat must not receive placements.
        registered = {s.node_id for s in self.store.list_servers()}
        return {n.report.node_id: n for n in self.store.list_nodes() if n.report.node_id in registered}

    def _planner(self) -> Callable:
        if self.planner is not None:
            return self.planner
        return plan

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
            out.append(rep)
        return out

    async def plan_for(self, spec: ModelSpec, replica_id: str | None = None) -> Placement:
        """Plan one replica. No side effects (ports are only 'handed out' within this call)."""
        now = self.clock()
        nodes = self._node_map()
        reports = self._reserved_nodes(nodes, now)
        if spec.pin_devices:
            pins = set(spec.pin_devices)
            for rep in reports:  # reports are private copies, safe to edit
                for d in rep.devices:
                    if f"{rep.node_id}/{d.device_id}" not in pins:
                        d.usable_mb = 0
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
        return self._planner()(meta, spec, reports, rid, port_alloc)

    @staticmethod
    def _new_replica_id(model: str) -> str:
        return f"{re.sub(r'[^A-Za-z0-9_.-]', '-', model)}-{uuid.uuid4().hex[:6]}"

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
            await self._detect_failures(nodes, now)
            await self._check_suspects(nodes)
            await self._process_drains(nodes, now)
            self._clear_stable_backoff(now)
            await self._enforce_counts(nodes, now)
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

    # ------------------------------------------------------------------ failure detection
    def _bump_backoff(self, model: str, now: float) -> tuple[int, float]:
        n, _ = self._backoff.get(model, (0, 0.0))
        n += 1
        delay = min(300.0, 5.0 * 2 ** (n - 1))
        self._backoff[model] = (n, now + delay)
        return n, delay

    def _clear_stable_backoff(self, now: float) -> None:
        """A model whose replica has stayed ready for STABLE_S is healthy again: forget its backoff."""
        for model in list(self._backoff):
            if any(now - r.updated_at >= self.STABLE_S
                   for r in self.store.list_replicas(model=model, states={"ready"})):
                self._backoff.pop(model, None)

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

        await asyncio.gather(*(one(n, e) for n, e in engine_ids(rec)))

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
        for m in [m for m in self._realloc if m not in specs or specs[m].replicas == 0]:
            self._realloc.pop(m)  # stopped on purpose: nothing left to re-allocate

        for spec in specs.values():
            active = self.store.list_replicas(model=spec.name, states=set(ACTIVE_STATES))
            moving = [r for r in active if r.state == "ready" and self._low_free(r, nodes)]
            moving_ids = {r.replica_id for r in moving}
            normal = [r for r in active if r.replica_id not in moving_ids]
            need = spec.replicas

            if len(normal) < need:
                await self._maybe_launch(spec, now)
            elif len(normal) > need:
                for r in sorted(normal, key=lambda r: r.created_at, reverse=True)[: len(normal) - need]:
                    await self.drain(r.replica_id)
            elif moving and sum(1 for r in normal if r.state == "ready") >= need:
                for r in moving:
                    log.info("replica %s: device low on free memory, replacement ready; draining", r.replica_id)
                    await self.drain(r.replica_id)

    async def _maybe_launch(self, spec: ModelSpec, now: float) -> None:
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
            return
        self._nofit.pop(spec.name, None)
        rec = ReplicaRecord(replica_id=placement.replica_id, model=spec.name, placement=placement,
                            state="launching", created_at=now, updated_at=now)
        self.store.put_replica(rec)
        entry = self._realloc.get(spec.name)
        if entry is not None and entry.lost is not None and not entry.started:
            entry.started = True
            lost_id, why = entry.lost
            where = ", ".join(f"{a.node_id}/{a.device_id} {a.layers} layers" for a in placement.assignments)
            self._emit("warning", "realloc_started",
                       f"Re-allocating {spec.name}: replica {lost_id} lost ({why}); new placement: {where}",
                       model=spec.name)
        self._launches[rec.replica_id] = asyncio.create_task(self._launch(rec, spec), name=f"launch-{rec.replica_id}")

    # ------------------------------------------------------------------ launch
    def _model_source(self, spec: ModelSpec) -> tuple[str, str]:
        """(cache name, source to send to the agent)."""
        src = spec.source
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

            rpc_ids: list[tuple[str, str]] = []
            for a in p.assignments:
                if not a.rpc_endpoint:
                    continue
                eid = f"{rid}-rpc-{a.device_id}"
                url = agent(a.node_id)
                created.append((url, eid))
                await self.client.start_engine(url, EngineSpec(
                    engine_id=eid, kind="rpc", port=_port_of(a.rpc_endpoint), devices=[a.device_id]))
                rpc_ids.append((url, eid))
            for url, eid in rpc_ids:
                await self._wait_running(url, eid)

            name, src = self._model_source(spec)
            path = await self.client.ensure_model(head_url, name, src)

            head_id = f"{rid}-head"
            created.append((head_url, head_id))
            await self.client.start_engine(head_url, EngineSpec(
                engine_id=head_id, kind="server", port=p.head_port,
                devices=[a.llama_device for a in p.assignments],
                rpc_endpoints=[a.rpc_endpoint for a in p.assignments if a.rpc_endpoint],
                tensor_split=p.tensor_split, model=spec.name, model_path=path,
                ctx_size=spec.ctx_size, parallel=spec.parallel))
            await self._wait_health(head_url, head_id, f"http://{head_host}:{p.head_port}/health")

            cur = self.store.get_replica(rid)
            if cur is None or cur.state != "launching":
                raise _Superseded
            self.store.set_replica_state(rid, "ready", None, now=self.clock())
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
            self._launches.pop(rid, None)

    async def _rollback(self, created: list[tuple[str, str]]) -> None:
        """Stop every engine created so far: a half-launched replica would pin VRAM on shared GPUs."""
        async def one(url: str, eid: str) -> None:
            try:
                await self.client.stop_engine(url, eid)
            except Exception as e:
                log.warning("rollback: stop %s failed: %s", eid, e)

        await asyncio.gather(*(one(u, e) for u, e in created))

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
