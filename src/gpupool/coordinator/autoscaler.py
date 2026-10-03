"""Autoscaler: how many replicas each model should have right now (design section 4.5).

The reconciler asks `desired()` instead of reading `spec.replicas`, the router calls
`note_request()` / `can_cold_start()`, and the API shows `view()`.

State lives in memory and is written through to the store (control_state, key "autoscaler:<model>")
whenever the desired count changes, so a restart keeps an unloaded model unloaded and a scaled-up
model scaled up. Persisted: desired count, last request time, last decision. Dropped on purpose: the
up/down "since when" timers (a restart only delays a scaling step by up_after_s/down_after_s).
last_request is also refreshed on a slow cadence (not per request) so the idle timer stays roughly
right. Times are wall-clock (clock()), so they remain meaningful across a restart. A model with no
saved state starts at its floor (max(min_replicas, 1)).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import AutoscalePolicy, ModelSpec, ReplicaRecord
from gpupool.common.net import internal_client
from gpupool.coordinator.events import Notifier
from gpupool.coordinator.store import Store

log = logging.getLogger("gpupool.autoscaler")

_PROCESSING = "llamacpp:requests_processing"
_DEFERRED = "llamacpp:requests_deferred"
_TPS = "llamacpp:predicted_tokens_seconds"
_WANTED = {_PROCESSING, _DEFERRED, _TPS}


def bounds(spec: ModelSpec) -> tuple[int, int]:
    """(min, max) replicas; None fields fall back to `replicas` (a fixed count)."""
    lo = spec.min_replicas if spec.min_replicas is not None else spec.replicas
    hi = spec.max_replicas if spec.max_replicas is not None else spec.replicas
    return lo, max(lo, hi)


def parse_metrics(text: str) -> dict[str, float]:
    """Pick the llama.cpp gauges we use out of Prometheus text; junk lines are skipped."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        if name not in _WANTED:
            continue
        try:
            out[name] = float(parts[1])
        except ValueError:
            continue
    return out


@dataclass
class _Scrape:
    ts: float
    processing: float | None = None
    deferred: float | None = None
    tps: float | None = None
    ok: bool = True


@dataclass
class _State:
    desired: int
    last_request: float
    up_since: float | None = None
    down_since: float | None = None
    last_decision: dict | None = None
    saved_at: float = float("-inf")  # last time last_request was written to the store (not persisted)


class Autoscaler:
    # A scrape older than this many poll intervals is not trusted; the router's count is used.
    FRESH_POLLS = 3.0
    SCRAPE_TIMEOUT_S = 2.0
    # scale down only when busy is below target * this factor (hysteresis against flapping)
    DOWN_FACTOR = 0.5
    SAVE_IDLE_S = 60.0  # how often a request refreshes the saved last_request

    def __init__(self, store: Store, cfg: CoordinatorConfig, outstanding: Callable[[str], int],
                 notifier: Notifier, clock: Callable[[], float] = time.time,
                 http: httpx.AsyncClient | None = None, wake: Callable[[], None] | None = None):
        self.store = store
        self.cfg = cfg
        self.outstanding = outstanding
        self.notifier = notifier
        self.clock = clock
        self._http = http
        self._own_http = http is None
        self.wake = wake
        self._state: dict[str, _State] = {}
        self._scrapes: dict[str, _Scrape] = {}
        self._clean: set[str] = set()  # stopped models whose saved state is known to be deleted
        # Liveness rule, set by the app to Reconciler.node_alive; None = pure last_seen rule.
        self.node_alive: Callable[[object, float], bool] | None = None

    # -- helpers ---------------------------------------------------------------------------

    def _emit(self, kind: str, message: str, model: str) -> None:
        try:
            self.notifier.emit("info", kind, message, model=model)
        except Exception:
            log.exception("emitting %s for %s failed", kind, model)

    @staticmethod
    def _floor(spec: ModelSpec) -> int:
        return max(bounds(spec)[0], 1)

    @staticmethod
    def _policy(spec: ModelSpec) -> AutoscalePolicy:
        return spec.autoscale or AutoscalePolicy()

    def _state_for(self, spec: ModelSpec, now: float) -> _State:
        st = self._state.get(spec.name)
        if st is None:
            st = self._state[spec.name] = self._load(spec, now)
            self._clean.discard(spec.name)
        return st

    def _key(self, model: str) -> str:
        return f"autoscaler:{model}"

    def _load(self, spec: ModelSpec, now: float) -> _State:
        try:
            saved = self.store.get_state(self._key(spec.name))
            if saved is not None:
                dec = saved.get("last_decision")
                return _State(desired=int(saved["desired"]), last_request=float(saved["last_request"]),
                              last_decision=dec if isinstance(dec, dict) else None, saved_at=now)
        except Exception:
            log.exception("loading autoscaler state of %s failed", spec.name)
        # the idle timer starts when the model starts, not at coordinator boot
        return _State(desired=self._floor(spec), last_request=now)

    def _save(self, model: str, st: _State, now: float) -> None:
        """Write-through; a store problem must not break scaling (the state just stays in memory)."""
        try:
            self.store.put_state(self._key(model), {
                "desired": st.desired, "last_request": st.last_request, "last_decision": st.last_decision})
            st.saved_at = now
        except Exception:
            log.exception("saving autoscaler state of %s failed", model)

    def _forget(self, model: str) -> None:
        """The model is stopped or deleted: drop its state, in memory and in the store."""
        had = self._state.pop(model, None) is not None
        if had or model not in self._clean:
            try:
                self.store.delete_state(self._key(model))
                self._clean.add(model)
            except Exception:
                log.exception("deleting autoscaler state of %s failed", model)

    def _clamped(self, spec: ModelSpec, desired: int) -> int:
        lo, hi = bounds(spec)
        if desired == 0 and lo == 0:
            return 0  # unloaded by the idle rule; stays until a request wakes it
        return min(max(desired, self._floor(spec)), hi)

    def _clamp(self, spec: ModelSpec, st: _State) -> None:
        st.desired = self._clamped(spec, st.desired)

    @staticmethod
    def _decide(st: _State, now: float, action: str, reason: str) -> None:
        st.last_decision = {"ts": now, "action": action, "reason": reason}
        st.up_since = st.down_since = None

    def _replica_busy(self, rec: ReplicaRecord, parallel: int, now: float) -> tuple[float, bool]:
        """(busy 0..1, queued) of one ready replica; scrape when fresh, else the router's count."""
        parallel = max(parallel, 1)
        sc = self._scrapes.get(rec.replica_id)
        if (sc is not None and sc.ok and sc.processing is not None
                and now - sc.ts <= self.FRESH_POLLS * self.cfg.poll_s):
            return min(1.0, sc.processing / parallel), (sc.deferred or 0) > 0
        out = self.outstanding(rec.replica_id)
        return min(1.0, out / parallel), out > parallel

    def _model_busy(self, spec: ModelSpec, ready: list[ReplicaRecord], now: float) -> tuple[float | None, bool]:
        if not ready:
            return None, False
        vals = [self._replica_busy(r, spec.parallel, now) for r in ready]
        return sum(b for b, _ in vals) / len(vals), any(q for _, q in vals)

    # -- decisions -------------------------------------------------------------------------

    def desired(self, spec: ModelSpec) -> int:
        """Replicas the reconciler should keep now: 0 when stopped (replicas == 0) or unloaded."""
        if spec.replicas == 0:
            self._forget(spec.name)
            return 0
        st = self._state_for(spec, self.clock())
        self._clamp(spec, st)
        return st.desired

    def peek(self, spec: ModelSpec) -> int:
        """What desired() would return, without creating or changing any state. For the simulator,
        which asks about hypothetical specs: a model with no state yet starts at its floor."""
        if spec.replicas == 0:
            return 0
        st = self._state.get(spec.name)
        return self._clamped(spec, st.desired if st is not None else self._floor(spec))

    def can_cold_start(self, model: str) -> bool:
        """The model is started and currently unloaded/zero, so a request may load it."""
        spec = self.store.get_model(model)
        return spec is not None and spec.replicas > 0 and bounds(spec)[0] == 0

    def note_request(self, model: str) -> bool:
        """Router saw a request. Returns True when this triggered a cold start (and woke the loop)."""
        try:
            spec = self.store.get_model(model)
            if spec is None or spec.replicas == 0:
                return False
            now = self.clock()
            st = self._state_for(spec, now)
            st.last_request = now
            if bounds(spec)[0] == 0 and st.desired == 0:
                st.desired = 1
                self._decide(st, now, "cold_start", f"request for unloaded model {model}")
                self._save(model, st, now)
                self._emit("cold_start", f"loading {model} for an incoming request", model)
                if self.wake is not None:
                    try:
                        self.wake()
                    except Exception:
                        log.exception("wake failed")
                return True
            if now - st.saved_at >= self.SAVE_IDLE_S:
                self._save(model, st, now)  # slow cadence: requests are frequent, restarts are not
        except Exception:
            log.exception("note_request(%s) failed", model)
        return False

    def evaluate(self) -> None:
        """Update each model's desired count from the latest busyness."""
        now = self.clock()
        specs = {s.name: s for s in self.store.list_models()}
        for name in [n for n in self._state if n not in specs]:
            self._forget(name)  # model deleted
        for spec in specs.values():
            try:
                self._evaluate_model(spec, now)
            except Exception:
                log.exception("autoscaler evaluate(%s) failed", spec.name)

    def _evaluate_model(self, spec: ModelSpec, now: float) -> None:
        if spec.replicas == 0:
            self._forget(spec.name)
            return
        lo, hi = bounds(spec)
        if hi == lo and spec.idle_unload_s is None:
            return  # fixed model: desired() alone answers
        st = self._state_for(spec, now)
        self._clamp(spec, st)
        if st.desired == 0:
            st.up_since = st.down_since = None
            return  # unloaded: only a request brings it back
        pol = self._policy(spec)
        reps = self.store.list_replicas(spec.name)
        ready = [r for r in reps if r.state == "ready"]
        avg, queued = self._model_busy(spec, ready, now)

        # idle unload first: nothing else matters when nobody uses the model
        if (lo == 0 and spec.idle_unload_s is not None and now - st.last_request >= spec.idle_unload_s
                and not any(self.outstanding(r.replica_id) > 0 for r in reps
                            if r.state in ("ready", "draining"))):
            st.desired = 0
            reason = f"no requests for {spec.idle_unload_s:g} s"
            self._decide(st, now, "unloaded_idle", reason)
            self._save(spec.name, st, now)
            self._emit("unloaded_idle", f"{spec.name} unloaded: {reason}", spec.name)
            return

        if hi == lo:
            return
        up = queued or (avg is not None and avg > pol.target_busy)
        down = (not queued) and avg is not None and avg < pol.target_busy * self.DOWN_FACTOR
        # "since when" is kept while the condition holds and reset the moment it breaks
        st.up_since = (st.up_since if st.up_since is not None else now) if up else None
        st.down_since = (st.down_since if st.down_since is not None else now) if down else None

        waiting = any(r.state in ("pending", "launching") for r in reps)  # let the last one finish
        if (st.up_since is not None and now - st.up_since >= pol.up_after_s
                and not waiting and st.desired < hi):
            held = now - st.up_since
            if queued:
                reason = f"requests queueing for {held:.0f} s"
            else:
                reason = f"busy {(avg or 0) * 100:.0f}% > {pol.target_busy * 100:.0f}% for {held:.0f} s"
            st.desired += 1
            self._decide(st, now, "scaled_up", reason)
            self._save(spec.name, st, now)
            self._emit("scaled_up", f"{spec.name} scaled up to {st.desired}: {reason}", spec.name)
        elif (st.down_since is not None and now - st.down_since >= pol.down_after_s
                and st.desired > self._floor(spec)):
            held = now - st.down_since
            reason = (f"busy {(avg or 0) * 100:.0f}% < {pol.target_busy * self.DOWN_FACTOR * 100:.0f}% "
                      f"for {held:.0f} s")
            st.desired -= 1
            self._decide(st, now, "scaled_down", reason)
            self._save(spec.name, st, now)
            self._emit("scaled_down", f"{spec.name} scaled down to {st.desired}: {reason}", spec.name)

    # -- metrics ---------------------------------------------------------------------------

    async def _scrape_one(self, rec: ReplicaRecord, host: str) -> None:
        url = f"http://{host}:{rec.placement.head_port}/metrics"
        now = self.clock()
        try:
            if self._http is None:
                self._http = internal_client()
            r = await self._http.get(url, timeout=self.SCRAPE_TIMEOUT_S)
            r.raise_for_status()
            m = parse_metrics(r.text)
            self._scrapes[rec.replica_id] = _Scrape(
                ts=now, processing=m.get(_PROCESSING), deferred=m.get(_DEFERRED), tps=m.get(_TPS),
                ok=_PROCESSING in m)  # a 200 without our gauges is as useless as a failure
        except asyncio.CancelledError:
            raise
        except Exception:
            # keep the last values (shown in the UI) but flag them; they go stale on their own
            old = self._scrapes.get(rec.replica_id)
            if old is None:
                self._scrapes[rec.replica_id] = _Scrape(ts=float("-inf"), ok=False)
            else:
                old.ok = False

    def _node_alive(self, n, now: float) -> bool:
        if self.node_alive is not None:
            return self.node_alive(n, now)
        return n.alive(now, self.cfg.heartbeat_timeout_s)

    async def scrape_once(self) -> None:
        """Read llama-server /metrics of every ready head."""
        try:
            now = self.clock()
            nodes = {n.report.node_id: n for n in self.store.list_nodes()}
            ready = self.store.list_replicas(states={"ready"})
            keep = {r.replica_id for r in ready}
            self._scrapes = {k: v for k, v in self._scrapes.items() if k in keep}
            jobs = []
            for rec in ready:
                n = nodes.get(rec.placement.head_node)
                if n is None or not self._node_alive(n, now):
                    continue
                jobs.append(self._scrape_one(rec, n.report.host))
            if jobs:
                await asyncio.gather(*jobs)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("scrape_once failed")

    async def run(self) -> None:
        """scrape_once + evaluate every cfg.poll_s, forever."""
        while True:
            try:
                await self.scrape_once()
                self.evaluate()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("autoscaler tick failed")
            await asyncio.sleep(self.cfg.poll_s)

    # -- view ------------------------------------------------------------------------------

    def view(self, model: str) -> dict:
        """Payload of GET /api/models/{name}/scaling.

        state: "stopped" (replicas == 0 or unknown model), "fixed" (min == max, no idle unload),
        "unloaded" (desired 0), "scaling_up" (desired > ready + launching), "scaling_down"
        (desired < ready + launching), else "steady".
        """
        now = self.clock()
        spec = self.store.get_model(model)
        if spec is None:
            return {"model": model, "min": 0, "max": 0, "desired": 0, "ready": 0, "launching": 0,
                    "avg_busy": None, "queued": False, "idle_s": None, "state": "stopped",
                    "last_decision": None, "replicas": []}
        lo, hi = bounds(spec)
        reps = self.store.list_replicas(model)
        ready = [r for r in reps if r.state == "ready"]
        launching = [r for r in reps if r.state in ("pending", "launching")]
        st = self._state.get(model)
        avg, queued = self._model_busy(spec, ready, now)
        if spec.replicas == 0:
            desired, state = 0, "stopped"
        else:
            desired = st.desired if st is not None else self._floor(spec)
            alive = len(ready) + len(launching)
            if hi == lo and spec.idle_unload_s is None:
                state = "fixed"
            elif desired == 0:
                state = "unloaded"
            elif desired > alive:
                state = "scaling_up"
            elif desired < alive:
                state = "scaling_down"
            else:
                state = "steady"
        rows = []
        for r in ready:
            sc = self._scrapes.get(r.replica_id)
            busy, _ = self._replica_busy(r, spec.parallel, now)
            rows.append({
                "replica_id": r.replica_id, "busy": busy,
                "requests_processing": sc.processing if sc else None,
                "requests_deferred": sc.deferred if sc else None,
                "measured_decode_tps": sc.tps if sc else None,
                "est_decode_tps": r.placement.est_decode_tps,
                "metrics_ok": bool(sc and sc.ok),
            })
        return {
            "model": model, "min": lo, "max": hi, "desired": desired,
            "ready": len(ready), "launching": len(launching), "avg_busy": avg, "queued": queued,
            "idle_s": (now - st.last_request) if st is not None else None,
            "state": state, "last_decision": st.last_decision if st is not None else None,
            "replicas": rows,
        }

    async def aclose(self) -> None:
        if self._http is not None and self._own_http:
            await self._http.aclose()
            self._http = None
