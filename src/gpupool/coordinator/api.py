"""JSON API behind the web UI: state, servers, GPU switches, models, events. All under /api."""
from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, params
from pydantic import BaseModel, Field, ValidationError

from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import (
    ACTIVE_STATES, LIVE_STATES, AutoscalePolicy, KvCacheType, ModelSpec, SpecMode, Spread,
)
from gpupool.coordinator.agent_client import AgentError
from gpupool.coordinator.autoscaler import bounds
from gpupool.coordinator.store import ServerRecord, gpu_key
from gpupool.scheduler.estimate import total_need_mb
from gpupool.scheduler.placement import NoFit

log = logging.getLogger("gpupool.api")

NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
PIN_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
COORD_PREFIX = "coordinator://"
# llama.cpp accepts a draft whose vocabulary differs from the target's by up to this many tokens.
MAX_VOCAB_DIFF = 128


class AddServerBody(BaseModel):
    agent_url: str


class GpuBody(BaseModel):
    enabled: bool


class ModelBody(BaseModel):
    file: str
    ctx_size: int = Field(default=4096, ge=1)
    parallel: int = Field(default=1, ge=1)
    pin_devices: list[str] = Field(default_factory=list)
    priority: int | None = Field(default=None, ge=0, le=100)  # None: keep the stored value, else 50
    spread: Spread | None = None  # None: keep the stored value, else "gpu"
    # Autoscaling: None keeps the stored value (else unset = fixed count).
    min_replicas: int | None = Field(default=None, ge=0)
    max_replicas: int | None = Field(default=None, ge=1)
    autoscale: AutoscalePolicy | None = None
    idle_unload_s: float | None = Field(default=None, gt=0)
    preemptible: bool | None = None  # None: keep the stored value, else True
    # KV cache / speculative decoding: None keeps the stored value (else f16 / none / no draft / 4).
    kv_cache_type: KvCacheType | None = None
    speculative: SpecMode | None = None
    draft_file: str | None = None  # library file of the draft model
    draft_n_max: int | None = Field(default=None, ge=1, le=16)


class RecommendBody(BaseModel):
    file: str
    ctx_size: int = Field(default=4096, ge=1)
    parallel: int = Field(default=1, ge=1)
    priority: int = Field(default=50, ge=0, le=100)
    spread: Spread = "gpu"
    pin_devices: list[str] = Field(default_factory=list)
    limit: int = Field(default=3, ge=1, le=10)
    kv_cache_type: KvCacheType = "f16"
    speculative: SpecMode = "none"
    draft_file: str | None = None
    draft_n_max: int = Field(default=4, ge=1, le=16)


class SimFields(BaseModel):
    """Optional ModelSpec fields a simulation may override; absent means unchanged."""

    replicas: int | None = Field(default=None, ge=0)
    min_replicas: int | None = Field(default=None, ge=0)
    max_replicas: int | None = Field(default=None, ge=1)
    priority: int | None = Field(default=None, ge=0, le=100)
    preemptible: bool | None = None
    ctx_size: int | None = Field(default=None, ge=1)
    parallel: int | None = Field(default=None, ge=1)
    spread: Spread | None = None
    pin_devices: list[str] | None = None
    kv_cache_type: KvCacheType | None = None
    speculative: SpecMode | None = None
    draft_file: str | None = None
    draft_n_max: int | None = Field(default=None, ge=1, le=16)

    def fields(self) -> dict:
        out = self.model_dump(exclude_none=True, include=set(SimFields.model_fields))
        file = out.pop("draft_file", None)  # ModelSpec stores the draft as a source string
        if file is not None:
            out["draft"] = COORD_PREFIX + file
        return out


class SimChange(SimFields):
    model: str


class SimAdd(SimFields):
    name: str
    file: str


class SimulateBody(BaseModel):
    changes: list[SimChange] = Field(default_factory=list)
    add: list[SimAdd] = Field(default_factory=list)


class StartBody(BaseModel):
    replicas: int = Field(default=1, ge=1)


class RebalanceBody(BaseModel):
    dry_run: bool = True


class ReadBody(BaseModel):
    up_to_id: int


def normalize_agent_url(raw: str) -> str:
    url = raw.strip().rstrip("/")
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise HTTPException(400, "agent_url must look like http://host:port")
    return url


def spec_or_404(store, name: str) -> ModelSpec:
    spec = store.get_model(name)
    if spec is None:
        raise HTTPException(404, f"unknown model {name}")
    return spec


def plan_http_error(e: Exception) -> HTTPException:
    """Map a planning failure to an HTTP error: no capacity is a conflict, a missing model file
    is not found, anything else a bad request."""
    code = 409 if isinstance(e, NoFit) else 404 if isinstance(e, FileNotFoundError) else 400
    return HTTPException(code, f"{type(e).__name__}: {e}")


async def drain_and_delete_model(store, reconciler, name: str) -> None:
    """Drain the model's replicas, then drop its spec. Shared by /api and /admin so they cannot drift."""
    spec_or_404(store, name)
    for r in store.list_replicas(model=name, states=set(ACTIVE_STATES)):
        await reconciler.drain(r.replica_id)
    store.delete_model(name)


def make_api_router(*, store, reconciler, poller, balancer, library, cfg: CoordinatorConfig,
                    admin_dep, notifier=None, autoscaler=None) -> APIRouter:
    """`admin_dep` may be a callable or an already-built Depends(...)."""
    dep = admin_dep if isinstance(admin_dep, params.Depends) else Depends(admin_dep)
    router = APIRouter(prefix="/api", dependencies=[dep])
    notifier = notifier or reconciler.notifier

    def now() -> float:
        return reconciler.clock()

    def emit(*a, **kw) -> None:
        try:
            notifier.emit(*a, **kw)
        except Exception:
            log.exception("emitting event failed")

    # ------------------------------------------------------------------ state helpers
    def server_entry(s: ServerRecord, nodes: dict, flags: dict) -> dict:
        n = nodes.get(s.node_id)
        devices = n.report.devices if n else []
        return {
            "node_id": s.node_id, "agent_url": s.agent_url, "added_at": s.added_at,
            "alive": bool(n and reconciler.node_alive(n, now())),
            "last_seen": n.last_seen if n else 0.0,
            "report": n.report.model_dump(mode="json") if n else None,
            "gpu_enabled": {d.device_id: flags.get((s.node_id, gpu_key(d)), True) for d in devices},
        }

    def model_entry(spec: ModelSpec, reps: list) -> dict:
        """`reps`: this model's replicas, oldest first."""
        active = [r for r in reps if r.state in LIVE_STATES]
        failed = [r for r in reps if r.state == "failed"]
        newest_failed = failed[-1] if failed else None
        ready = any(r.state == "ready" for r in reps)
        launching = any(r.state in ("pending", "launching") for r in reps)
        error = None
        desired = autoscaler.desired(spec) if autoscaler is not None else spec.replicas
        if spec.replicas > 0 and desired == 0 and not active:
            state = "idle"  # started but unloaded; the next request loads it
        elif spec.replicas > 0:
            if ready:
                state = "running"
            elif launching:
                state = "starting"
            elif newest_failed is not None:
                state, error = "failed", newest_failed.error or "replica failed"
            elif (nofit := reconciler.nofit_reason(spec.name)):
                state, error = "failed", nofit
            else:
                state = "starting"  # desired > 0, the next reconcile tick will launch it
        elif any(r.state in ("draining", "ready", "launching", "pending") for r in reps):
            state = "stopping"
        else:
            state = "stopped"
        if error is None and newest_failed is not None and state == "starting":
            error = newest_failed.error
        listed = active + ([newest_failed] if newest_failed is not None else [])
        listed.sort(key=lambda r: (r.created_at, r.replica_id))
        file = spec.source[len(COORD_PREFIX):] if spec.source.startswith(COORD_PREFIX) else None
        lo, hi = bounds(spec)
        avg_busy = autoscaler.view(spec.name).get("avg_busy") if autoscaler is not None else None
        return {
            "spec": spec.model_dump(mode="json"), "file": file, "state": state, "error": error,
            "scaling": {"min": lo, "max": hi, "desired": desired, "avg_busy": avg_busy,
                        "unloaded": spec.replicas > 0 and desired == 0},
            "replicas": [{**r.model_dump(mode="json"), "outstanding": balancer.outstanding(r.replica_id)}
                         for r in listed],
        }

    def server_or_404(node_id: str) -> ServerRecord:
        s = store.get_server(node_id)
        if s is None:
            raise HTTPException(404, f"unknown server {node_id}")
        return s

    # ------------------------------------------------------------------ state
    @router.get("/state")
    async def state() -> dict:
        nodes = {n.report.node_id: n for n in store.list_nodes()}
        flags = store.gpu_flags()
        servers = [server_entry(s, nodes, flags) for s in store.list_servers()]
        gpus_total = gpus_enabled = pool_total = pool_usable = 0
        for s in servers:
            if s["report"] is None:
                continue
            for d in s["report"]["devices"]:
                if d["kind"] != "cuda":
                    continue
                on = s["gpu_enabled"].get(d["device_id"], True)
                gpus_total += 1
                if on:
                    gpus_enabled += 1
                    if s["alive"]:
                        pool_total += d["total_mb"]
                        pool_usable += d["usable_mb"]
        by_model: dict[str, list] = {}
        for r in store.list_replicas():  # one query, not one per model; oldest first
            by_model.setdefault(r.model, []).append(r)
        models = [model_entry(m, by_model.get(m.name, [])) for m in store.list_models()]
        return {
            "summary": {
                "servers_total": len(servers), "servers_online": sum(1 for s in servers if s["alive"]),
                "gpus_total": gpus_total, "gpus_enabled": gpus_enabled,
                "pool_total_mb": pool_total, "pool_usable_mb": pool_usable,
                "models_running": sum(1 for m in models if m["state"] == "running"),
            },
            "servers": servers,
            "models": models,
            "library": [i.model_dump(mode="json") for i in library.list()],
            "settings": {"public_url": cfg.public_url, "cluster_token": cfg.cluster_token,
                         "api_keys_set": bool(cfg.api_keys)},
            "events": [e.model_dump(mode="json") for e in store.list_events(limit=50)],
            "unread_events": store.unread_count(),
            "rebalance": reconciler.rebalance_state(),
        }

    # ------------------------------------------------------------------ servers
    @router.post("/servers")
    async def add_server(body: AddServerBody) -> dict:
        url = normalize_agent_url(body.agent_url)
        try:
            report = await poller.probe(url)
        except AgentError as e:
            if e.status == 401:
                raise HTTPException(400, f"agent at {url} rejected the token (401): check the cluster token")
            raise HTTPException(400, f"agent at {url} answered {e.status}: {e.body}")
        except httpx.HTTPError as e:
            raise HTTPException(400, f"cannot reach agent at {url}: {type(e).__name__} {e}".strip())
        except (ValidationError, ValueError) as e:
            raise HTTPException(400, f"{url} did not return a valid agent report: {e}")
        if store.get_server(report.node_id) is not None:
            raise HTTPException(409, f"server {report.node_id} is already registered")
        t = now()
        rec = ServerRecord(node_id=report.node_id, agent_url=url, added_at=t)
        store.add_server(rec)
        store.clear_removed(report.node_id)  # a manual add undoes an earlier removal
        store.upsert_node(report, t)
        emit("info", "server_added", f"Server {rec.node_id} added ({url})", node_id=rec.node_id)
        nodes = {n.report.node_id: n for n in store.list_nodes()}
        return server_entry(rec, nodes, store.gpu_flags())

    @router.delete("/servers/{node_id}")
    async def delete_server(node_id: str) -> dict:
        server_or_404(node_id)
        await reconciler.remove_node(node_id)
        store.mark_removed(node_id, now())  # sticks against the agent's auto-join until re-added here
        emit("info", "server_removed", f"Server {node_id} removed; its replicas were stopped", node_id=node_id)
        return {"ok": True}

    @router.put("/servers/{node_id}/gpus/{device_id}")
    async def set_gpu(node_id: str, device_id: str, body: GpuBody) -> dict:
        server_or_404(node_id)
        # The UI addresses GPUs by device_id; flags are stored by the card's identity.
        node = next((n for n in store.list_nodes() if n.report.node_id == node_id), None)
        dev = next((d for d in node.report.devices if d.device_id == device_id), None) if node else None
        store.set_gpu_enabled(node_id, gpu_key(dev) if dev else device_id, body.enabled)
        reconciler.wake()
        return {"node_id": node_id, "device_id": device_id, "enabled": body.enabled}

    # ------------------------------------------------------------------ models
    async def check_draft(spec: ModelSpec, label: str = "") -> None:
        """422 unless the draft model of a speculative "draft" spec is usable with its target."""
        if spec.speculative != "draft":
            return
        pre = f"{label}: " if label else ""
        if not spec.draft or not spec.draft.startswith(COORD_PREFIX):
            raise HTTPException(422, f"{pre}speculative 'draft' needs a draft_file from the library")
        file = spec.draft[len(COORD_PREFIX):]
        item = library.get(file)
        if item is None or item.status != "ready":
            raise HTTPException(422, f"{pre}draft file {file!r} is not a ready library item")
        if spec.source == spec.draft:
            raise HTTPException(422, f"{pre}the draft model must differ from the model itself")
        try:
            meta = await reconciler.meta_for(spec)
            dmeta = await reconciler.draft_meta_for(spec)
        except Exception as e:
            raise HTTPException(422, f"{pre}cannot read model metadata: {type(e).__name__}: {e}") from e
        if dmeta is None:
            return
        if meta.tokenizer_model and dmeta.tokenizer_model and meta.tokenizer_model != dmeta.tokenizer_model:
            raise HTTPException(422, f"{pre}draft tokenizer {dmeta.tokenizer_model!r} does not match "
                                     f"the model's {meta.tokenizer_model!r}")
        if (meta.vocab_size and dmeta.vocab_size
                and abs(meta.vocab_size - dmeta.vocab_size) > MAX_VOCAB_DIFF):
            raise HTTPException(422, f"{pre}draft vocabulary size {dmeta.vocab_size} is too different from "
                                     f"the model's {meta.vocab_size} (at most {MAX_VOCAB_DIFF} apart)")

    def check_name(name: str) -> None:
        if not NAME_RE.match(name):
            raise HTTPException(422, "model name must match [A-Za-z0-9._-]{1,64}")

    @router.put("/models/{name}")
    async def put_model(name: str, body: ModelBody) -> dict:
        check_name(name)
        item = library.get(body.file)
        if item is None or item.status != "ready":
            raise HTTPException(422, f"file {body.file!r} is not a ready library item")
        registered = {s.node_id for s in store.list_servers()}
        for pin in body.pin_devices:
            if not PIN_RE.match(pin):
                raise HTTPException(422, f"pin_devices entry {pin!r} must look like node/device")
            if pin.split("/", 1)[0] not in registered:
                raise HTTPException(422, f"pin_devices entry {pin!r} names an unregistered server")
        existing = store.get_model(name)

        def keep(new, field: str):
            return new if new is not None else getattr(existing, field) if existing else None

        lo, hi = keep(body.min_replicas, "min_replicas"), keep(body.max_replicas, "max_replicas")
        idle = keep(body.idle_unload_s, "idle_unload_s")
        if lo is not None and hi is not None and lo > hi:
            raise HTTPException(422, f"min_replicas ({lo}) must not exceed max_replicas ({hi})")
        if idle is not None and lo != 0:
            raise HTTPException(422, "idle_unload_s needs min_replicas == 0 (a model that may unload)")
        speculative = body.speculative if body.speculative is not None else existing.speculative if existing else "none"
        draft = COORD_PREFIX + body.draft_file if body.draft_file is not None else existing.draft if existing else None
        if speculative != "draft":
            draft = None  # a draft only means something to speculative "draft"
        spec = ModelSpec(
            name=name, source=COORD_PREFIX + body.file, ctx_size=body.ctx_size, parallel=body.parallel,
            kv_cache_type=body.kv_cache_type or (existing.kv_cache_type if existing else "f16"),
            speculative=speculative, draft=draft,
            draft_n_max=body.draft_n_max or (existing.draft_n_max if existing else 4),
            replicas=existing.replicas if existing else 0, pin_devices=list(dict.fromkeys(body.pin_devices)),
            priority=body.priority if body.priority is not None else existing.priority if existing else 50,
            spread=body.spread if body.spread is not None else existing.spread if existing else "gpu",
            min_replicas=lo, max_replicas=hi, autoscale=keep(body.autoscale, "autoscale"), idle_unload_s=idle,
            preemptible=body.preemptible if body.preemptible is not None else existing.preemptible if existing else True)
        await check_draft(spec)
        store.put_model(spec)
        reconciler.wake()
        return spec.model_dump(mode="json")

    @router.post("/models/{name}/start")
    async def start_model(name: str, body: StartBody | None = Body(default=None)) -> dict:
        spec = spec_or_404(store, name)
        spec = spec.model_copy(update={"replicas": (body or StartBody()).replicas})
        store.put_model(spec)
        emit("info", "model_started", f"Model {name} started ({spec.replicas} replica(s) requested)", model=name)
        reconciler.wake()  # not tick(): that waits on the tick lock and agent HTTP calls
        return spec.model_dump(mode="json")

    @router.get("/models/{name}/scaling")
    async def model_scaling(name: str) -> dict:
        if autoscaler is None:
            raise HTTPException(404, "autoscaling not available")
        spec_or_404(store, name)
        return autoscaler.view(name)

    @router.post("/models/{name}/stop")
    async def stop_model(name: str) -> dict:
        spec = spec_or_404(store, name).model_copy(update={"replicas": 0})
        store.put_model(spec)
        emit("info", "model_stopped", f"Model {name} stopped", model=name)
        reconciler.wake()
        return spec.model_dump(mode="json")

    @router.delete("/models/{name}")
    async def delete_model(name: str) -> dict:
        await drain_and_delete_model(store, reconciler, name)
        emit("info", "model_stopped", f"Model {name} removed", model=name)
        reconciler.wake()
        return {"ok": True}

    @router.post("/models/{name}/plan")
    async def plan_model(name: str) -> Any:
        spec = spec_or_404(store, name)
        try:
            return (await reconciler.plan_for(spec)).model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as e:
            raise plan_http_error(e) from e

    # ------------------------------------------------------------------ capacity
    async def capacity_view() -> dict:
        registered = {s.node_id for s in store.list_servers()}
        nodes = {n.report.node_id: n for n in store.list_nodes() if n.report.node_id in registered}
        flags = store.gpu_flags()
        t = now()
        free = {(r.node_id, d.device_id): d.usable_mb for r in reconciler.available_reports() for d in r.devices}
        occ: dict[tuple[str, str], list] = {}
        for o in reconciler.occupants():
            occ.setdefault((o.node_id, o.device_id), []).append(o)
        gpus = []
        for node_id in sorted(nodes):
            n = nodes[node_id]
            alive = reconciler.node_alive(n, t)
            for d in n.report.devices:
                enabled = flags.get((node_id, gpu_key(d)), True)
                live = enabled and alive
                free_mb = free.get((node_id, d.device_id), 0) if live else 0
                here = occ.get((node_id, d.device_id), [])
                gpus.append({
                    "node_id": node_id, "device_id": d.device_id, "uuid": d.uuid, "name": d.name,
                    "kind": d.kind, "enabled": enabled, "alive": alive,
                    "total_mb": d.total_mb, "usable_mb": d.usable_mb, "free_for_new_mb": free_mb,
                    "reserved_mb": max(0, d.usable_mb - free_mb) if live else 0,
                    "bandwidth_gbps": d.bandwidth_gbps,
                    "busy": min(1.0, sum(o.busy for o in here)),
                    "replicas": [{"replica_id": o.replica_id, "model": o.model, "est_mb": o.est_mb,
                                  "busy": o.busy} for o in here],
                })
        cuda = [g for g in gpus if g["kind"] == "cuda"]
        per_node: dict[str, int] = {}
        for g in cuda:
            per_node[g["node_id"]] = per_node.get(g["node_id"], 0) + g["free_for_new_mb"]
        return {"gpus": gpus, "summary": {
            "gpus": len(cuda), "free_for_new_mb": sum(g["free_for_new_mb"] for g in cuda),
            "largest_single_gpu_mb": max((g["free_for_new_mb"] for g in cuda), default=0),
            "largest_single_node_mb": max(per_node.values(), default=0)}}

    @router.get("/capacity")
    async def capacity() -> dict:
        return await capacity_view()

    # ------------------------------------------------------------------ simulate
    async def checked_spec(spec: ModelSpec) -> ModelSpec:
        """The same validation put_model applies, for a spec a simulation builds."""
        spec = ModelSpec.model_validate(spec.model_dump())  # field constraints (model_copy skips them)
        lo, hi = spec.min_replicas, spec.max_replicas
        if lo is not None and hi is not None and lo > hi:
            raise HTTPException(422, f"{spec.name}: min_replicas ({lo}) must not exceed max_replicas ({hi})")
        if spec.idle_unload_s is not None and lo != 0:
            raise HTTPException(422, f"{spec.name}: idle_unload_s needs min_replicas == 0")
        registered = {s.node_id for s in store.list_servers()}
        for pin in spec.pin_devices:
            if not PIN_RE.match(pin) or pin.split("/", 1)[0] not in registered:
                raise HTTPException(422, f"{spec.name}: pin_devices entry {pin!r} is not a registered node/device")
        if spec.speculative != "draft":
            spec = spec.model_copy(update={"draft": None})
        await check_draft(spec, spec.name)
        return spec

    @router.post("/simulate")
    async def simulate(body: SimulateBody) -> dict:
        """What the reconciler would do if `changes` and `add` were applied. Changes nothing.

        Per model it uses the autoscaler's current desired count (a model added here starts at its
        floor) and the same order, placement and preemption rules as a real tick. Unlike the real
        reconciler it ignores the preemption cooldown and treats memory freed by stops and
        evictions as available at once. Running replicas are not restarted by a changed
        ctx_size/parallel, so those only shape new ones."""
        specs = {s.name: s for s in store.list_models()}
        for ch in body.changes:
            cur = specs.get(ch.model)
            if cur is None:
                raise HTTPException(404, f"unknown model {ch.model}")
            specs[ch.model] = await checked_spec(cur.model_copy(update=ch.fields()))
        for add in body.add:
            check_name(add.name)
            if add.name in specs:
                raise HTTPException(422, f"model {add.name!r} already exists")
            item = library.get(add.file)
            if item is None or item.status != "ready":
                raise HTTPException(422, f"file {add.file!r} is not a ready library item")
            base = ModelSpec(name=add.name, source=COORD_PREFIX + add.file)
            specs[add.name] = await checked_spec(base.model_copy(update=add.fields()))
        try:
            return await reconciler.simulate(list(specs.values()))
        except HTTPException:
            raise
        except Exception as e:
            raise plan_http_error(e) from e

    # ------------------------------------------------------------------ rebalance
    @router.post("/rebalance")
    async def rebalance(body: RebalanceBody | None = Body(default=None)) -> dict:
        """Replica moves that would clearly improve placement; with dry_run false, starts the best
        one (make-before-break) when the cluster is quiet. One move at a time."""
        dry_run = True if body is None else body.dry_run
        try:
            moves = await reconciler.rebalance_candidates()
        except Exception as e:
            raise plan_http_error(e) from e
        started = None
        if not dry_run and moves and await reconciler.start_move(moves[0]):
            started = {"replica_id": moves[0]["replica_id"], "model": moves[0]["model"]}
            reconciler.wake()
        mv = reconciler.rebalance_state()["in_progress"]
        return {"moves": moves, "started": started, "in_progress": mv}

    # ------------------------------------------------------------------ recommend
    @router.post("/recommend")
    async def recommend(body: RecommendBody) -> dict:
        item = library.get(body.file)
        if item is None or item.status != "ready":
            raise HTTPException(422, f"file {body.file!r} is not a ready library item")
        stem = re.sub(r"[^A-Za-z0-9._-]", "-", body.file.rsplit("/", 1)[-1].rsplit(".", 1)[0])[:64]

        def spec_for(ctx: int) -> ModelSpec:
            return ModelSpec(
                name=stem or "recommend", source=COORD_PREFIX + body.file, ctx_size=ctx, parallel=body.parallel,
                replicas=1, pin_devices=body.pin_devices, priority=body.priority, spread=body.spread,
                kv_cache_type=body.kv_cache_type, speculative=body.speculative,
                draft=COORD_PREFIX + body.draft_file if body.speculative == "draft" and body.draft_file else None,
                draft_n_max=body.draft_n_max)

        await check_draft(spec_for(body.ctx_size))
        try:
            meta = await reconciler.meta_for(spec_for(body.ctx_size))
            dmeta = await reconciler.draft_meta_for(spec_for(body.ctx_size))
        except Exception as e:
            raise HTTPException(400, f"cannot read model metadata: {type(e).__name__}: {e}") from e

        async def largest_ctx(hi_ctx: int, ok) -> int | None:
            # Binary search over multiples of 256; assumes a bigger context never needs less memory.
            lo, hi, best = 1, hi_ctx // 256, None
            while lo <= hi:
                mid = (lo + hi) // 2
                if ok(await reconciler.rank_for(spec_for(mid * 256), 10)):
                    best, lo = mid * 256, mid + 1
                else:
                    hi = mid - 1
            return best

        need = total_need_mb(meta, body.ctx_size, body.kv_cache_type)
        if dmeta is not None:
            need += total_need_mb(dmeta, body.ctx_size, body.kv_cache_type)
        try:
            ranked = (await reconciler.rank_for(spec_for(body.ctx_size), body.limit))[: body.limit]
            max_single = await largest_ctx(131072, lambda opts: any(o.tier == "single_gpu" for o in opts))
            not_possible = None
            preempt_option = None
            if not ranked:
                try:  # best effort: a failure here must not hide the plain "nothing fits" answer
                    preempt_option = await reconciler.rank_with_preemption(spec_for(body.ctx_size), 1)
                except Exception:
                    log.exception("preemption lookup for recommend failed")
            if not ranked and preempt_option is None:
                cap = (await capacity_view())["summary"]
                not_possible = {"need_mb": need, "largest_single_gpu_mb": cap["largest_single_gpu_mb"],
                                "largest_single_node_mb": cap["largest_single_node_mb"],
                                "max_ctx_that_fits": await largest_ctx(body.ctx_size - 1, bool)}
        except Exception as e:
            raise plan_http_error(e) from e
        def option(i: int, p, extra: dict) -> dict:
            return {"rank": i + 1, "score": p.score, "tier": p.tier, "fits_now": not extra, **extra,
                    "assignments": [{"node_id": a.node_id, "device_id": a.device_id, "layers": a.layers,
                                     "est_mb": a.est_mb} for a in p.assignments],
                    "est_decode_tps": p.est_decode_tps, "est_total_mb": p.est_total_mb, "reasons": p.reasons}

        options = [option(i, p, {}) for i, p in enumerate(ranked)]
        if preempt_option is not None:
            victims, placements = preempt_option
            prio = {s.name: s.priority for s in store.list_models()}
            options.append(option(0, placements[0], {"requires_preemption": [
                {"replica_id": v.replica_id, "model": v.model, "priority": prio.get(v.model, 0)}
                for v in victims]}))
        return {
            "need_mb": need,
            "options": options,
            "max_ctx_single_gpu": max_single,
            "not_possible": not_possible,
        }

    # ------------------------------------------------------------------ events
    @router.get("/events")
    async def events(limit: int = 200, after_id: int | None = None) -> dict:
        limit = max(1, min(limit, 1000))
        return {"events": [e.model_dump(mode="json") for e in store.list_events(limit=limit, after_id=after_id)],
                "unread": store.unread_count()}

    @router.post("/events/read")
    async def mark_read(body: ReadBody) -> dict:
        store.mark_read(body.up_to_id)
        return {"unread": store.unread_count()}

    return router
