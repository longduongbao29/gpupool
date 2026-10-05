"""Shared doubles for the coordinator tests (no tests in here)."""
from __future__ import annotations

import asyncio

from gpupool.common.config import CoordinatorConfig
from gpupool.common.models import (
    Device,
    DeviceAssignment,
    EngineSpec,
    EngineStatus,
    ModelMeta,
    ModelSpec,
    NodeReport,
    Placement,
    ReplicaRecord,
)
from gpupool.coordinator.reconciler import Reconciler
from gpupool.coordinator.store import Store
from gpupool.scheduler.placement import NoFit

META = ModelMeta(arch="llama", n_layers=4, n_embd=64, n_head=4, n_head_kv=4, head_dim=16,
                 layer_bytes=[1] * 4, other_bytes=1, output_bytes=1)


def dev(device_id="CUDA0", free=8000, usable=7000, uuid=None, budget=None) -> Device:
    return Device(device_id=device_id, kind="cuda", name=device_id, total_mb=10000, free_mb=free, usable_mb=usable,
                  uuid=uuid, budget_mb=budget)


def node(node_id="a", host=None, devices=None, engines=None, ts=0.0) -> NodeReport:
    host = host or {"a": "10.0.0.1", "b": "10.0.0.2", "c": "10.0.0.3"}.get(node_id, "10.0.0.9")
    return NodeReport(node_id=node_id, agent_url=f"http://{host}:7070", host=host,
                      devices=devices if devices is not None else [dev()],
                      engines=engines or [], llama_version="x", models=[], ts=ts)


def make_planner(est_mb=1000, rpc=False):
    """Fake scheduler.plan: first device with enough usable_mb is the head; with rpc=True a
    second device on another node is added as RPC0."""

    def planner(meta, spec, nodes, rid, port_alloc, exclude_nodes=frozenset(), **kw):
        for n in nodes:
            for d in n.devices:
                if d.usable_mb >= est_mb:
                    head_port = port_alloc(n.node_id)
                    asg = [DeviceAssignment(node_id=n.node_id, device_id=d.device_id, llama_device=d.device_id,
                                            layers=2, est_mb=est_mb)]
                    if rpc:
                        o = next(x for x in nodes if x.node_id != n.node_id)
                        asg.append(DeviceAssignment(
                            node_id=o.node_id, device_id=o.devices[0].device_id, llama_device="RPC0",
                            rpc_endpoint=f"{o.host}:{port_alloc(o.node_id)}", layers=2, est_mb=est_mb))
                    return Placement(model=spec.name, replica_id=rid, tier="multi_node" if rpc else "single_gpu",
                                     head_node=n.node_id, head_port=head_port, assignments=asg,
                                     tensor_split=[1.0] * len(asg), est_total_mb=est_mb * len(asg))
        raise NoFit("nothing fits")

    return planner


class FakeClient:
    def __init__(self):
        self.calls: list[tuple] = []
        self.engines: dict[tuple[str, str], EngineStatus] = {}
        self.fail_start_on: str | None = None  # engine_id suffix that makes start_engine raise
        self.head_state = "running"
        self.memory: dict | None = None  # what engine_memory returns; an Exception instance is raised
        self.memory_calls: list[tuple] = []

    async def start_engine(self, url, spec: EngineSpec):
        self.calls.append(("start", url, spec.engine_id, spec))
        if self.fail_start_on and spec.engine_id.endswith(self.fail_start_on):
            raise RuntimeError("boom")
        st = EngineStatus(engine_id=spec.engine_id, kind=spec.kind, state="running", port=spec.port)
        self.engines[(url, spec.engine_id)] = st
        return st

    async def stop_engine(self, url, engine_id):
        self.calls.append(("stop", url, engine_id))
        return self.engines.pop((url, engine_id), None)

    async def get_engine(self, url, engine_id):
        self.calls.append(("get", url, engine_id))
        st = self.engines.get((url, engine_id))
        if st is not None and engine_id.endswith("-head"):
            st = st.model_copy(update={"state": self.head_state, "log_tail": ["head says hi"]})
        return st

    async def ensure_model(self, url, name, source):
        self.calls.append(("ensure", url, name, source))
        return "/cache/" + name

    async def engine_memory(self, url, engine_id):
        self.memory_calls.append((url, engine_id))
        if isinstance(self.memory, Exception):
            raise self.memory
        return self.memory

    async def aclose(self):
        pass

    def kinds(self):
        return [c[0] for c in self.calls if c[0] != "get"]


def make_cfg(tmp_path=None, **kw) -> CoordinatorConfig:
    kw.setdefault("heartbeat_timeout_s", 10.0)
    kw.setdefault("launch_timeout_s", 5.0)
    if tmp_path is not None:
        kw.setdefault("models_dir", tmp_path / "models")
        # Never the default .gpupool/ in the working directory: CI checkouts do not have it.
        kw.setdefault("db_path", tmp_path / "coordinator.db")
    return CoordinatorConfig(**kw)


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_reconciler(cfg=None, planner=None, client=None, outstanding=None, clock=None, store=None):
    store = store or Store(":memory:")
    clock = clock or Clock()
    cfg = cfg or make_cfg()

    async def meta_for(spec):
        return META

    rec = Reconciler(store, cfg, client or FakeClient(), meta_for, outstanding or (lambda rid: 0), clock)
    rec.planner = planner or make_planner()
    rec.poll_s = 0.01
    return rec, store, clock


async def settle(rec: Reconciler):
    while rec._launches:
        await asyncio.gather(*list(rec._launches.values()), return_exceptions=True)
        await asyncio.sleep(0)


def put_replica(store, rid="m-1", model="m", state="ready", head="a", rpc_node=None, now=1000.0,
                head_port=9000, rpc_port=9001, head_device="CUDA0", head_uuid=None) -> ReplicaRecord:
    asg = [DeviceAssignment(node_id=head, device_id=head_device, llama_device="CUDA0", layers=2, est_mb=1000,
                            device_uuid=head_uuid)]
    if rpc_node:
        asg.append(DeviceAssignment(node_id=rpc_node, device_id="CUDA0", llama_device="RPC0",
                                    rpc_endpoint=f"10.0.0.2:{rpc_port}", layers=2, est_mb=1000))
    p = Placement(model=model, replica_id=rid, tier="single_gpu", head_node=head, head_port=head_port,
                  assignments=asg, tensor_split=[1.0] * len(asg), est_total_mb=1000)
    rec = ReplicaRecord(replica_id=rid, model=model, placement=p, state=state, created_at=now, updated_at=now)
    store.put_replica(rec)
    return rec


SPEC = ModelSpec(name="m", source="http://x/m.gguf")


class FakeAutoscaler:
    """Stands in for coordinator.autoscaler.Autoscaler: fixed answers, records calls."""

    def __init__(self, desired=None, cold=False, avg_busy=None):
        self.want = desired if desired is not None else {}  # model -> count; missing = spec.replicas
        self.cold = cold
        self.avg_busy = avg_busy
        self.requests: list[str] = []
        self.ran = False
        self.closed = False

    def desired(self, spec) -> int:
        return self.want.get(spec.name, spec.replicas)

    def peek(self, spec) -> int:
        return self.desired(spec)

    def can_cold_start(self, model) -> bool:
        return self.cold

    def note_request(self, model) -> bool:
        self.requests.append(model)
        return False

    def measured_tps(self, replica_id) -> float | None:
        return None

    def view(self, model) -> dict:
        return {"model": model, "avg_busy": self.avg_busy}

    async def run(self) -> None:
        self.ran = True
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def make_ranker(est_mb=1000):
    """Fake scheduler.rank, consistent with make_planner: every device with enough usable_mb is one
    single-GPU candidate, in report order."""

    def ranker(meta, spec, nodes, occupants=(), limit=5):
        out = []
        for n in nodes:
            for d in n.devices:
                if d.usable_mb >= est_mb:
                    asg = [DeviceAssignment(node_id=n.node_id, device_id=d.device_id, llama_device=d.device_id,
                                            layers=2, est_mb=est_mb)]
                    out.append(Placement(model=spec.name, replica_id="", tier="single_gpu", head_node=n.node_id,
                                         head_port=0, assignments=asg, tensor_split=[1.0], est_total_mb=est_mb,
                                         est_decode_tps=40.0))
        return out[:limit]

    return ranker


def make_scored_ranker(scores: dict[tuple[str, str], float], est_mb=1000):
    """Fake scheduler.rank with controlled scores: a placement scores `scores[(node, device)]` of
    its first assignment. Candidates (replica_id "") exist for every device with enough usable_mb
    and a score entry; `extra` placements are always returned with their own id and a fresh score;
    a candidate identical to an extra is dropped. `.calls` records every call's kwargs."""

    def ranker(meta, spec, nodes, occupants=(), limit=5, extra=()):
        key = lambda p: tuple((a.node_id, a.device_id) for a in p.assignments)  # noqa: E731
        out = [p.model_copy(update={"score": scores[key(p)[0]]}) for p in extra]
        taken = {key(p) for p in extra}
        cands = []
        for n in nodes:
            for d in n.devices:
                k = ((n.node_id, d.device_id),)
                if d.usable_mb < est_mb or k[0] not in scores or k in taken:
                    continue
                asg = [DeviceAssignment(node_id=n.node_id, device_id=d.device_id, llama_device=d.device_id,
                                        layers=2, est_mb=est_mb)]
                cands.append(Placement(model=spec.name, replica_id="", tier="single_gpu", head_node=n.node_id,
                                       head_port=0, assignments=asg, tensor_split=[1.0], est_total_mb=est_mb,
                                       score=scores[k[0]], reasons=[f"fits {n.node_id}/{d.device_id}"]))
        cands.sort(key=lambda p: -p.score)
        ranker.calls.append({"spec": spec, "nodes": nodes, "occupants": list(occupants), "limit": limit,
                             "extra": list(extra)})
        return out + cands[:limit]

    ranker.calls = []
    return ranker
