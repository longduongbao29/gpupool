"""Wire contracts shared by agent, scheduler, coordinator and router.

Every module codes against these shapes. Changing a field here is a protocol
change: agent and coordinator of different versions must keep talking.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DeviceKind = Literal["cuda", "cpu"]
EngineKind = Literal["rpc", "server"]
EngineState = Literal["starting", "running", "exited", "failed"]
ReplicaState = Literal["pending", "launching", "ready", "draining", "stopped", "failed"]
Tier = Literal["single_gpu", "single_node", "multi_node"]

# Replica state groups (not part of the wire format).
ALL_REPLICA_STATES: tuple[str, ...] = ("pending", "launching", "ready", "draining", "stopped", "failed")
ACTIVE_STATES = frozenset({"pending", "launching", "ready"})  # count toward the desired replicas
LIVE_STATES = ACTIVE_STATES | {"draining"}  # still hold ports/engines: everything but stopped/failed
TERMINAL_STATES = frozenset({"stopped", "failed"})


class GpuProcess(BaseModel):
    pid: int
    name: str = ""  # "" when the OS hides it (other users' processes)
    used_mb: int | None = None


class Device(BaseModel):
    device_id: str  # llama.cpp name on its own node: "CUDA0", "CPU"
    kind: DeviceKind
    name: str
    total_mb: int
    free_mb: int  # measured (NVML / psutil)
    usable_mb: int  # max(0, min(free_mb - margin, budget_mb))
    util_pct: int | None = None
    # Optional telemetry for the UI (agents before 0.2 do not send them).
    temp_c: int | None = None
    power_w: int | None = None
    processes: list[GpuProcess] = Field(default_factory=list)
    driver: str | None = None  # NVIDIA driver version, e.g. "535.154.05"
    cuda: str | None = None  # highest CUDA version the driver supports, e.g. "12.2"
    # Stable identity of a physical GPU (NVML), unlike device_id which is a position that
    # shifts when a GPU disappears. None for CPU devices and agents before 0.3.
    uuid: str | None = None  # "GPU-8f2c..."
    pci_bus_id: str | None = None  # "00000000:01:00.0"


class EngineSpec(BaseModel):
    """Coordinator -> agent: start one llama.cpp process."""

    engine_id: str  # "<replica_id>-head" | "<replica_id>-rpc-<device_id>"
    kind: EngineKind
    port: int
    # rpc: exactly one local device. server: full ordered list, e.g. ["CUDA0", "RPC0", "RPC1"].
    devices: list[str]
    model: str | None = None  # server: alias served at /v1/models
    model_path: str | None = None  # server: GGUF path on the head node
    rpc_endpoints: list[str] = Field(default_factory=list)  # "host:port", order matches RPC0, RPC1...
    tensor_split: list[float] = Field(default_factory=list)  # same length and order as devices
    ctx_size: int = 4096  # total context, divided across slots
    parallel: int = 1
    extra_args: list[str] = Field(default_factory=list)


class EngineStatus(BaseModel):
    engine_id: str
    kind: EngineKind
    state: EngineState
    pid: int | None = None
    port: int
    exit_code: int | None = None
    log_tail: list[str] = Field(default_factory=list)  # at most 50 last lines


class NodeReport(BaseModel):
    """Agent -> coordinator heartbeat payload."""

    node_id: str
    agent_url: str  # "http://10.0.0.5:7070"
    host: str  # address other nodes use to reach this node's rpc-servers
    devices: list[Device]
    engines: list[EngineStatus]
    llama_version: str
    models: list[str]  # GGUF file names present in the local cache
    ts: float
    # Optional host telemetry for the UI.
    cpu_pct: float | None = None
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None


class ModelSpec(BaseModel):
    name: str  # alias returned by /v1/models
    # https://...gguf | absolute local path (on the coordinator) | coordinator://<file in models_dir>
    source: str
    ctx_size: int = 4096
    parallel: int = 1
    replicas: int = 1  # desired count; 0 = stopped
    # "node_id/device_id" entries the replicas may use; empty = the scheduler chooses freely.
    pin_devices: list[str] = Field(default_factory=list)


class LibraryItem(BaseModel):
    """A GGUF file the coordinator can serve to heads as coordinator://<name>."""

    name: str  # unique file name, e.g. "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    path: str  # absolute path on the coordinator machine
    source: Literal["hf", "path"]
    hf_repo: str | None = None
    hf_file: str | None = None  # path inside the repo (may contain "/")
    bytes: int | None = None  # total size when known
    downloaded: int = 0  # bytes written so far (HF downloads)
    status: Literal["downloading", "ready", "failed"]
    error: str | None = None
    created_at: float


class ModelMeta(BaseModel):
    """Read from the GGUF header; enough to estimate memory per layer."""

    arch: str
    n_layers: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    layer_bytes: list[int]  # total tensor bytes of block "blk.{i}." for i in range(n_layers)
    other_bytes: int  # every tensor not in a block: token_embd, output, output_norm...
    # Bytes llama.cpp puts on the LAST offload device: output.weight (or token_embd.weight
    # when the model ties embeddings and has no output.weight) + output_norm.*.
    # token_embd itself stays in host RAM, so it is not counted against any GPU.
    output_bytes: int
    file_bytes: int | None = None


class DeviceAssignment(BaseModel):
    node_id: str
    device_id: str  # name on its own node ("CUDA0")
    llama_device: str  # name as seen from the head: "CUDA0" (local) or "RPC0", "RPC1"...
    rpc_endpoint: str | None = None  # "host:port"; None when local to the head
    layers: int
    est_mb: int
    device_uuid: str | None = None  # Device.uuid at planning time, when the agent reports one


class Placement(BaseModel):
    model: str
    replica_id: str
    tier: Tier
    head_node: str
    head_port: int
    assignments: list[DeviceAssignment]  # order == --device order == tensor_split order
    tensor_split: list[float]
    est_total_mb: int


class ReplicaRecord(BaseModel):
    replica_id: str
    model: str
    placement: Placement
    state: ReplicaState
    created_at: float
    updated_at: float
    error: str | None = None


class ReplicaEndpoint(BaseModel):
    """What the router needs to send traffic to a ready replica."""

    replica_id: str
    model: str
    base_url: str  # "http://10.0.0.5:9001" (llama-server on the head)
