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
Spread = Literal["gpu", "node", "none"]
# KV cache element type. Bytes per element: f16 2, q8_0 34/32, q4_0 18/32 (llama.cpp block formats).
KvCacheType = Literal["f16", "q8_0", "q4_0"]
# Speculative decoding: "ngram" guesses from the text so far (no extra memory); "draft" runs a small
# model with the same tokenizer on the head's GPU. Fewer target passes = fewer RPC round trips.
SpecMode = Literal["none", "ngram", "draft"]

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
    # The cap configured on the agent for this device, None when uncapped. usable_mb alone cannot
    # enforce it: free memory does not show what our own engines hold (no per-process NVML on
    # WDDM), so the coordinator subtracts its replicas' estimates from this itself. Optional: old
    # agents do not send it.
    budget_mb: int | None = None
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
    # Peak memory bandwidth (NVML bus width x memory clock). Decode speed is bandwidth-bound, so
    # this ranks GPUs; None for CPU devices and older agents.
    bandwidth_gbps: float | None = None


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
    # Optional since 0.3; an older agent ignores them (it would run f16 / no speculation).
    cache_type: KvCacheType = "f16"  # -ctk/-ctv for the model (and its draft)
    spec_type: SpecMode = "none"
    draft_model_path: str | None = None  # spec_type "draft": GGUF on the head node
    draft_device: str | None = None  # llama device the draft runs on, e.g. "CUDA0" (local to the head)
    draft_n_max: int = 4
    # rpc engines: hosts allowed to connect (the replica's head). Empty = no restriction. Enforced
    # only by agents with rpc_firewall on; ggml-rpc-server itself has no authentication.
    allowed_peers: list[str] = Field(default_factory=list)


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
    # Allowed devices, "<node>/<device>" or "<node>/*" (every device of that server). Empty = all.
    # A limit, not a placement: the scheduler still chooses among the allowed devices.
    pin_devices: list[str] = Field(default_factory=list)
    # Higher places first each tick, so it gets scarce VRAM before lower priorities.
    priority: int = Field(default=50, ge=0, le=100)
    # Where replicas of this model should avoid each other: "gpu" (different GPUs), "node"
    # (different servers), "none". Soft: a shared GPU is still used when nothing else fits.
    spread: Spread = "gpu"
    # Autoscaling bounds. None = `replicas` (fixed count, today's behaviour). `replicas` stays the
    # on/off switch: 0 stops the model whatever these say.
    min_replicas: int | None = Field(default=None, ge=0)
    max_replicas: int | None = Field(default=None, ge=1)
    autoscale: AutoscalePolicy | None = None  # None = default thresholds when max > min
    # Only with min_replicas == 0: unload after this many seconds without a request; the next
    # request loads it again (cold start).
    idle_unload_s: float | None = Field(default=None, gt=0)
    # False: a higher-priority model may never stop this model's replicas to make room.
    preemptible: bool = True
    kv_cache_type: KvCacheType = "f16"
    speculative: SpecMode = "none"
    draft: str | None = None  # speculative "draft": source of the draft model, e.g. coordinator://<file>
    # Measured on a GTX 1650 (Qwen2.5-3B + 0.5B draft): 4 drafted tokens +5 %, 8 slower than none.
    draft_n_max: int = Field(default=4, ge=1, le=16)


class AutoscalePolicy(BaseModel):
    target_busy: float = Field(default=0.7, gt=0, le=1)  # busy slots / slots that triggers scale-up
    up_after_s: float = Field(default=30.0, ge=0)  # sustained this long before adding a replica
    down_after_s: float = Field(default=300.0, ge=0)  # below target/2 this long before removing one


class LibraryItem(BaseModel):
    """A GGUF file the coordinator can serve to heads as coordinator://<name>."""

    name: str  # unique file name, e.g. "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    path: str  # absolute path on the coordinator machine
    source: Literal["hf", "path", "convert"]  # convert: written by a conversion job, owned like hf
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
    # Tokenizer identity, to check a draft model matches its target (llama.cpp refuses otherwise).
    vocab_size: int | None = None
    tokenizer_model: str | None = None  # tokenizer.ggml.model, e.g. "gpt2", "llama"


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
    # Why the scheduler chose this placement (absent on placements made before 0.3).
    score: float | None = None
    est_decode_tps: float | None = None  # bandwidth-based estimate, None when unknown
    reasons: list[str] = Field(default_factory=list)
    # Speculative "draft": the draft model's memory, placed on the head's first local CUDA device
    # (assignments[0]); included in that assignment's est_mb and in est_total_mb.
    draft_est_mb: int | None = None
    # Calibrated memory factor every est_mb above was multiplied by. Calibration divides it back out:
    # a sample against already-scaled estimates would converge on sqrt(true ratio), not the ratio.
    mem_factor: float = 1.0


class Occupant(BaseModel):
    """An engine already holding a GPU, as the scheduler sees it when scoring a new placement."""

    node_id: str
    device_id: str  # current device_id on its node (the reconciler resolves shifts by uuid)
    model: str
    replica_id: str
    est_mb: int
    busy: float = 0.0  # 0..1: outstanding requests / parallel slots


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
