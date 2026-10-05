"""Decode-speed estimate for a placement (design section 4.3).

Decode is memory-bandwidth bound: every token streams all weights once, and layers on
different devices run one after another, so time per token is the sum over devices of
(bytes on device / effective bandwidth) plus a fixed cost per network hop.
Calibrated on a GTX 1650 (160 GB/s): Qwen2.5-0.5B q4_k_m measured 182 tok/s by llama-bench.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gpupool.common.models import Device, ModelMeta, NodeReport

ETA = 0.5  # fraction of peak bandwidth llama.cpp reaches in decode
HOP_S = 0.002  # per RPC hop, per token
CPU_BW_GBPS = 25.0  # host RAM, when the agent reports nothing
CUDA_BW_FALLBACK_GBPS = 100.0  # when no CUDA device in the pool reports bandwidth
# The output layer's device hands llama-server n_vocab f32 logits per token. When that device is
# remote they cross the network: priced at 1 Gbit/s (125 MB/s), a common LAN.
NET_BYTES_PER_S = 125e6
VOCAB_FALLBACK = 128_000


def default_cuda_bw(nodes: Sequence[NodeReport]) -> float:
    """Lowest known CUDA bandwidth: an unknown (old agent) GPU ranks as the slowest known one."""
    known = [d.bandwidth_gbps for n in nodes for d in n.devices
             if d.kind == "cuda" and d.bandwidth_gbps]
    return min(known) if known else CUDA_BW_FALLBACK_GBPS


def device_bw(dev: Device, cuda_default: float) -> float:
    if dev.bandwidth_gbps:
        return dev.bandwidth_gbps
    return CPU_BW_GBPS if dev.kind == "cpu" else cuda_default


def decode_bytes(meta: ModelMeta) -> list[int]:
    """Bytes one decoded token reads per layer: every weight of a dense layer, the shared part
    plus the routed share (expert_used_count / expert_count) of a MoE layer's experts."""
    return meta.active_bytes if meta.active_bytes is not None else meta.layer_bytes


@dataclass
class SpeedModel:
    """eta: fraction of peak bandwidth decode reaches; hop_s: seconds per RPC server per token.
    Start at the GTX 1650 calibration; the coordinator refines both from measured decode speed
    (coordinator/reconciler.py, _learn_speed) and installs them with set_speed_model."""

    eta: float = ETA
    hop_s: float = HOP_S


ETA_RANGE = (0.15, 0.95)
HOP_RANGE = (0.0001, 0.05)
_SPEED = SpeedModel()


def speed_model() -> SpeedModel:
    return _SPEED


def set_speed_model(eta: float, hop_s: float) -> None:
    """Process-wide (the planner runs in the API handlers and the reconciler alike), clamped."""
    global _SPEED
    _SPEED = SpeedModel(min(max(eta, ETA_RANGE[0]), ETA_RANGE[1]),
                        min(max(hop_s, HOP_RANGE[0]), HOP_RANGE[1]))


def bandwidth_seconds(
    meta: ModelMeta,
    devices_with_layers: Sequence[tuple[Device, int]],
    cuda_default_gbps: float = CUDA_BW_FALLBACK_GBPS,
) -> float:
    """Seconds per token to stream the weights at full peak bandwidth (eta = 1, no network)."""
    t, start = 0.0, 0
    last = len(devices_with_layers) - 1
    read = decode_bytes(meta)
    for i, (dev, count) in enumerate(devices_with_layers):
        b = sum(read[start:start + count])
        if i == last:
            b += meta.output_bytes
        t += b / (device_bw(dev, cuda_default_gbps) * 1e9)
        start += count
    return t


def current_tps(bw_s: float, hops: int, logits_s: float) -> float:
    """Decode tok/s of stored speed parts under the current (learned) speed model."""
    t = bw_s / _SPEED.eta + hops * _SPEED.hop_s + logits_s
    return 1.0 / t if t > 0 else 0.0


def logits_seconds(meta: ModelMeta) -> float:
    """Network time of one token's logits when the output layer is on another server."""
    return (meta.vocab_size or VOCAB_FALLBACK) * 4 / NET_BYTES_PER_S


def est_decode_tps(
    meta: ModelMeta,
    devices_with_layers: Sequence[tuple[Device, int]],
    n_rpc: int = 0,
    cuda_default_gbps: float = CUDA_BW_FALLBACK_GBPS,
    remote_last: bool = False,
) -> float:
    """Tokens/s for devices in layer order; the last one also holds the output tensors.
    remote_last: that device is reached over RPC, so the logits travel back to the head."""
    sm = _SPEED
    t = bandwidth_seconds(meta, devices_with_layers, cuda_default_gbps) / sm.eta
    t += n_rpc * sm.hop_s
    if remote_last:
        t += logits_seconds(meta)
    return 1.0 / t if t > 0 else 0.0
