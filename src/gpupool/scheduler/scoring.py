"""Decode-speed estimate for a placement (design section 4.3).

Decode is memory-bandwidth bound: every token streams all weights once, and layers on
different devices run one after another, so time per token is the sum over devices of
(bytes on device / effective bandwidth) plus a fixed cost per network hop.
Calibrated on a GTX 1650 (160 GB/s): Qwen2.5-0.5B q4_k_m measured 182 tok/s by llama-bench.
"""
from __future__ import annotations

from collections.abc import Sequence

from gpupool.common.models import Device, ModelMeta, NodeReport

ETA = 0.5  # fraction of peak bandwidth llama.cpp reaches in decode
HOP_S = 0.002  # per RPC hop, per token
CPU_BW_GBPS = 25.0  # host RAM, when the agent reports nothing
CUDA_BW_FALLBACK_GBPS = 100.0  # when no CUDA device in the pool reports bandwidth


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


def est_decode_tps(
    meta: ModelMeta,
    devices_with_layers: Sequence[tuple[Device, int]],
    n_rpc: int = 0,
    cuda_default_gbps: float = CUDA_BW_FALLBACK_GBPS,
) -> float:
    """Tokens/s for devices in layer order; the last one also holds the output tensors."""
    t, start = 0.0, 0
    last = len(devices_with_layers) - 1
    read = decode_bytes(meta)
    for i, (dev, count) in enumerate(devices_with_layers):
        b = sum(read[start:start + count])
        if i == last:
            b += meta.output_bytes
        t += b / (device_bw(dev, cuda_default_gbps) * 1e9 * ETA)
        start += count
    t += n_rpc * HOP_S
    return 1.0 / t if t > 0 else 0.0
