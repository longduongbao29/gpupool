"""Pure memory-estimate functions (design section 6).

Calibrated against llama.cpp b11342 on a GTX 1650 (verbose load logs):
  - model buffer on the device == sum of its layer tensors (+ output tensors on the last
    device), exact to the MiB for Qwen2.5-0.5B and 3B;
  - KV buffer == 2 * ctx * n_head_kv * head_dim * 2 bytes per layer, exact;
  - compute buffer == 37.8 MiB (n_embd 896) and 80.5 MiB (n_embd 2048), i.e.
    ~ 21 * ubatch(512) * n_embd * 4 bytes (rounded up from ~20.6 for 0.5B, ~19.6 for 3B);
  - CUDA context ~65 MB on Windows/WDDM; budget 128 MB because Linux drivers and
    newer GPUs load more kernels up front.
"""
from __future__ import annotations

import math

from gpupool.common.models import ModelMeta

_MB = 1024 * 1024
CONTEXT_MB = {"cuda": 128, "cpu": 32}  # runtime context / allocator slack per device
_COMPUTE_BYTES_PER_EMBD = 21 * 512 * 4  # measured factor (rounded up) x default ubatch x f32


def compute_mb(meta: ModelMeta) -> int:
    """Per-device compute buffer for the default ubatch (512)."""
    return math.ceil(_COMPUTE_BYTES_PER_EMBD * meta.n_embd / _MB)


def overhead_mb(meta: ModelMeta, kind: str) -> int:
    """Memory a device needs beyond weights and KV: compute buffer + runtime context."""
    return compute_mb(meta) + CONTEXT_MB[kind]


def kv_bytes_per_layer(meta: ModelMeta, ctx_size: int) -> int:
    return 2 * ctx_size * meta.n_head_kv * meta.head_dim * 2


def device_need_mb(meta: ModelMeta, layers: range, ctx_size: int, kind: str, is_last: bool) -> int:
    total = sum(meta.layer_bytes[i] for i in layers) + len(layers) * kv_bytes_per_layer(meta, ctx_size)
    if is_last:
        total += meta.output_bytes
    return math.ceil(total / _MB) + overhead_mb(meta, kind)


def total_need_mb(meta: ModelMeta, ctx_size: int) -> int:
    return device_need_mb(meta, range(meta.n_layers), ctx_size, "cuda", True)
