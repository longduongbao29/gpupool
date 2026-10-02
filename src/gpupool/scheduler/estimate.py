"""Pure memory-estimate functions (design section 6)."""
from __future__ import annotations

import math

from gpupool.common.models import ModelMeta

OVERHEAD_MB = {"cuda": 300, "cpu": 150}
_MB = 1024 * 1024


def kv_bytes_per_layer(meta: ModelMeta, ctx_size: int) -> int:
    return 2 * ctx_size * meta.n_head_kv * meta.head_dim * 2


def device_need_mb(meta: ModelMeta, layers: range, ctx_size: int, kind: str, is_last: bool) -> int:
    total = sum(meta.layer_bytes[i] for i in layers) + len(layers) * kv_bytes_per_layer(meta, ctx_size)
    if is_last:
        total += meta.output_bytes
    return math.ceil(total / _MB) + OVERHEAD_MB[kind]


def total_need_mb(meta: ModelMeta, ctx_size: int) -> int:
    return device_need_mb(meta, range(meta.n_layers), ctx_size, "cuda", True)
