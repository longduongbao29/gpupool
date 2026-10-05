"""Pure memory-estimate functions (design section 6).

Calibrated against llama.cpp b11342 on a GTX 1650 (verbose load logs):
  - model buffer on the device == sum of its layer tensors (+ output tensors on the last
    device), exact to the MiB for Qwen2.5-0.5B and 3B;
  - KV buffer == 2 * ctx * n_head_kv * head_dim * 2 bytes per layer, exact;
  - compute buffer == 37.8 MiB (n_embd 896) and 80.5 MiB (n_embd 2048), i.e.
    ~ 21 * ubatch * n_embd * 4 bytes (rounded up from ~20.6 for 0.5B, ~19.6 for 3B);
  - CUDA context ~65 MB on Windows/WDDM; budget 128 MB because Linux drivers and
    newer GPUs load more kernels up front.
"""
from __future__ import annotations

import math

from gpupool.common.models import DEFAULT_UBATCH, ModelMeta

_MB = 1024 * 1024
CONTEXT_MB = {"cuda": 128, "cpu": 32}  # runtime context / allocator slack per device
_COMPUTE_BYTES_PER_EMBD = 21 * 4  # measured factor (rounded up) x f32, per micro-batch token


def compute_mb(meta: ModelMeta, ubatch: int = DEFAULT_UBATCH, flash_attn: str = "auto",
               ctx_size: int = 0) -> int:
    """Per-device compute buffer. Scales with the micro-batch (-ub): the graph's activations are
    ubatch x n_embd wide. Without flash attention llama.cpp also materialises the f32 attention
    scores, n_head x ubatch x n_kv; ctx_size stands in for n_kv (an upper bound)."""
    b = _COMPUTE_BYTES_PER_EMBD * ubatch * meta.n_embd
    if flash_attn == "off":
        b += 4 * meta.n_head * ubatch * ctx_size
    return math.ceil(b / _MB)


def overhead_mb(meta: ModelMeta, kind: str, ubatch: int = DEFAULT_UBATCH, flash_attn: str = "auto",
                ctx_size: int = 0) -> int:
    """Memory a device needs beyond weights and KV: compute buffer + runtime context."""
    return compute_mb(meta, ubatch, flash_attn, ctx_size) + CONTEXT_MB[kind]


# Bytes per KV element: ggml block layouts (q8_0: 32 values + f16 scale = 34 B; q4_0: 16 B + scale = 18 B).
# Measured on Qwen2.5-3B ctx 8192: q8_0 -132 MB, q4_0 -204 MB vs f16; theory -142 / -217 MB.
_KV_BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}


def kv_bytes_per_layer(meta: ModelMeta, ctx_size: int, cache_type: str = "f16") -> int:
    """KV bytes of one full-attention layer by the plain rule (n_head_kv x head_dim, K and V)."""
    return math.ceil(2 * ctx_size * meta.n_head_kv * meta.head_dim * _KV_BYTES_PER_ELEM[cache_type])


def _pad(n: int, to: int = 256) -> int:
    return (n + to - 1) // to * to


def _main_layers(meta: ModelMeta) -> int:
    return meta.n_layers - meta.n_nextn


def layer_kv_bytes(meta: ModelMeta, i: int, ctx_size: int, cache_type: str = "f16",
                   parallel: int = 1, ubatch: int = DEFAULT_UBATCH, mtp: bool = False) -> int:
    """Cache bytes of layer i: KV rows (sliding-window layers hold only their window) plus the
    recurrent state of every sequence. Mirrors llama.cpp b11342: llama-kv-cache-iswa.cpp sizes a
    SWA layer at pad(min(n_ctx_seq, n_swa + n_ubatch), 256) cells per sequence stream."""
    if i >= _main_layers(meta) and not mtp:
        return 0  # MTP blocks: no cache in the target context
    if meta.kv_k is None or meta.kv_v is None:
        return kv_bytes_per_layer(meta, ctx_size, cache_type)
    cells = ctx_size
    if meta.swa and meta.swa[i] and meta.n_swa > 0:
        n_seq = max(1, parallel)
        per_seq = _pad(math.ceil(ctx_size / n_seq))
        cells = min(cells, n_seq * _pad(min(per_seq, meta.n_swa + ubatch)))
    total = cells * (meta.kv_k[i] + meta.kv_v[i]) * _KV_BYTES_PER_ELEM[cache_type]
    if meta.state_bytes:
        total += meta.state_bytes[i] * max(1, parallel)
    return math.ceil(total)


def layer_weight_bytes(meta: ModelMeta, i: int, mtp: bool = False) -> int:
    """Weights of layer i on its device. MTP blocks loaded only on demand (layer_bytes 0) count
    only with mtp; llama.cpp places them by the tensor split like any other layer."""
    b = meta.layer_bytes[i]
    if mtp and meta.n_nextn and i >= _main_layers(meta):
        b += math.ceil(meta.nextn_bytes / meta.n_nextn)
    return b


def kv_total_bytes(meta: ModelMeta, ctx_size: int, cache_type: str = "f16", parallel: int = 1,
                   ubatch: int = DEFAULT_UBATCH, mtp: bool = False) -> int:
    return sum(layer_kv_bytes(meta, i, ctx_size, cache_type, parallel, ubatch, mtp)
               for i in range(meta.n_layers))


def device_need_mb(meta: ModelMeta, layers: range, ctx_size: int, kind: str, is_last: bool,
                   cache_type: str = "f16", ubatch: int = DEFAULT_UBATCH,
                   flash_attn: str = "auto", parallel: int = 1, mtp: bool = False) -> int:
    total = sum(layer_weight_bytes(meta, i, mtp)
                + layer_kv_bytes(meta, i, ctx_size, cache_type, parallel, ubatch, mtp) for i in layers)
    if is_last:
        total += meta.output_bytes
    return math.ceil(total / _MB) + overhead_mb(meta, kind, ubatch, flash_attn, ctx_size)


def total_need_mb(meta: ModelMeta, ctx_size: int, cache_type: str = "f16",
                  ubatch: int = DEFAULT_UBATCH, flash_attn: str = "auto", parallel: int = 1,
                  mtp: bool = False) -> int:
    return device_need_mb(meta, range(meta.n_layers), ctx_size, "cuda", True, cache_type,
                          ubatch, flash_attn, parallel, mtp)


def draft_need_mb(draft_meta: ModelMeta, ctx_size: int, cache_type: str = "f16",
                  ubatch: int = DEFAULT_UBATCH, flash_attn: str = "auto", parallel: int = 1) -> int:
    """A speculative draft model runs whole on one CUDA device (the head's), with its own
    KV at the same ctx and its own compute buffer / runtime context."""
    return device_need_mb(draft_meta, range(draft_meta.n_layers), ctx_size, "cuda", True, cache_type,
                          ubatch, flash_attn, parallel)
