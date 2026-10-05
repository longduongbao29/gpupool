"""Per-layer cache layout read from the GGUF header, checked against llama.cpp b11342's rules."""
import gguf
import numpy as np
import pytest

from gpupool.common.models import ModelMeta
from gpupool.scheduler.estimate import (
    device_need_mb, kv_bytes_per_layer, kv_total_bytes, layer_kv_bytes, total_need_mb,
)
from gpupool.scheduler.gguf_meta import read_meta
from gpupool.scheduler.scoring import est_decode_tps
from gpupool.common.models import Device


def write(tmp_path, arch, n_layers, kv=(), tensors=None, n_embd=64, heads=4, heads_kv=2):
    """A tiny GGUF of `arch`: kv = [(suffix, kind, value)] for f"{arch}.{suffix}" keys."""
    path = tmp_path / f"{arch}.gguf"
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(n_layers)
    w.add_embedding_length(n_embd)
    w.add_head_count(heads)
    if heads_kv is not None:
        if isinstance(heads_kv, list):
            w.add_array(f"{arch}.attention.head_count_kv", heads_kv)
        else:
            w.add_head_count_kv(heads_kv)
    for suffix, kind, value in kv:
        getattr(w, f"add_{kind}")(f"{arch}.{suffix}", value)
    for name, rows in (tensors or [(f"blk.{i}.attn_q.weight", 4) for i in range(n_layers)]):
        w.add_tensor(name, np.zeros((rows, 8), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return read_meta(str(path))


def test_dense_model_keeps_the_old_rule(tmp_path):
    m = write(tmp_path, "llama", 4)
    assert m.kv_k == [2 * 16] * 4 and m.kv_v == [2 * 16] * 4 and not any(m.swa)
    for i in range(4):
        assert layer_kv_bytes(m, i, 4096) == kv_bytes_per_layer(m, 4096)


def test_gemma3_sliding_window_layers_cache_only_their_window(tmp_path):
    # gemma3 loader: load_swa_pattern(ml, 6): layers 0-4 SWA, 5 full, 6-10 SWA, 11 full
    m = write(tmp_path, "gemma3", 12, kv=[("attention.sliding_window", "uint32", 1024)])
    assert m.swa == [i % 6 < 5 for i in range(12)] and m.n_swa == 1024
    full = layer_kv_bytes(m, 5, 32768)
    swa = layer_kv_bytes(m, 0, 32768, ubatch=512)
    assert full == kv_bytes_per_layer(m, 32768)
    assert swa == full * 1536 // 32768  # pad(min(32768, 1024 + 512), 256) = 1536 cells
    # several sequences: each stream keeps its own window (n_stream = n_parallel)
    assert layer_kv_bytes(m, 0, 32768, parallel=4, ubatch=512) == swa * 4
    # a short context is never padded above itself
    assert layer_kv_bytes(m, 0, 1024) == kv_bytes_per_layer(m, 1024)


def test_swa_without_a_window_key_is_full_attention(tmp_path):
    m = write(tmp_path, "gemma3", 6)
    assert not any(m.swa)


def test_explicit_pattern_array_wins(tmp_path):
    pattern = [True, False, True, False]
    m = write(tmp_path, "gemma4", 4, kv=[("attention.sliding_window", "uint32", 512),
                                        ("attention.sliding_window_pattern", "array", pattern)])
    assert m.swa == pattern


def test_unknown_architecture_with_a_window_counts_full_layers(tmp_path):
    m = write(tmp_path, "somearch", 4, kv=[("attention.sliding_window", "uint32", 512)])
    assert not any(m.swa)  # never assume SWA we cannot see: an over-estimate, not an OOM


def test_mla_caches_only_the_latent_k(tmp_path):
    m = write(tmp_path, "deepseek2", 3, heads=8, heads_kv=1,
              kv=[("attention.key_length", "uint32", 576), ("attention.value_length", "uint32", 512),
                  ("attention.key_length_mla", "uint32", 192),
                  ("attention.value_length_mla", "uint32", 128)])
    assert m.kv_k == [576] * 3 and m.kv_v == [0] * 3
    assert layer_kv_bytes(m, 0, 1000) == 1000 * 576 * 2


def test_qwen35_hybrid_has_kv_only_on_full_attention_layers(tmp_path):
    kv = [("full_attention_interval", "uint32", 4), ("ssm.conv_kernel", "uint32", 4),
          ("ssm.inner_size", "uint32", 128), ("ssm.state_size", "uint32", 16),
          ("ssm.group_count", "uint32", 2), ("nextn_predict_layers", "uint32", 1)]
    tensors = [(f"blk.{i}.attn_q.weight", 4) for i in range(9)]
    m = write(tmp_path, "qwen35", 9, kv=kv, tensors=tensors)
    attn = [i for i in range(8) if m.kv_k[i]]
    assert attn == [3, 7]  # (i + 1) % 4 == 0
    state = 4 * (3 * (128 + 2 * 2 * 16) + 16 * 128)
    assert m.state_bytes[0] == state and m.state_bytes[3] == 0
    assert layer_kv_bytes(m, 0, 4096, parallel=2) == 2 * state  # one state per sequence
    # the MTP block (layer 8): skipped by llama.cpp unless draft-mtp
    assert m.n_nextn == 1 and m.layer_bytes[8] == 0 and m.nextn_bytes == 4 * 8 * 4
    assert layer_kv_bytes(m, 8, 4096) == 0 and layer_kv_bytes(m, 8, 4096, mtp=True) > 0
    assert total_need_mb(m, 4096, mtp=True) >= total_need_mb(m, 4096)


def test_zero_kv_heads_mark_recurrent_layers(tmp_path):
    kv = [("ssm.conv_kernel", "uint32", 4), ("ssm.inner_size", "uint32", 64),
          ("ssm.state_size", "uint32", 8), ("ssm.group_count", "uint32", 1)]
    m = write(tmp_path, "nemotron_h", 4, kv=kv, heads_kv=[0, 2, 0, 2])
    assert [bool(k) for k in m.kv_k] == [False, True, False, True]
    assert m.state_bytes[0] > 0 and m.state_bytes[1] == 0


def test_moe_decode_reads_only_routed_experts(tmp_path):
    tensors = []
    for i in range(2):
        tensors += [(f"blk.{i}.attn_q.weight", 8), (f"blk.{i}.ffn_down_exps.weight", 64),
                    (f"blk.{i}.ffn_down_shexp.weight", 8)]
    m = write(tmp_path, "qwen3moe", 2, tensors=tensors,
              kv=[("expert_count", "uint32", 8), ("expert_used_count", "uint32", 2)])
    row = 8 * 4
    assert m.layer_bytes == [80 * row] * 2
    assert m.active_bytes == [(8 + 8 + 16) * row] * 2  # experts at 2/8
    gpu = Device(device_id="CUDA0", kind="cuda", name="g", total_mb=1, free_mb=1, usable_mb=1,
                 bandwidth_gbps=100.0)
    dense = m.model_copy(update={"active_bytes": None})
    assert est_decode_tps(m, [(gpu, 2)]) > est_decode_tps(dense, [(gpu, 2)])


def test_meta_without_layout_fields_keeps_the_old_estimate():
    m = ModelMeta(arch="x", n_layers=2, n_embd=64, n_head=4, n_head_kv=2, head_dim=16,
                  layer_bytes=[1 << 20] * 2, other_bytes=0, output_bytes=0)
    assert kv_total_bytes(m, 4096) == 2 * kv_bytes_per_layer(m, 4096)
    assert device_need_mb(m, range(2), 4096, "cuda", True, parallel=4) == \
        device_need_mb(m, range(2), 4096, "cuda", True)


@pytest.mark.parametrize("ct", ["f16", "q8_0", "q4_0"])
def test_quantized_cache_scales_every_layer_kind(tmp_path, ct):
    m = write(tmp_path, "gemma3", 6, kv=[("attention.sliding_window", "uint32", 1024)])
    assert layer_kv_bytes(m, 0, 32768, ct) < layer_kv_bytes(m, 5, 32768, ct)


def test_mtp_placement_reserves_the_nextn_blocks(tmp_path):
    from gpupool.common.models import ModelSpec
    from gpupool.scheduler.placement import plan
    from tests.test_coordinator_helpers import node
    kv = [("full_attention_interval", "uint32", 4), ("ssm.conv_kernel", "uint32", 4),
          ("ssm.inner_size", "uint32", 128), ("ssm.state_size", "uint32", 16),
          ("ssm.group_count", "uint32", 2), ("nextn_predict_layers", "uint32", 1)]
    tensors = [(f"blk.{i}.attn_q.weight", 4096) for i in range(9)]
    m = write(tmp_path, "qwen35", 9, kv=kv, tensors=tensors)
    gpu = Device(device_id="CUDA0", kind="cuda", name="g", total_mb=4000, free_mb=4000, usable_mb=4000)
    nodes = [node("a", devices=[gpu])]
    off = plan(m, ModelSpec(name="q", source="x", ctx_size=4096), nodes, "r", lambda n: 9000)
    on = plan(m, ModelSpec(name="q", source="x", ctx_size=4096, speculative="mtp"), nodes, "r", lambda n: 9000)
    assert on.est_total_mb > off.est_total_mb


def test_unified_kv_sizes_swa_layers_for_all_sequences_at_once(tmp_path):
    m = write(tmp_path, "gemma3", 6, kv=[("attention.sliding_window", "uint32", 1024)])
    full = layer_kv_bytes(m, 5, 32768)
    # one stream: pad(min(32768, 1024 x 4 + 512), 256) = 4608 cells instead of 4 x 1536
    assert layer_kv_bytes(m, 0, 32768, parallel=4, ubatch=512, kv_unified=True) == full * 4608 // 32768
    assert layer_kv_bytes(m, 5, 32768, parallel=4, kv_unified=True) == full  # full layers: same
