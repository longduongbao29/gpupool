import pytest

from gpupool.converter.models import ConvertError, SourceSpec
from gpupool.converter.quant import (
    QUANT_OPTIONS, check_output_name, default_output_name, estimate_bytes, estimate_vram_mb,
    option, plan_steps, recommend,
)


def _opts(single: set[str] | None, pool: set[str] | None, params: int):
    out = []
    for o in QUANT_OPTIONS:
        o = o.model_copy()
        o.est_bytes = estimate_bytes(params, o.type)
        if single is not None:
            o.fits_single_gpu = o.type in single
            o.fits_pool = o.type in (pool or set())
        out.append(o)
    return out


def test_static_options_cover_every_type_best_first():
    types = [o.type for o in QUANT_OPTIONS]
    assert types[:4] == ["BF16", "F16", "Q8_0", "Q6_K"] and types[-1] == "Q2_K"
    assert len(set(types)) == 14
    assert option("Q4_K_M").bpw == pytest.approx(4.90, abs=0.02)
    assert option("Q8_0").via == "convert" and option("Q6_K").via == "quantize"
    assert option("F16").tier == "lossless" and option("Q2_K").tier == "tiny"


def test_estimate_bytes():
    assert estimate_bytes(8_000_000_000, "F16") == 16_000_000_000
    assert 4.5e9 < estimate_bytes(8_030_000_000, "Q4_K_M") < 5.2e9


def test_plan_steps():
    assert plan_steps("F16", "auto", "bfloat16") == ("f16", None)
    assert plan_steps("BF16", "f16", None) == ("bf16", None)
    assert plan_steps("Q8_0", "f32", None) == ("q8_0", None)
    assert plan_steps("Q4_K_M", "auto", "bfloat16") == ("bf16", "Q4_K_M")
    assert plan_steps("Q4_K_M", "auto", "float32") == ("f16", "Q4_K_M")
    assert plan_steps("Q5_K_S", "f32", "bfloat16") == ("f32", "Q5_K_S")


def test_output_names():
    assert default_output_name(SourceSpec(hf_repo="Qwen/Qwen2.5-7B-Instruct"), "Q4_K_M") == "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    assert default_output_name(SourceSpec(path=r"C:\models\my model" + "\\"), "Q8_0") == "my-model-Q8_0.gguf"
    assert default_output_name(SourceSpec(path="/"), "F16") == "model-F16.gguf"
    assert check_output_name("a-b_c.1.gguf") == "a-b_c.1.gguf"
    for bad in ["", "x", "x.bin", ".x.gguf", "../x.gguf", "a/b.gguf", "a b.gguf", "m-00001-of-00002.gguf"]:
        with pytest.raises(ConvertError) as e:
            check_output_name(bad)
        assert e.value.status == 400


def test_recommend_single_gpu_best_of_ladder():
    p = 8_000_000_000
    t, why = recommend(p, _opts({"Q5_K_M", "Q4_K_M"}, {"Q6_K", "Q5_K_M", "Q4_K_M"}, p))
    assert t == "Q5_K_M" and "single GPU" in why[0]


def test_recommend_pool_mentions_rpc():
    p = 70_000_000_000
    t, why = recommend(p, _opts(set(), {"Q4_K_M"}, p))
    assert t == "Q4_K_M" and any("RPC" in r and "slower" in r for r in why)


def test_recommend_nothing_fits():
    p = 70_000_000_000
    t, why = recommend(p, _opts(set(), set(), p))
    assert t == "Q4_K_M" and "does not fit" in why[0]


def test_recommend_small_model_floor():
    p = 1_000_000_000
    # only Q4_K_M fits one GPU -> the floor does not apply when nothing >= Q5_K_M fits
    t, why = recommend(p, _opts({"Q4_K_M"}, {"Q4_K_M"}, p))
    assert t == "Q4_K_M" and "Small models" in why[0]
    # Q5_K_M fits -> chosen, never Q4_K_M
    t, _ = recommend(p, _opts({"Q5_K_M", "Q4_K_M"}, {"Q5_K_M", "Q4_K_M"}, p))
    assert t == "Q5_K_M"


def test_recommend_size_rule_without_cluster_info():
    assert recommend(1_000_000_000, _opts(None, None, 1))[0] == "Q8_0"
    assert recommend(7_000_000_000, _opts(None, None, 1))[0] == "Q5_K_M"
    assert recommend(14_999_999_999, _opts(None, None, 1))[0] == "Q5_K_M"
    assert recommend(15_000_000_000, _opts(None, None, 1))[0] == "Q4_K_M"


def test_recommend_unknown_params():
    t, why = recommend(None, list(QUANT_OPTIONS))
    assert t == "Q4_K_M" and why


def test_estimate_vram_basic_gqa():
    # Llama-3-8B shape: 32 layers, 8 kv heads, head_dim 128 -> 512 MB of KV at ctx 4096
    cfg = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096}
    assert estimate_vram_mb(1024 * 1024 * 1000, cfg) == 1000 + 512 + 300


def test_estimate_vram_head_dim_text_config_and_fallback():
    cfg = {"text_config": {"num_hidden_layers": 2, "num_attention_heads": 4, "head_dim": 256}}
    # kv heads = 4, head_dim 256: 2*2*4096*4*256*2 bytes = 32 MB
    assert estimate_vram_mb(0, cfg) == 32 + 300
    assert estimate_vram_mb(1024 * 1024 * 100, {}) == int(100 * 1.1 + 300)


def test_estimate_bytes_matches_real_llama_quantize_outputs():
    from gpupool.converter.quant import embedding_params
    # Real Q4_K_M sizes written by llama.cpp b11342 in this repo's Docker image.
    qwen = {"vocab_size": 151936, "hidden_size": 896, "tie_word_embeddings": True}
    smol = {"vocab_size": 49152, "hidden_size": 576, "tie_word_embeddings": True}
    for params, cfg, real in ((494_032_768, qwen, 397_807_488), (134_515_008, smol, 105_453_984)):
        est = estimate_bytes(params, "Q4_K_M", cfg)
        assert abs(est - real) / real < 0.08, (cfg, est, real)
        assert estimate_bytes(params, "Q4_K_M") < 0.85 * real  # the plain average was far too low
    # The reference model (rows divisible by 256) is unchanged by the embedding split.
    llama3 = {"vocab_size": 128256, "hidden_size": 4096, "tie_word_embeddings": False}
    assert abs(estimate_bytes(8_030_000_000, "Q4_K_M", llama3)
               - estimate_bytes(8_030_000_000, "Q4_K_M")) < 2e6
    assert embedding_params({"vocab_size": 10}) == 0
    assert estimate_bytes(1000, "F16", qwen) == 2000  # >= 8.5 bpw types: no split
