from typing import get_args

import pytest

from gpupool.converter.models import ConvertError, QuantType, SourceSpec
from gpupool.converter.quant import (
    QUANT_OPTIONS, check_output_name, default_output_name, estimate_bytes, estimate_vram_mb,
    imatrix_wanted, name_stem, option, plan_steps, recommend,
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
    assert types[:4] == ["BF16", "F16", "Q8_0", "Q6_K"] and types[-1] == "IQ1_S"
    assert types[2:] == list(get_args(QuantType))[2:] and set(types) == set(get_args(QuantType))
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


NEEDS = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS"}


def test_imatrix_flags_and_iq_tiers():
    assert {o.type for o in QUANT_OPTIONS if o.needs_imatrix} == NEEDS
    for t in ("IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"):
        assert option(t).tier == "tiny" and option(t).via == "quantize"
    # sizes shrink down the list below Q4: the whole list is best-first
    iq = [option(t).bpw for t in ("IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XS",
                                  "IQ2_XXS", "IQ1_M", "IQ1_S")]
    assert iq == sorted(iq, reverse=True)
    for o in QUANT_OPTIONS:
        if o.type.startswith("IQ1") or o.type.startswith("IQ2"):
            assert "importance matrix" in o.note


def test_imatrix_wanted_modes():
    # types the converter writes itself never use one, whatever the mode
    for t in ("F16", "BF16", "Q8_0"):
        assert [imatrix_wanted(t, m) for m in ("auto", "on", "off")] == [False, False, False]
    # auto: needs one, or below 4 bits per weight
    for t in NEEDS | {"Q2_K", "Q3_K_S", "IQ3_S", "IQ3_M"}:
        assert imatrix_wanted(t, "auto") is True, t
    for t in ("Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_0", "Q3_K_L", "Q3_K_M"):
        assert imatrix_wanted(t, "auto") is False, t
        assert imatrix_wanted(t, "on") is True and imatrix_wanted(t, "off") is False
    assert imatrix_wanted("IQ1_S", "off") is False  # refusing that is the job manager's call


def test_iq_types_use_the_iq4_nl_fallback_for_odd_hidden_sizes():
    cfg = {"hidden_size": 896, "vocab_size": 1000, "num_hidden_layers": 4, "tie_word_embeddings": True}
    # 896 % 256 != 0: every IQ type is stored as IQ4_NL (4.5 bpw) in the core weights
    for t in ("IQ2_XXS", "IQ1_S", "IQ3_M"):
        plain = estimate_bytes(500_000_000, t)
        assert estimate_bytes(500_000_000, t, cfg) > plain


def test_name_stem_and_inspect_default_name():
    spec = SourceSpec(hf_repo="Qwen/Qwen2.5-7B-Instruct")
    assert name_stem(spec) == "Qwen2.5-7B-Instruct"
    assert default_output_name(spec, "IQ2_M") == f"{name_stem(spec)}-IQ2_M.gguf"
    assert name_stem(SourceSpec(path="/")) == "model"


def test_recommend_mentions_iq_when_nothing_on_the_ladder_fits():
    p = 70_000_000_000
    t, why = recommend(p, _opts(set(), set(), p))
    assert t == "Q4_K_M" and any("IQ3" in r and "importance matrix" in r for r in why)


def test_iq_estimates_match_real_imatrix_quantized_files():
    # Real files written by llama.cpp b11342 (coordinator image, CI and a measured run).
    smol = {"vocab_size": 49152, "hidden_size": 576, "tie_word_embeddings": True}
    qwen15 = {"vocab_size": 151936, "hidden_size": 1536, "tie_word_embeddings": True}
    for params, cfg, t, real in ((134_515_008, smol, "IQ2_XS", 84_573_088),
                                 (1_543_714_304, qwen15, "IQ3_M", 776_663_904),
                                 (134_515_008, smol, "Q8_0", 144_810_912)):
        est = estimate_bytes(params, t, cfg)
        assert abs(est - real) / real < 0.08, (t, est, real)


def test_estimates_hold_from_135m_to_72b():
    # Real Q4_K_M files: bartowski's Hugging Face builds (sizes from the HF API) and our own runs.
    def cfg(vocab, hidden, tied, inter=None, layers=None):
        c = {"vocab_size": vocab, "hidden_size": hidden, "tie_word_embeddings": tied}
        if inter:
            c.update(intermediate_size=inter, num_hidden_layers=layers)
        return c
    cases = [
        (134_515_008, cfg(49152, 576, True), 105_453_984),                      # SmolLM2-135M (ours)
        (494_032_768, cfg(151936, 896, True), 397_807_488),                     # Qwen2.5-0.5B (ours)
        (7_615_616_512, cfg(152064, 3584, False, 18944, 28), 4_683_074_240),    # Qwen2.5-7B
        (8_030_261_248, cfg(128256, 4096, False, 14336, 32), 4_920_000_000),    # Llama-3-8B (llama.cpp's table)
        (32_763_876_352, cfg(152064, 5120, False, 27648, 64), 19_851_336_576),  # Qwen2.5-32B
        (70_553_706_496, cfg(128256, 8192, False, 28672, 80), 42_520_398_816),  # Llama-3.3-70B
        (72_706_203_648, cfg(152064, 8192, False, 29568, 80), 47_415_715_488),  # Qwen2.5-72B (ffn fallback)
    ]
    for params, c, real in cases:
        est = estimate_bytes(params, "Q4_K_M", c)
        assert abs(est - real) / real < 0.05, (params, est, real)
