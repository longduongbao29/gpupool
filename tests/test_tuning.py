"""Tuning suggestions (coordinator/tuning.py), CUDA architecture helpers and the batch-aware estimate."""
from __future__ import annotations

import pytest

from gpupool.common.cuda import arch_name, has_tensor_cores, kernel_support, parse_cc
from gpupool.common.models import (
    DeviceAssignment, LibraryItem, ModelMeta, ModelSpec, Placement,
)
from gpupool.coordinator.tuning import draft_compatible, same_base_model, suggest
from gpupool.scheduler.estimate import compute_mb, total_need_mb
from gpupool.scheduler.placement import plan
from tests.test_scheduler_placement import Ports, dev, make_meta, node

GIB = 1024 ** 3


# ---------------------------------------------------------------- cuda
def test_parse_cc_and_arch_names():
    assert parse_cc("8.6") == (8, 6) and parse_cc("86") == (8, 6) and parse_cc("120a") == (12, 0)
    assert parse_cc(None) is None and parse_cc("x") is None
    assert [arch_name(c) for c in ("6.1", "7.0", "7.5", "8.6", "8.9", "9.0", "12.0")] == [
        "Pascal", "Volta", "Turing", "Ampere", "Ada", "Hopper", "Blackwell"]
    assert has_tensor_cores("6.1") is False and has_tensor_cores("7.0") is True and has_tensor_cores(None) is None


@pytest.mark.parametrize("cc,archs,want", [
    ("8.6", ["61", "86"], "native"),
    ("8.9", ["61", "86"], "compatible"),   # same major, older minor: SASS runs
    ("12.0", ["61", "86"], "jit"),         # only older majors: PTX compiled at load
    ("6.1", ["70", "86"], "missing"),      # every target newer than the GPU
    ("6.1", None, "unknown"), (None, ["86"], "unknown"),
])
def test_kernel_support(cc, archs, want):
    assert kernel_support(cc, archs) == want


# ---------------------------------------------------------------- estimate
def test_compute_buffer_scales_with_ubatch_and_flash_attn():
    m = make_meta(8)
    assert compute_mb(m) == compute_mb(m, 512) and compute_mb(m, 2048) >= 4 * compute_mb(m, 512) - 3
    assert compute_mb(m, 512, "off", 32768) > compute_mb(m, 512, "on", 32768) == compute_mb(m, 512)
    assert total_need_mb(m, 4096, ubatch=2048) > total_need_mb(m, 4096)


def test_plan_charges_the_spec_micro_batch():
    m = make_meta(8)
    nodes = [node("a", dev("CUDA0", 20000))]
    base = plan(m, ModelSpec(name="m", source="x", ctx_size=512), nodes, "r", Ports())
    big = plan(m, ModelSpec(name="m", source="x", ctx_size=512, ubatch=2048, batch=2048), nodes, "r", Ports())
    assert big.est_total_mb - base.est_total_mb == compute_mb(m, 2048) - compute_mb(m, 512)
    assert big.est_total_mb == total_need_mb(m, 512, ubatch=2048)


# ---------------------------------------------------------------- tuning
def meta(file_gib=8.0, arch="llama", layers=32, vocab=32000, tok="llama"):
    return ModelMeta(arch=arch, n_layers=layers, n_embd=4096, n_head=32, n_head_kv=8, head_dim=128,
                     layer_bytes=[int(file_gib * GIB / layers)] * layers, other_bytes=0, output_bytes=0,
                     file_bytes=int(file_gib * GIB), vocab_size=vocab, tokenizer_model=tok)


def pl(tier="single_gpu", devices=(("a", "CUDA0"),), mb=8000):
    asg = [DeviceAssignment(node_id=n, device_id=d, llama_device=d, layers=1, est_mb=mb) for n, d in devices]
    return Placement(model="m", replica_id="", tier=tier, head_node="a", head_port=0, assignments=asg,
                     tensor_split=[1.0] * len(asg), est_total_mb=mb, est_decode_tps=50.0)


def lib(*names):
    return [LibraryItem(name=n, path="/" + n, source="path", status="ready", created_at=0) for n in names]


def gpu(cc, did="CUDA0"):
    d = dev(did, 24000)
    return d.model_copy(update={"compute_cap": cc, "name": "GPU" + cc})


SPEC = ModelSpec(name="m", source="coordinator://big.gguf", ctx_size=4096)


async def run(spec, m, best, rank, metas=None, devices=None, **kw):
    metas = metas or {}

    async def r(s):
        return rank(s)

    async def meta_of(f):
        return metas.get(f)

    return await suggest(spec, m, best, rank=r, meta_of=meta_of, library=lib(*metas, "big.gguf"),
                         devices=devices, **kw)


def ids(tips):
    return [t["id"] for t in tips]


async def test_fits_one_gpu_suggests_parallel_ngram_and_ubatch_on_tensor_cores():
    tips = await run(SPEC, meta(), pl(), lambda s: pl(), devices={("a", "CUDA0"): gpu("8.6")})
    assert ids(tips) == ["parallel", "ngram", "ubatch"]
    par = tips[0]
    assert par["apply"] == {"parallel": 4, "ctx_size": 16384}  # each slot keeps today's 4096
    assert tips[2]["apply"] == {"ubatch": 2048, "batch": 2048} and "Ampere" in tips[2]["detail"]


async def test_no_ubatch_tip_without_tensor_cores_and_draft_n_max_lower():
    metas = {"small.gguf": meta(0.5, layers=24)}
    m = meta(12.0)
    tips = await run(SPEC, m, pl(), lambda s: pl(), metas, devices={("a", "CUDA0"): gpu("6.1")})
    assert "ubatch" not in ids(tips)
    draft = next(t for t in tips if t["id"] == "draft")
    assert draft["apply"] == {"speculative": "draft", "draft_file": "small.gguf", "draft_n_max": 4}
    # same model on tensor cores drafts more tokens
    tips = await run(SPEC, m, pl(), lambda s: pl(), metas, devices={("a", "CUDA0"): gpu("8.9")})
    assert next(t for t in tips if t["id"] == "draft")["apply"]["draft_n_max"] == 8


async def test_forced_flash_attn_and_big_ubatch_on_pascal_are_flagged():
    spec = SPEC.model_copy(update={"flash_attn": "on", "ubatch": 2048, "batch": 2048})
    tips = await run(spec, meta(), pl(), lambda s: pl(), devices={("a", "CUDA0"): gpu("6.1")})
    assert {"flash_attn_old_gpu", "ubatch_old_gpu"} <= set(ids(tips))
    assert next(t for t in tips if t["id"] == "ubatch_old_gpu")["apply"] == {"ubatch": 512}


async def test_split_model_gets_kv_and_smaller_quant_to_fit_one_gpu():
    metas = {"big-q4.gguf": meta(4.5), "other.gguf": meta(4.0, arch="qwen2")}
    split = pl("single_node", (("a", "CUDA0"), ("a", "CUDA1")))

    def rank(s):
        if s.kv_cache_type == "q8_0" or s.source.endswith("big-q4.gguf"):
            return pl()
        return split

    tips = await run(SPEC, meta(), split, rank, metas, max_ctx_single_gpu=None)
    assert ids(tips)[:2] == ["kv_cache", "smaller_quant"]
    assert tips[0]["apply"] == {"kv_cache_type": "q8_0"} and tips[0]["tier"] == "single_gpu"
    assert tips[1]["apply"] == {"file": "big-q4.gguf"}  # same base model only, not other.gguf


async def test_kv_tip_turns_flash_attention_back_on():
    spec = SPEC.model_copy(update={"flash_attn": "off"})
    tips = await run(spec, meta(), None, lambda s: pl() if s.kv_cache_type == "q8_0" else None)
    assert ids(tips) == ["flash_attn", "kv_cache"]
    assert tips[1]["apply"] == {"kv_cache_type": "q8_0", "flash_attn": "auto"}


async def test_nothing_fits_skips_throughput_tips_and_never_worsens_tier():
    assert await run(SPEC, meta(), None, lambda s: None) == []
    # a variant that would need more GPUs is never suggested
    tips = await run(SPEC, meta(), pl(), lambda s: pl("single_node") if s.parallel > 1 or s.ubatch > 512 else pl(),
                     devices={("a", "CUDA0"): gpu("8.6")})
    assert ids(tips) == ["ngram"]


async def test_small_ctx_per_slot_is_raised():
    spec = SPEC.model_copy(update={"parallel": 4, "ctx_size": 4096})
    tips = await run(spec, meta(), pl(), lambda s: pl())
    t = next(t for t in tips if t["id"] == "ctx_per_slot")
    assert t["apply"] == {"ctx_size": 8192}


async def test_rank_errors_are_not_suggestions():
    def rank(s):
        raise RuntimeError("planner said no")
    assert ids(await run(SPEC, meta(), pl(), rank)) == ["ngram"]


def test_draft_and_base_model_matching():
    assert draft_compatible(meta(), meta(0.5))
    assert not draft_compatible(meta(), meta(0.5, vocab=151936))
    assert not draft_compatible(meta(), meta(0.5, tok="gpt2"))
    assert same_base_model(meta(8), meta(4)) and not same_base_model(meta(), meta(layers=40))
