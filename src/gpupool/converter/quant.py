"""Quantization types: static facts, size/VRAM estimates, the recommended type, output naming."""
from __future__ import annotations

import re
from typing import Any

from gpupool.converter.models import (
    ConvertError, ImatrixMode, Intermediate, QuantOption, QuantType, SourceSpec,
)

# bpw comes from llama-quantize's published Llama-3-8B sizes (8.03B parameters). Those sizes are
# GiB (llama.cpp prints binary units): bpw = GiB * 2**30 * 8 / 8.03e9. Q4_K_M 4.58 GiB -> 4.90.
_E = "ppl delta vs F16 on Llama-3-8B"
# The IQ rows are different: llama-quantize's own descriptions give the nominal format size
# (IQ3_M 3.66, IQ3_S 3.44, IQ3_XS 3.3, IQ3_XXS 3.06, IQ2_M 2.7, IQ2_S 2.5, IQ2_XS 2.31, IQ2_XXS 2.06,
# IQ1_M 1.75, IQ1_S 1.56 bpw), but whole files are bigger because llama-quantize keeps the output
# matrix and the most sensitive tensors at higher types. The table's bpw is a whole-model average
# (see above), so these rows use the whole-file sizes community quantizers publish for Llama-3-8B
# (IQ1_S 2.0 GB ... IQ3_M 3.8 GB); the nominal figure is quoted in the note.
_IQ_NEEDS = "needs an importance matrix (gpupool computes one)"
_RAW: list[tuple[str, float, str, str, str]] = [
    ("BF16", 16.0, "lossless", "Original precision, no quality loss. Largest file.", "convert"),
    ("F16", 16.0, "lossless", "Half precision, practically no quality loss. Largest file.", "convert"),
    ("Q8_0", 8.52, "near_lossless", f"+0.0026 {_E}. Indistinguishable from F16 in practice.", "convert"),
    ("Q6_K", 6.57, "near_lossless", f"+0.0217 {_E}. Very close to the original.", "quantize"),
    ("Q5_K_M", 5.70, "balanced", f"+0.0569 {_E}. Excellent quality, a good default for small models.", "quantize"),
    ("Q5_K_S", 5.57, "balanced", f"+0.1049 {_E}. Slightly smaller and lossier than Q5_K_M.", "quantize"),
    ("Q4_K_M", 4.90, "balanced", f"+0.1754 {_E}. The usual size/quality sweet spot.", "quantize"),
    ("Q4_K_S", 4.67, "small", f"+0.2689 {_E}. Smaller than Q4_K_M, a little lossier.", "quantize"),
    ("IQ4_XS", 4.25, "small", "About 4.25 bits per weight, close to Q4_K_S quality at a smaller size.", "quantize"),
    ("Q4_0", 4.64, "small", f"+0.4685 {_E}. Legacy format, Q4_K_S is usually better.", "quantize"),
    ("Q3_K_L", 4.31, "small", f"+0.5562 {_E}. Noticeable quality loss.", "quantize"),
    ("Q3_K_M", 4.00, "tiny", f"+0.6569 {_E}. Clear quality loss; only when memory is tight.", "quantize"),
    ("IQ3_M", 3.76, "small", "3.66 bits per weight in the format; usually better than Q3_K_M at a smaller "
     "size. An importance matrix helps (gpupool computes one).", "quantize"),
    ("IQ3_S", 3.67, "small", "3.44 bits per weight in the format; better than Q3_K_S at a similar size. "
     "An importance matrix helps (gpupool computes one).", "quantize"),
    ("Q3_K_S", 3.65, "tiny", f"+1.6321 {_E}. Large quality loss.", "quantize"),
    ("IQ3_XS", 3.50, "tiny", f"3.3 bits per weight in the format; {_IQ_NEEDS}. Clear quality loss.", "quantize"),
    ("IQ3_XXS", 3.26, "tiny", f"3.06 bits per weight in the format; {_IQ_NEEDS}. Large quality loss.", "quantize"),
    ("Q2_K", 3.17, "tiny", f"+3.5199 {_E}. Severe quality loss; a last resort.", "quantize"),
    ("IQ2_M", 2.94, "tiny", f"2.7 bits per weight in the format; {_IQ_NEEDS}. Heavy quality loss, "
     "for when nothing larger fits.", "quantize"),
    ("IQ2_S", 2.75, "tiny", f"2.5 bits per weight in the format; {_IQ_NEEDS}. Heavy quality loss, "
     "for when nothing larger fits.", "quantize"),
    ("IQ2_XS", 2.60, "tiny", f"2.31 bits per weight in the format; {_IQ_NEEDS}. Heavy quality loss, "
     "for when nothing larger fits.", "quantize"),
    ("IQ2_XXS", 2.39, "tiny", f"2.06 bits per weight in the format; {_IQ_NEEDS}. Heavy quality loss, "
     "for when nothing larger fits.", "quantize"),
    ("IQ1_M", 2.15, "tiny", f"1.75 bits per weight in the format; {_IQ_NEEDS}. Extreme quality loss; "
     "only for very large models that must fit.", "quantize"),
    ("IQ1_S", 2.01, "tiny", f"1.56 bits per weight in the format; {_IQ_NEEDS}. Extreme quality loss; "
     "only for very large models that must fit.", "quantize"),
]

# llama-quantize b11342 refuses these without an importance matrix (IQ2_M and IQ3_XS files contain
# IQ2_XS / IQ3_XXS tensors, which it also refuses).
_NEEDS_IMATRIX = frozenset({"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS"})
# Below this many bits per weight an importance matrix noticeably helps: auto mode computes one.
IMATRIX_AUTO_BPW = 4.0

QUANT_OPTIONS: list[QuantOption] = [
    QuantOption(type=t, bpw=bpw, tier=tier, note=note, via=via,  # type: ignore[arg-type]
                needs_imatrix=t in _NEEDS_IMATRIX)
    for t, bpw, tier, note, via in _RAW
]
_BY_TYPE = {o.type: o for o in QUANT_OPTIONS}

_SMALL_MODEL = 3_000_000_000
_MID_MODEL = 15_000_000_000
_LADDER: list[QuantType] = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
_SMALL_LADDER: list[QuantType] = ["Q8_0", "Q6_K", "Q5_K_M"]  # never below Q5_K_M for small models
_RUNTIME_OVERHEAD_MB = 300


def option(t: QuantType) -> QuantOption:
    return _BY_TYPE[t]


def imatrix_wanted(t: QuantType, mode: ImatrixMode) -> bool:
    """Whether a job of type `t` computes an importance matrix under `mode`.

    Types the converter writes directly (F16/BF16/Q8_0) never do: there is no llama-quantize step
    to feed one into. Otherwise on = yes, off = no, auto = yes when the type needs one or is below
    4 bits per weight. (off for a type that needs one is refused by the job manager, not here.)"""
    o = _BY_TYPE[t]
    if o.via == "convert" or mode == "off":
        return False
    if mode == "on":
        return True
    return o.needs_imatrix or o.bpw < IMATRIX_AUTO_BPW


# llama-quantize keeps the token embedding / output matrices near 8 bits in the low-bit types.
# The table's bpw is a whole-model average measured on Llama-3-8B, whose embeddings are ~13 % of
# the weights; models with a big vocabulary and few layers (Qwen2.5-0.5B: 28 %) came out ~25 %
# larger than that average predicts. So the embedding part is costed separately when known.
_EMBED_BPW = 8.5
_REF_PARAMS = 8.03e9
_REF_EMBED = 128_256 * 4096 * 2  # Llama-3-8B: vocab x hidden, untied input + output


def _core_bpw(t: QuantType) -> float:
    """bpw of the non-embedding weights, derived from the whole-model reference average."""
    bpw = _BY_TYPE[t].bpw
    if bpw >= _EMBED_BPW:
        return bpw
    return (bpw * _REF_PARAMS - _EMBED_BPW * _REF_EMBED) / (_REF_PARAMS - _REF_EMBED)


def embedding_params(config: dict) -> int:
    """vocab x hidden, twice when input and output embeddings are not tied; 0 when unknown."""
    cfg = {**config, **config["text_config"]} if isinstance(config.get("text_config"), dict) else config
    vocab = _cfg_int(cfg, "vocab_size", "padded_vocab_size")
    hidden = _cfg_int(cfg, "hidden_size", "n_embd", "d_model")
    if not (vocab and hidden):
        return 0
    tied = cfg.get("tie_word_embeddings", config.get("tie_word_embeddings", True))
    return vocab * hidden * (1 if tied else 2)


# K-quants (and IQ4_XS) need rows that are a multiple of 256 values. When the hidden size is not
# (Qwen2.5-0.5B: 896, SmolLM2-135M: 576) llama-quantize falls back to a legacy type per tensor:
# Q2_K/Q3_K/IQ4_XS -> IQ4_NL, Q4_K -> Q5_0, Q5_K -> Q5_1, Q6_K -> Q8_0. bpw of those fallbacks:
_FALLBACK_BPW: dict[str, float] = {
    "Q2_K": 4.5, "Q3_K_S": 4.5, "Q3_K_M": 4.5, "Q3_K_L": 4.5, "IQ4_XS": 4.5,
    "Q4_K_S": 5.5, "Q4_K_M": 5.5, "Q5_K_S": 6.0, "Q5_K_M": 6.0, "Q6_K": 8.5,
    # tensor_type_fallback in llama-quant.cpp: every 256-block IQ type (IQ1_*, IQ2_*, IQ3_*, IQ4_XS)
    # falls back to IQ4_NL, 4.5 bits per weight.
    "IQ3_M": 4.5, "IQ3_S": 4.5, "IQ3_XS": 4.5, "IQ3_XXS": 4.5, "IQ2_M": 4.5, "IQ2_S": 4.5,
    "IQ2_XS": 4.5, "IQ2_XXS": 4.5, "IQ1_M": 4.5, "IQ1_S": 4.5,
}


def estimate_bytes(params: int, t: QuantType, config: dict | None = None) -> int:
    """Estimated GGUF size. With the model's config.json the embeddings and llama-quantize's
    fallback for rows not divisible by 256 are costed; without it, the whole-model average."""
    bpw = _BY_TYPE[t].bpw
    if not config or bpw >= _EMBED_BPW:
        return int(params * bpw / 8)
    embed = embedding_params(config)
    if embed >= params:
        embed = 0
    core = _core_bpw(t) if embed else bpw
    cfg = {**config, **config["text_config"]} if isinstance(config.get("text_config"), dict) else config
    hidden = _cfg_int(cfg, "hidden_size", "n_embd", "d_model")
    if hidden and hidden % 256 and t in _FALLBACK_BPW:
        core = max(core, _FALLBACK_BPW[t])
    return int(((params - embed) * core + embed * _EMBED_BPW) / 8)


def _cfg_int(config: dict, *keys: str) -> int | None:
    for k in keys:
        v = config.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def estimate_vram_mb(file_bytes: int, config: dict, ctx: int = 4096) -> int:
    """GGUF file + f16 KV cache at `ctx` + runtime overhead, in MB (1 MB = 2**20 bytes)."""
    file_mb = file_bytes / (1024 * 1024)
    cfg: dict[str, Any] = config
    if isinstance(config.get("text_config"), dict):
        cfg = {**config, **config["text_config"]}  # multimodal wrappers nest the LLM's config
    layers = _cfg_int(cfg, "num_hidden_layers", "n_layer", "num_layers")
    heads = _cfg_int(cfg, "num_attention_heads", "n_head")
    hidden = _cfg_int(cfg, "hidden_size", "n_embd", "d_model")
    kv_heads = _cfg_int(cfg, "num_key_value_heads") or heads
    head_dim = _cfg_int(cfg, "head_dim") or (hidden // heads if hidden and heads else None)
    if not (layers and kv_heads and head_dim):
        return int(file_mb * 1.1 + _RUNTIME_OVERHEAD_MB)
    kv_bytes = 2 * layers * ctx * kv_heads * head_dim * 2
    return int(file_mb + kv_bytes / (1024 * 1024) + _RUNTIME_OVERHEAD_MB)


def plan_steps(t: QuantType, intermediate: Intermediate, source_dtype: str | None) -> tuple[str, str | None]:
    """-> (convert --outtype, llama-quantize type or None)."""
    if t == "F16":
        return "f16", None
    if t == "BF16":
        return "bf16", None
    if t == "Q8_0":
        return "q8_0", None
    inter = intermediate
    if inter == "auto":
        inter = "bf16" if source_dtype == "bfloat16" else "f16"
    return inter, t


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


def name_stem(spec: SourceSpec) -> str:
    """Sanitized last component of the repo / folder: the part of the default output name before
    "-<QUANT>.gguf"."""
    raw = spec.hf_repo if spec.hf_repo is not None else (spec.path or "")
    parts = [p for p in raw.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    last = parts[-1] if parts else "model"
    return _UNSAFE.sub("-", last).strip(".-_") or "model"


def default_output_name(spec: SourceSpec, t: QuantType) -> str:
    return f"{name_stem(spec)}-{t}.gguf"


def check_output_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.gguf", name or "") or name.startswith("."):
        raise ConvertError("output name must be a plain file name ending in .gguf "
                           "(letters, digits, '.', '_', '-')", 400)
    if _SPLIT_RE.search(name):
        raise ConvertError("output name must not look like a split part (-00001-of-00002.gguf)", 400)
    return name


def _fits(o: QuantOption, scope: str) -> bool:
    return (o.fits_single_gpu if scope == "single" else o.fits_pool) is True


def _pick(options: dict[str, QuantOption], ladder: list[QuantType]) -> tuple[QuantType, str] | None:
    for scope in ("single", "pool"):
        for t in ladder:
            if t in options and _fits(options[t], scope):
                return t, scope
    return None


def recommend(params: int | None, options: list[QuantOption]) -> tuple[QuantType, list[str]]:
    if params is None:
        return "Q4_K_M", ["The parameter count is unknown, so Q4_K_M, the usual size/quality balance, is suggested."]
    small = params < _SMALL_MODEL
    by = {o.type: o for o in options}
    has_fit_info = any(o.fits_single_gpu is not None or o.fits_pool is not None for o in options)

    if not has_fit_info:
        if small:
            return "Q8_0", ["Small models lose quality fastest when quantized, so Q8_0 keeps them close to the original."]
        if params < _MID_MODEL:
            return "Q5_K_M", ["Q5_K_M keeps quality high for a model of this size at a moderate file size."]
        return "Q4_K_M", ["Q4_K_M is the usual size/quality balance for large models."]

    reasons: list[str] = []
    ladder = _SMALL_LADDER if small else _LADDER
    hit = _pick(by, ladder)
    if hit is None and small:
        hit = _pick(by, _LADDER)
        if hit is not None:
            reasons.append("Small models lose quality fastest when quantized, but nothing "
                           "above Q4_K_M fits the cluster right now.")
    if hit is None:
        reasons.append("Q4_K_M does not fit the current cluster. Lower types (Q3_K_M, Q2_K) "
                       "are listed as options and lose noticeable quality. The IQ3 and IQ2 types "
                       "are smaller still; they need an importance matrix, which gpupool computes "
                       "automatically (it makes the conversion slower).")
        return "Q4_K_M", reasons
    t, scope = hit
    if scope == "single":
        reasons.append(f"{t} is the best of Q8_0, Q6_K, Q5_K_M and Q4_K_M that fits on a single GPU.")
    else:
        reasons.append(f"{t} is the best of Q8_0, Q6_K, Q5_K_M and Q4_K_M that fits the GPU pool. "
                       "It does not fit one GPU, so it will be split over RPC, which is slower.")
    if small and t in _SMALL_LADDER:
        reasons.append("Small models are never defaulted below Q5_K_M because their quality drops fastest.")
    return t, reasons
