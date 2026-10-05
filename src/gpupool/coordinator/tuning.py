"""Tuning suggestions for a model's launch settings (the "Recommend" panel of the deploy form).

Each suggestion is checked against the real pool with the same ranker a launch uses, so it never
proposes something that would not fit, or that would push the model onto more GPUs than today.
Suggestions are fields of the deploy request ("apply") the UI can copy into the form as they are.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence

from gpupool.common.cuda import arch_name, has_tensor_cores, parse_cc
from gpupool.common.models import DEFAULT_UBATCH, Device, LibraryItem, ModelMeta, ModelSpec, Placement
from gpupool.scheduler.estimate import compute_mb, kv_total_bytes

log = logging.getLogger(__name__)

COORD_PREFIX = "coordinator://"
GIB = 1024 ** 3
_TIER_ORDER = {"single_gpu": 0, "single_node": 1, "multi_node": 2}
# A draft pays off when the target is big; below this a draft costs about what it saves.
DRAFT_MIN_TARGET_BYTES = 4 * GIB
DRAFT_MAX_RATIO = 1 / 6  # a draft bigger than this share of the target saves too little
DRAFT_IDEAL_RATIO = 1 / 15
MAX_VOCAB_DIFF = 128  # same limit as the API's draft check
PARALLEL_STEPS = (4, 2)
MIN_CTX_PER_SLOT = 2048
UBATCH_STEPS = (2048, 1024)
_KV_LEVEL = {"f16": 0, "q8_0": 1, "q4_0": 2}

Rank = Callable[[ModelSpec], Awaitable[Placement | None]]
MetaOf = Callable[[str], Awaitable[ModelMeta | None]]


def _tier(p: Placement | None) -> int:
    return 99 if p is None else _TIER_ORDER.get(p.tier, 3)


def _file(spec_source: str | None) -> str | None:
    if spec_source and spec_source.startswith(COORD_PREFIX):
        return spec_source[len(COORD_PREFIX):]
    return None


def _tip(tid: str, kind: str, title: str, detail: str, apply: dict, p: Placement | None) -> dict:
    return {"id": tid, "kind": kind, "title": title, "detail": detail, "apply": apply,
            "tier": p.tier if p else None, "est_decode_tps": p.est_decode_tps if p else None,
            "est_total_mb": p.est_total_mb if p else None}


def _mb(n: float) -> str:
    return f"{n / 1024:.1f} GB" if n >= 1024 else f"{round(n)} MB"


def draft_compatible(meta: ModelMeta, dmeta: ModelMeta) -> bool:
    if meta.tokenizer_model and dmeta.tokenizer_model and meta.tokenizer_model != dmeta.tokenizer_model:
        return False
    if meta.vocab_size and dmeta.vocab_size and abs(meta.vocab_size - dmeta.vocab_size) > MAX_VOCAB_DIFF:
        return False
    return bool(meta.vocab_size and dmeta.vocab_size) or bool(meta.tokenizer_model and dmeta.tokenizer_model)


def same_base_model(a: ModelMeta, b: ModelMeta) -> bool:
    """Two quantizations of one model: same architecture and shape."""
    return (a.arch, a.n_layers, a.n_embd, a.n_head, a.n_head_kv) == (b.arch, b.n_layers, b.n_embd, b.n_head, b.n_head_kv)


def gpus_of(p: Placement | None, devices: Mapping[tuple[str, str], Device]) -> list[Device]:
    if p is None:
        return []
    out = [devices.get((a.node_id, a.device_id)) for a in p.assignments]
    return [d for d in out if d is not None and d.kind == "cuda"]


def _describe(gpus: list[Device]) -> str:
    names = sorted({f"{d.name} ({arch_name(d.compute_cap) or 'unknown arch'}, cc {d.compute_cap})"
                    for d in gpus})
    return ", ".join(names)


async def suggest(spec: ModelSpec, meta: ModelMeta, best: Placement | None, *, rank: Rank,
                  meta_of: MetaOf, library: Sequence[LibraryItem],
                  max_ctx_single_gpu: int | None = None,
                  devices: Mapping[tuple[str, str], Device] | None = None) -> list[dict]:
    """Suggestions for `spec`, most useful first. `best` is the top placement for `spec` as it is
    (None when nothing fits). `rank(spec)` returns the top placement of a variant or None.
    `devices` ((node_id, device_id) -> Device) lets the advice follow the GPU generation of the
    placement: tensor cores (Volta, cc 7.0+) decide whether big micro-batches and forced flash
    attention pay off."""
    tips: list[dict] = []
    devices = devices or {}
    gpus = gpus_of(best, devices)
    known = [d for d in gpus if parse_cc(d.compute_cap) is not None]
    # Tensor cores on every GPU of the placement (None: generation unknown, old agents).
    tensor = None if not known or len(known) < len(gpus) else all(has_tensor_cores(d.compute_cap) for d in known)
    old_gpus = [d for d in known if not has_tensor_cores(d.compute_cap)]
    own_file = _file(spec.source)
    ready = [i for i in library if i.status == "ready" and i.name != own_file]

    async def try_rank(s: ModelSpec) -> Placement | None:
        try:
            return await rank(s)
        except Exception as e:  # a variant the planner rejects is simply not suggested
            log.debug("tuning variant rejected: %s", e)
            return None

    async def metas() -> list[tuple[LibraryItem, ModelMeta]]:
        out = []
        for item in ready:
            try:
                m = await meta_of(item.name)
            except Exception:
                m = None
            if m is not None:
                out.append((item, m))
        return out

    lib_metas: list[tuple[LibraryItem, ModelMeta]] | None = None

    # -- flash attention off: more memory, slower attention. Nothing to gain on GPUs that support it.
    if spec.flash_attn == "off":
        v = spec.model_copy(update={"flash_attn": "auto"})
        p = await try_rank(v)
        saved = compute_mb(meta, spec.ubatch, "off", spec.ctx_size) - compute_mb(meta, spec.ubatch)
        tips.append(_tip("flash_attn", "speed", "Turn flash attention back on (auto)",
                         f"Faster attention on long contexts and about {_mb(saved)} less compute memory "
                         "per GPU. Auto keeps it off on the rare GPUs that lack support.",
                         {"flash_attn": "auto"}, p))

    if spec.flash_attn == "on" and old_gpus:
        p = await try_rank(spec.model_copy(update={"flash_attn": "auto"}))
        tips.append(_tip("flash_attn_old_gpu", "fix", "Let llama.cpp decide flash attention (auto)",
                         f"{_describe(old_gpus)} has no tensor cores: flash attention runs on slower "
                         "fallback kernels there and some head sizes are not covered at all. Auto turns "
                         "it on only where it is supported.", {"flash_attn": "auto"}, p))

    # -- one GPU instead of several: layers on different GPUs run one after another, so a split
    #    never decodes faster than one GPU; it costs hops (and RPC round trips across servers).
    if best is None or best.tier != "single_gpu":
        goal = "fit on a single GPU" if best is not None else "fit in the pool"
        for kv in ("q8_0", "q4_0"):
            if _KV_LEVEL[kv] <= _KV_LEVEL[spec.kv_cache_type]:
                continue  # only ever suggest a smaller cache than today's
            fa = {"flash_attn": "auto"} if spec.flash_attn == "off" else {}  # quantized V needs FA
            p = await try_rank(spec.model_copy(update={"kv_cache_type": kv, **fa}))
            if p is not None and _tier(p) < _tier(best):
                kw = {"parallel": spec.parallel, "ubatch": spec.ubatch}
                saved = (kv_total_bytes(meta, spec.ctx_size, spec.kv_cache_type, **kw)
                         - kv_total_bytes(meta, spec.ctx_size, kv, **kw)) / 1024 ** 2
                quality = "negligible quality loss" if kv == "q8_0" else "a small quality loss"
                tips.append(_tip("kv_cache", "fix", f"Quantize the KV cache to {kv} to {goal}",
                                 f"Saves about {_mb(saved)} of KV memory with {quality}; "
                                 f"the model then runs as {p.tier.replace('_', ' ')}.",
                                 {"kv_cache_type": kv, **fa}, p))
                break
        lib_metas = await metas()
        smaller = sorted(((i, m) for i, m in lib_metas
                          if same_base_model(meta, m) and m.file_bytes and meta.file_bytes
                          and m.file_bytes < meta.file_bytes),
                         key=lambda im: -(im[1].file_bytes or 0))  # best quality first
        for item, m in smaller:
            p = await try_rank(spec.model_copy(update={"source": COORD_PREFIX + item.name}))
            if p is not None and _tier(p) < _tier(best):
                tips.append(_tip("smaller_quant", "fix", f"Use the smaller quantization {item.name}",
                                 f"Same model, {_mb((meta.file_bytes - m.file_bytes) / 1024 ** 2)} smaller: "
                                 f"runs as {p.tier.replace('_', ' ')}, and decode speed grows with "
                                 "fewer bytes per token. Check the quality is acceptable.",
                                 {"file": item.name}, p))
                break
        if (best is not None and max_ctx_single_gpu and max_ctx_single_gpu < spec.ctx_size
                and max_ctx_single_gpu >= MIN_CTX_PER_SLOT * spec.parallel):
            p = await try_rank(spec.model_copy(update={"ctx_size": max_ctx_single_gpu}))
            if p is not None and p.tier == "single_gpu":
                tips.append(_tip("ctx_single", "fix", f"Lower the context to {max_ctx_single_gpu} to fit on a single GPU",
                                 f"{max_ctx_single_gpu // spec.parallel} tokens per slot instead of "
                                 f"{spec.ctx_size // spec.parallel}.",
                                 {"ctx_size": max_ctx_single_gpu}, p))

    if best is None:
        return tips  # the rest tunes a model that fits

    # -- parallel slots: continuous batching serves several requests in one pass over the weights.
    per_slot = spec.ctx_size // max(1, spec.parallel)
    if spec.parallel == 1:
        for n in PARALLEL_STEPS:
            v = spec.model_copy(update={"parallel": n, "ctx_size": spec.ctx_size * n})
            p = await try_rank(v)
            if p is not None and _tier(p) <= _tier(best):
                extra = (p.est_total_mb - best.est_total_mb) if p.est_total_mb and best.est_total_mb else None
                tips.append(_tip("parallel", "throughput", f"Serve {n} requests at once ({n} slots)",
                                 f"Each slot keeps {per_slot} tokens of context (total {spec.ctx_size * n}). "
                                 "Total tokens/s grows almost linearly with concurrent users while one "
                                 "request alone slows only slightly."
                                 + (f" Costs about {_mb(extra)} more KV memory." if extra and extra > 0 else ""),
                                 {"parallel": n, "ctx_size": spec.ctx_size * n}, p))
                break
    if spec.parallel > 1 and not spec.kv_unified:
        # one shared KV pool: a long request may take the whole context while others are short
        p = await try_rank(spec.model_copy(update={"kv_unified": True}))
        if p is not None and _tier(p) <= _tier(best):
            tips.append(_tip("kv_unified", "fix", "Share the context between the slots",
                             f"Today each of the {spec.parallel} slots owns {per_slot} tokens. A shared KV "
                             f"pool lets any one request use up to {spec.ctx_size} tokens while the others "
                             "are short, at the same memory.", {"kv_unified": True}, p))
    if spec.parallel > 1 and per_slot < MIN_CTX_PER_SLOT and not spec.kv_unified:
        want = MIN_CTX_PER_SLOT * spec.parallel
        p = await try_rank(spec.model_copy(update={"ctx_size": want}))
        if p is not None and _tier(p) <= _tier(best):
            tips.append(_tip("ctx_per_slot", "fix", f"Raise the context to {want}",
                             f"The context is divided across {spec.parallel} slots: each request gets only "
                             f"{per_slot} tokens today, so long prompts fail or get truncated.",
                             {"ctx_size": want}, p))

    # -- MTP: the model's own multi-token-prediction blocks draft the next tokens. No second file,
    #    no tokenizer to match, and the drafts come from the model itself (high acceptance).
    if meta.n_nextn and spec.speculative in ("none", "ngram"):
        v = spec.model_copy(update={"speculative": "mtp", "draft": None, "draft_n_max": 3})
        p = await try_rank(v)
        if p is not None and _tier(p) <= _tier(best):
            hops = " and fewer network round trips" if best is not None and best.tier == "multi_node" else ""
            tips.append(_tip("mtp", "speed", "Speculative decoding with the model's own MTP blocks",
                             "This GGUF ships multi-token-prediction layers: llama.cpp drafts with them, "
                             f"usually 1.5-2x faster generation{hops}; costs about "
                             f"{_mb(meta.nextn_bytes / 1024 ** 2)} plus their KV, no extra model file.",
                             {"speculative": "mtp", "draft_n_max": 3}, p))

    # -- speculative decoding: a small model of the same family drafts tokens the big one verifies
    #    in one pass. Wins most on big models and when the model spans servers (fewer round trips).
    if (spec.speculative not in ("draft", "mtp") and not any(t["id"] == "mtp" for t in tips)
            and meta.file_bytes and meta.file_bytes >= DRAFT_MIN_TARGET_BYTES):
        if lib_metas is None:
            lib_metas = await metas()
        drafts = [(i, m) for i, m in lib_metas
                  if m.file_bytes and m.file_bytes <= meta.file_bytes * DRAFT_MAX_RATIO
                  and draft_compatible(meta, m)]
        drafts.sort(key=lambda im: abs(math.log(im[1].file_bytes / (meta.file_bytes * DRAFT_IDEAL_RATIO))))
        # Verifying a batch of drafted tokens is nearly free with tensor cores; without them each
        # extra token costs real compute, so draft fewer.
        n_max = 8 if meta.file_bytes >= 10 * GIB and tensor is not False else 4
        for item, m in drafts[:3]:
            v = spec.model_copy(update={"speculative": "draft", "draft": COORD_PREFIX + item.name,
                                        "draft_n_max": n_max})
            p = await try_rank(v)
            if p is not None and _tier(p) <= _tier(best):
                hops = " and fewer network round trips" if best.tier == "multi_node" else ""
                tips.append(_tip("draft", "speed", f"Speculative decoding with draft {item.name}",
                                 f"Same tokenizer, {round(meta.file_bytes / m.file_bytes)}x smaller. Typically "
                                 f"1.5-2.5x faster generation on code and predictable text{hops}; "
                                 f"costs about {_mb((m.file_bytes or 0) / 1024 ** 2)} plus its KV on the head GPU.",
                                 {"speculative": "draft", "draft_file": item.name, "draft_n_max": n_max}, p))
                break
    if spec.speculative == "none" and not any(t["id"] in ("draft", "mtp") for t in tips):
        big = bool(meta.file_bytes and meta.file_bytes >= DRAFT_MIN_TARGET_BYTES)
        more = (" A draft model would help more: add a small model of the same family to the library."
                if big else "")
        tips.append(_tip("ngram", "speed", "Speculative decoding with n-gram (no extra memory)",
                         "Guesses the next tokens from the text so far. Helps when outputs repeat the "
                         "input (code edits, RAG, extraction); little effect on free-form chat." + more,
                         {"speculative": "ngram"}, best))

    # -- micro-batch: prompt processing (prefill) runs ubatch tokens per pass. With tensor cores a
    #    bigger pass is faster per token; without them (Pascal and older) prefill is already
    #    compute bound and a bigger micro-batch only costs memory. Generation speed does not change.
    if spec.ubatch == DEFAULT_UBATCH and spec.flash_attn != "off" and tensor is not False:
        for ub in UBATCH_STEPS:
            v = spec.model_copy(update={"ubatch": ub, "batch": max(spec.batch, ub)})
            p = await try_rank(v)
            if p is not None and _tier(p) <= _tier(best):
                extra = compute_mb(meta, ub) - compute_mb(meta, spec.ubatch)
                where = f" on {_describe(gpus)}" if tensor else " on GPUs with tensor cores (Volta and newer)"
                tips.append(_tip("ubatch", "speed", f"Micro-batch {ub} for faster prompt processing",
                                 f"Long prompts (RAG, documents, big system prompts) are read faster{where}; "
                                 f"about {_mb(extra)} more compute memory per GPU. Generation speed is "
                                 "unchanged.",
                                 {"ubatch": ub, "batch": max(spec.batch, ub)}, p))
                break
    elif spec.ubatch > DEFAULT_UBATCH and old_gpus:
        p = await try_rank(spec.model_copy(update={"ubatch": DEFAULT_UBATCH}))
        saved = compute_mb(meta, spec.ubatch) - compute_mb(meta, DEFAULT_UBATCH)
        tips.append(_tip("ubatch_old_gpu", "fix", f"Micro-batch {DEFAULT_UBATCH} on GPUs without tensor cores",
                         f"{_describe(old_gpus)} gains little from micro-batch {spec.ubatch}; "
                         f"{DEFAULT_UBATCH} frees about {_mb(saved)} per GPU.",
                         {"ubatch": DEFAULT_UBATCH}, p))
    return tips
