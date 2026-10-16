"""Which engine should serve a model: a recommendation with reasons, shown before deploying.

Pure functions. The API gathers the inputs (what the model is, which servers run which engine,
whether a placement fits right now) and this module ranks the engines:

  - a GGUF from the library: llama.cpp, the only engine gpupool runs GGUFs on;
  - a Hugging Face repo: vLLM, then SGLang (both serve it as is), or llama.cpp after a conversion to
    GGUF, which is the way to pool the GPUs of several servers when no single server fits it.

What the model is decides what is supported at all (a vision model, an AWQ checkpoint or an
embedding model is not something gpupool converts); availability and fit then order the rest.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from gpupool.common.models import Placement
from gpupool.converter.source import PREQUANT_SUPPORTED
from gpupool.scheduler.hf_meta import has_vision

ENGINE_LABELS = {"llama.cpp": "llama.cpp", "vllm": "vLLM", "sglang": "SGLang"}
ENV_OF = {"vllm": "GPUPOOL_VLLM_CMD", "sglang": "GPUPOOL_SGLANG_CMD"}

# Base preference among supported engines; availability and fit move them from there.
_BASE = {"llama.cpp": 100, "vllm": 100, "sglang": 95}
_CONVERT_BASE = 80  # llama.cpp for a Hugging Face repo: a conversion first
_NOT_INSTALLED = 50
_NO_FIT = 30


@dataclass
class ModelFacts:
    kind: str  # "gguf" | "hf"
    task: str = "generate"  # "generate" | "embedding" | "rerank"
    vision: bool = False
    quant: str | None = None  # quantization_config.quant_method ("awq", "gptq", "fp8"...)
    arch: str | None = None


def facts_from_config(config: Mapping, files: Iterable[str], repo: str = "") -> ModelFacts:
    """What a Hugging Face repo holds, from its config.json and file names."""
    names = set(files)
    archs = [str(a) for a in config.get("architectures") or [] if isinstance(a, str)]
    qc = config.get("quantization_config")
    quant = str(qc.get("quant_method")).lower() if isinstance(qc, dict) and qc.get("quant_method") else None
    task = "generate"
    generative = any(a.endswith(("ForCausalLM", "ForConditionalGeneration", "LMHeadModel")) for a in archs)
    if any(a.endswith("ForSequenceClassification") for a in archs) and not generative:
        task = "rerank" if "rerank" in repo.lower() or config.get("num_labels") in (1, None) else "embedding"
    elif "modules.json" in names or (archs and not generative and any(a.endswith("Model") for a in archs)):
        task = "embedding"  # sentence-transformers layout, or a bare encoder (BertModel, XLMRobertaModel...)
    return ModelFacts(kind="hf", task=task, vision=has_vision(dict(config)), quant=quant,
                      arch=archs[0] if archs else str(config.get("model_type") or "") or None)


@dataclass
class EngineOption:
    engine: str
    supported: bool
    score: int = 0
    servers: list[str] = field(default_factory=list)  # servers that run the engine
    fits_now: bool | None = None  # None: not checked (unsupported, not installed, or a conversion first)
    needs_conversion: bool = False
    placement: Placement | None = None
    reasons: list[str] = field(default_factory=list)  # why it is a good choice
    warnings: list[str] = field(default_factory=list)  # what stands in the way

    def as_dict(self) -> dict:
        p = self.placement
        return {
            "engine": self.engine, "label": ENGINE_LABELS[self.engine], "supported": self.supported,
            "score": self.score, "servers": self.servers, "fits_now": self.fits_now,
            "needs_conversion": self.needs_conversion, "reasons": self.reasons, "warnings": self.warnings,
            "placement": None if p is None else {
                "tier": p.tier, "head_node": p.head_node, "est_total_mb": p.est_total_mb,
                "est_decode_tps": p.est_decode_tps,
                "devices": [f"{a.node_id}/{a.device_id}" for a in p.assignments]},
        }


def supported_engines(facts: ModelFacts, convert_ok: bool) -> dict[str, tuple[bool, list[str], list[str]]]:
    """engine -> (supported, reasons, warnings) from what the model is alone."""
    out: dict[str, tuple[bool, list[str], list[str]]] = {}
    if facts.kind == "gguf":
        out["llama.cpp"] = (True, ["GGUF file: llama.cpp serves it and can split it across GPUs and servers"], [])
        for e in ("vllm", "sglang"):
            out[e] = (False, [], [f"{ENGINE_LABELS[e]} serves Hugging Face repos, not library GGUF files"])
        return out
    what = []
    if facts.vision:
        what.append("vision-language model: images in chat messages work")
    if facts.quant:
        what.append(f"{facts.quant.upper()} checkpoint served as it is")
    if facts.task != "generate":
        what.append(f"{facts.task} model (/v1/embeddings, /v1/rerank, /v1/score)")
    vllm_reasons = what + ["the widest model coverage, served straight from the Hugging Face repo"]
    out["vllm"] = (True, vllm_reasons, [])
    sg_warn = []
    if facts.task != "generate":
        sg_warn.append("SGLang serves embedding models only with --is-embedding in the engine flags")
    out["sglang"] = (True, what + ["fast prefix caching (RadixAttention) for chats and agents"], sg_warn)
    # llama.cpp needs a GGUF: the coordinator's converter makes one from text generation checkpoints.
    blockers = []
    if facts.vision:
        blockers.append("vision models need an mmproj file, which gpupool does not manage for llama.cpp yet")
    if facts.task != "generate":
        blockers.append(f"gpupool runs llama.cpp for text generation only, not {facts.task}")
    if facts.quant and facts.quant not in PREQUANT_SUPPORTED:
        blockers.append(f"the GGUF converter cannot read {facts.quant.upper()} checkpoints")
    if not convert_ok:
        blockers.append("the coordinator has no GGUF conversion toolchain")
    if blockers:
        out["llama.cpp"] = (False, [], blockers)
    else:
        out["llama.cpp"] = (True, ["convert to GGUF first (Library, Convert); then it can pool GPUs of "
                                   "several servers and use quantizations down to 2 bits"], [])
    return out


def recommend(facts: ModelFacts, servers_by_engine: Mapping[str, list[str]],
              fits: Mapping[str, Placement | None], convert_ok: bool) -> dict:
    """The ranked options and the recommended engine.

    servers_by_engine: engine -> servers whose agent runs it (llama.cpp: every server).
    fits: engine -> best placement right now, None when nothing fits; engines absent were not
    checked (a conversion first, or not installed anywhere)."""
    options = []
    for engine, (ok, reasons, warnings) in supported_engines(facts, convert_ok).items():
        opt = EngineOption(engine=engine, supported=ok, reasons=list(reasons), warnings=list(warnings),
                           servers=list(servers_by_engine.get(engine, [])))
        if not ok:
            options.append(opt)
            continue
        convert = facts.kind == "hf" and engine == "llama.cpp"
        opt.needs_conversion = convert
        opt.score = _CONVERT_BASE if convert else _BASE[engine]
        if not opt.servers:
            opt.score -= _NOT_INSTALLED
            opt.warnings.append(f"no server runs {ENGINE_LABELS[engine]}: use the gpupool-agent \"-all\" image "
                                f"or set {ENV_OF.get(engine, 'GPUPOOL_LLAMA_DIR')} on a GPU server")
        elif engine in fits:
            p = fits[engine]
            opt.placement, opt.fits_now = p, p is not None
            if p is None:
                opt.score -= _NO_FIT
                where = "on one server" if engine != "llama.cpp" else "in the pool"
                opt.warnings.append(f"does not fit {where} in the GPU memory free right now")
            else:
                speed = f", ~{p.est_decode_tps:.0f} tok/s" if p.est_decode_tps else ""
                opt.reasons.insert(0, f"fits now on {len(p.assignments)} GPU(s) of {p.head_node}{speed}")
        options.append(opt)
    if facts.kind == "hf" and not any(o.fits_now for o in options if o.engine in ("vllm", "sglang")):
        for o in options:
            if o.engine == "llama.cpp" and o.supported:
                o.score += 10
                o.reasons.append("no single server fits it as is: llama.cpp splits a GGUF across servers")
    options.sort(key=lambda o: (not o.supported, -o.score, o.engine))
    best = next((o for o in options if o.supported), None)
    return {
        "model": {"kind": facts.kind, "task": facts.task, "vision": facts.vision, "quant": facts.quant,
                  "arch": facts.arch},
        "recommended": best.engine if best else None,
        "options": [o.as_dict() for o in options],
    }
