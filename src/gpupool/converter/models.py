"""Contracts of the Hugging Face -> GGUF conversion feature.

Shared by the converter package (inspect, jobs), the coordinator API and the web UI: every field
here is part of the HTTP API, so renaming one is an API change.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Curated llama-quantize types. The IQ1/IQ2/IQ3_XXS/IQ3_XS ones need an importance matrix
# (llama-quantize refuses them without one); gpupool computes it with llama-imatrix.
QuantType = Literal[
    "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_0",
    "Q3_K_L", "Q3_K_M", "IQ3_M", "IQ3_S", "Q3_K_S", "IQ3_XS", "IQ3_XXS", "Q2_K",
    "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
]
# auto = compute an importance matrix when the type needs one or is below ~4 bits, where it
# helps most; on = always (slower, better at every size); off = never (refused for types that need it).
ImatrixMode = Literal["auto", "on", "off"]
# What convert_hf_to_gguf.py writes before llama-quantize runs. "auto" = the highest-fidelity
# 16-bit type for the source (bf16 for bf16 weights, else f16).
Intermediate = Literal["auto", "f16", "bf16", "f32"]
JobState = Literal[
    "queued", "downloading", "converting", "calibrating", "quantizing", "validating",
    "needs_review", "done", "failed", "cancelled",
]
ACTIVE_STATES: frozenset[str] = frozenset(
    {"queued", "downloading", "converting", "calibrating", "quantizing", "validating"})
TERMINAL_STATES: frozenset[str] = frozenset({"done", "failed", "cancelled"})


class ConvertError(Exception):
    """A conversion request or job operation failed; `status` is the HTTP status to return."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class ClusterVram(BaseModel):
    """What the cluster can hold right now, for fit checks and the recommended quant type."""

    largest_gpu_mb: int = 0  # usable MB of the largest single GPU (0 = no GPU servers)
    pool_mb: int = 0  # usable MB summed over every GPU


class QuantOption(BaseModel):
    type: QuantType
    bpw: float  # bits per weight used for size estimates
    tier: Literal["lossless", "near_lossless", "balanced", "small", "tiny"]
    note: str  # quality note, e.g. llama-quantize's "+0.1754 ppl @ Llama-3-8B"
    via: Literal["convert", "quantize"]  # F16/BF16/Q8_0 are written by the converter directly
    needs_imatrix: bool = False  # llama-quantize refuses this type without an importance matrix
    # Filled by inspect for a concrete model; None in the static option list.
    est_bytes: int | None = None
    est_vram_mb: int | None = None  # file + KV cache at ctx 4096 + runtime overhead
    fits_single_gpu: bool | None = None
    fits_pool: bool | None = None
    recommended: bool = False


class SourceSpec(BaseModel):
    """Exactly one of hf_repo / path."""

    hf_repo: str | None = None  # "owner/name"
    revision: str = "main"
    # Absolute directory holding config.json + weights. Host paths are translated like library
    # paths (GPUPOOL_PATH_MAP), so a folder typed in the UI works when the coordinator is in Docker.
    path: str | None = None

    @model_validator(mode="after")
    def _one(self):
        if (self.hf_repo is None) == (self.path is None):
            raise ValueError("give exactly one of hf_repo or path")
        return self


class SourceFile(BaseModel):
    name: str  # path relative to the repo / directory root ("/" separated)
    bytes: int


class InspectResult(BaseModel):
    source: SourceSpec
    architecture: str | None = None  # config.json architectures[0], e.g. "Qwen2ForCausalLM"
    model_type: str | None = None  # config.json model_type
    supported: bool | None = None  # by the pinned converter; None = toolchain missing, unknown
    params: int | None = None  # parameter count
    n_layers: int | None = None
    context_length: int | None = None
    weight_format: Literal["safetensors", "pytorch_bin", "none"] = "none"
    prequantized: str | None = None  # quantization_config.quant_method (awq, gptq, fp8, ...)
    prequant_supported: bool | None = None  # can the converter dequantize it
    source_bytes: int = 0  # bytes of the selected files (download size for HF)
    files: list[SourceFile] = Field(default_factory=list)  # files the conversion will use
    skipped: list[str] = Field(default_factory=list)  # files ignored (code, other formats)
    remote_code: bool = False  # the source ships *.py / auto_map (not run unless allowed)
    gated: bool = False
    base_model: str | None = None  # HF card base_model: convert that instead of a prequantized repo
    gguf_alternatives: list[str] = Field(default_factory=list)  # HF repos with ready GGUF builds
    options: list[QuantOption] = Field(default_factory=list)  # every QuantType, with estimates
    recommended: QuantType = "Q4_K_M"
    name_stem: str = "model"  # default output name is f"{name_stem}-{quant}.gguf"
    recommend_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ConvertAdvanced(BaseModel):
    intermediate: Intermediate = "auto"
    output_tensor_type: str | None = None  # llama-quantize --output-tensor-type (ggml type name)
    token_embedding_type: str | None = None  # llama-quantize --token-embedding-type
    leave_output_tensor: bool = False  # llama-quantize --leave-output-tensor
    pure: bool = False  # llama-quantize --pure: no mixed types
    # Download and let the converter run the repo's own *.py (custom tokenizers). Off by default:
    # that code runs inside the coordinator.
    allow_remote_code: bool = False
    validate_generation: bool = True  # generate a few tokens on CPU after converting
    imatrix: ImatrixMode = "auto"
    # Calibration text for the importance matrix: an absolute .txt path on the server (host paths
    # translated like library paths). None = the multilingual text shipped with gpupool.
    calibration_path: str | None = None
    imatrix_chunks: int = Field(default=0, ge=0)  # 512-token chunks to process; 0 = default (100)
    threads: int = Field(default=0, ge=0)  # 0 = coordinator default (GPUPOOL_CONVERT_THREADS)


class ConvertRequest(BaseModel):
    source: SourceSpec
    quant: QuantType = "Q4_K_M"
    name: str | None = None  # output file name ending in .gguf; default "<model>-<QUANT>.gguf"
    keep_source: bool = False  # keep downloaded HF files after success (path sources are never touched)
    advanced: ConvertAdvanced = Field(default_factory=ConvertAdvanced)


class TokenizerCase(BaseModel):
    text: str
    hf: list[int]
    gguf: list[int]
    match: bool


class Validation(BaseModel):
    header_ok: bool | None = None
    architecture: str | None = None  # general.architecture in the GGUF
    n_layers: int | None = None
    vocab_size: int | None = None
    chat_template: bool | None = None
    tokenizer_ok: bool | None = None  # None = the comparison could not run
    tokenizer_cases: list[TokenizerCase] = Field(default_factory=list)
    generation_ok: bool | None = None  # None = skipped
    generation_sample: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ConvertJob(BaseModel):
    id: str
    request: ConvertRequest
    state: JobState
    stage_progress: float | None = None  # 0..1 within the current stage, None when unknown
    bytes_done: int = 0  # download stage
    bytes_total: int | None = None
    output_name: str
    failed_stage: JobState | None = None  # the stage that was running when it failed / was cancelled
    imatrix_used: bool = False  # an importance matrix was (or will be) computed for this job
    output_bytes: int | None = None
    est_output_bytes: int | None = None
    validation: Validation | None = None
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)  # last lines of the current/last tool
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
