"""ModelMeta for Hugging Face checkpoints served by vLLM / SGLang, without downloading weights.

config.json gives the shape (layers, heads, KV layout); the repo's file listing gives the weight
bytes. Coarser than the GGUF reader: weights are spread evenly over the layers (tensor parallelism
shards every layer over every GPU, so only the total matters for placement), and the KV layout is
the attention layout the engines allocate with a bf16 cache:
  - plain / grouped-query attention: K and V rows of n_head_kv x head_dim;
  - MLA (DeepSeek-V2/V3, Kimi K2...): one latent row of kv_lora_rank + qk_rope_head_dim, no V;
  - layer_types "sliding_attention": sliding window; "linear_attention" / "mamba": no KV rows.
Vision-language models: the vision tower's weights are in the files, so they are counted.
"""
from __future__ import annotations

import fnmatch

from gpupool.common.models import ModelMeta

# Layer types without a KV cache (their small recurrent state is not estimated).
_NO_KV_TYPES = {"linear_attention", "mamba", "mamba2", "conv", "recurrent", "ssm"}


def _flat(config: dict) -> dict:
    """Text model fields: VLMs (Qwen2.5-VL, Gemma 3, Llava...) nest them under text_config."""
    tc = config.get("text_config")
    if not isinstance(tc, dict):
        tc = config.get("llm_config") if isinstance(config.get("llm_config"), dict) else None
    return {**config, **tc} if isinstance(tc, dict) else config


def _int(cfg: dict, *keys: str) -> int | None:
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def weight_files(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """The files the engines load: top-level *.safetensors (HF layout model-*.safetensors when a
    repo also ships consolidated.safetensors for mistral-inference), else pytorch_model*.bin."""
    top = [(n, b) for n, b in files if "/" not in n]
    st = [(n, b) for n, b in top if n.endswith(".safetensors")]
    hf_layout = [(n, b) for n, b in st if fnmatch.fnmatch(n, "model*.safetensors")]
    if hf_layout:
        return hf_layout
    if st:
        return st
    return [(n, b) for n, b in top if fnmatch.fnmatch(n, "pytorch_model*.bin")]


def meta_from_config(config: dict, weight_bytes: int) -> ModelMeta:
    """ValueError when config.json lacks what the estimate needs."""
    c = _flat(config)
    n_layers = _int(c, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    n_embd = _int(c, "hidden_size", "d_model", "n_embd", "dim")
    n_head = _int(c, "num_attention_heads", "n_head", "n_heads")
    if not (n_layers and n_embd and n_head):
        raise ValueError("config.json has no num_hidden_layers / hidden_size / num_attention_heads")
    if weight_bytes <= 0:
        raise ValueError("no safetensors or pytorch_model*.bin weights in the repo")
    n_head_kv = _int(c, "num_key_value_heads", "n_head_kv", "num_kv_heads") or n_head
    head_dim = _int(c, "head_dim") or n_embd // n_head

    kv_lora = _int(c, "kv_lora_rank")
    if kv_lora:  # MLA: the engines cache the compressed latent plus the rope part of K
        row_k, row_v = kv_lora + (_int(c, "qk_rope_head_dim") or 0), 0
    else:
        row_k = row_v = n_head_kv * head_dim
    kv_k, kv_v = [row_k] * n_layers, [row_v] * n_layers
    swa = [False] * n_layers
    n_swa = _int(c, "sliding_window") or 0
    types = c.get("layer_types")
    if isinstance(types, list) and len(types) == n_layers:
        for i, t in enumerate(types):
            t = str(t)
            if t in _NO_KV_TYPES:
                kv_k[i] = kv_v[i] = 0
            elif t.startswith("sliding") and n_swa:
                swa[i] = True

    per_layer = weight_bytes // n_layers
    layer_bytes = [per_layer] * n_layers
    active = None
    n_exp = _int(c, "num_experts", "num_local_experts", "n_routed_experts")
    k_exp = _int(c, "num_experts_per_tok", "num_experts_per_token", "moe_topk")
    if n_exp and k_exp and k_exp < n_exp:
        # MoE: a token reads the attention / shared part and k of the n experts; experts are
        # most of the weights in practice, so ~10% shared is assumed.
        active = [int(per_layer * (0.1 + 0.9 * k_exp / n_exp))] * n_layers
    return ModelMeta(
        arch=str(c.get("model_type") or config.get("model_type") or "unknown"),
        n_layers=n_layers, n_embd=n_embd, n_head=n_head, n_head_kv=n_head_kv, head_dim=head_dim,
        layer_bytes=layer_bytes, other_bytes=weight_bytes - per_layer * n_layers, output_bytes=0,
        file_bytes=weight_bytes, vocab_size=_int(c, "vocab_size"),
        kv_k=kv_k, kv_v=kv_v, swa=swa if any(swa) else None, n_swa=n_swa if any(swa) else 0,
        active_bytes=active)


def has_vision(config: dict) -> bool:
    return any(isinstance(config.get(k), dict) for k in ("vision_config", "visual", "vision_tower"))


async def fetch_hf_meta(hf, repo: str, revision: str | None = None) -> ModelMeta:
    """`hf`: converter.source.HfClient. Reads config.json and the file listing only."""
    rev = revision or "main"
    files = await hf.list_files(repo, rev)
    weights = weight_files([(f.name, f.bytes) for f in files])
    config = await hf.fetch_json(repo, rev, "config.json")
    return meta_from_config(config, sum(b for _, b in weights))
