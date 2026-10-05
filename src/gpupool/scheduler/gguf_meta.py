"""Sequential GGUF header parser: metadata without reading the tensor data.

Own parser (not gguf.GGUFReader) because GGUFReader memory-maps the whole file;
the coordinator needs metadata of multi-GB files on disk or behind a URL while
transferring only the header.
"""
from __future__ import annotations

import math
import os
import re
import struct
from collections.abc import Iterator
from typing import BinaryIO

import gguf
import httpx

from gpupool.common.net import external_sync_kwargs

from gpupool.common.models import ModelMeta

_MAGIC = b"GGUF"
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_T_STRING = 8
_T_ARRAY = 9
_MAX_KEEP_ARRAY = 4096  # keep small numeric arrays (per-layer head counts); skip tokenizer data
# Per-layer arrays the cache layout needs; every other array (tokenizer data) is skipped.
_KEEP_ARRAYS = ("head_count_kv", "head_count", "sliding_window_pattern", "recurrent_layers")
_MAX_STR = 1 << 28


class _Stream:
    """Exact-length reads over a file object or an iterator of byte chunks."""

    def __init__(self, src: BinaryIO | Iterator[bytes]):
        self._file = src if hasattr(src, "read") else None
        self._it = None if self._file else src
        self._buf = bytearray()
        self.consumed = 0

    def read(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("corrupt GGUF: negative length")
        if self._file is not None:
            data = self._file.read(n)
            while len(data) < n:
                more = self._file.read(n - len(data))
                if not more:
                    break
                data += more
        else:
            while len(self._buf) < n:
                try:
                    self._buf += next(self._it)
                except StopIteration:
                    break
            data = bytes(self._buf[:n])
            del self._buf[:n]
        if len(data) < n:
            raise ValueError(
                f"truncated GGUF: wanted {n} bytes at offset {self.consumed}, got {len(data)}"
            )
        self.consumed += n
        return data

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > _MAX_STR:
            raise ValueError(f"corrupt GGUF: string length {n}")
        return self.read(n).decode("utf-8", errors="replace")


class _Count(int):
    """Element count of an array whose contents were skipped (not kept)."""


def _read_value(s: _Stream, vtype: int, keep: bool):
    if vtype in _SCALARS:
        return s.unpack(_SCALARS[vtype])
    if vtype == _T_STRING:
        return s.string()
    if vtype == _T_ARRAY:
        etype = s.unpack("<I")
        count = s.unpack("<Q")
        if etype in _SCALARS:
            size = struct.calcsize(_SCALARS[etype])
            if count * size > 1 << 32:
                raise ValueError(f"corrupt GGUF: array of {count} elements")
            raw = s.read(count * size)
            if keep and count <= _MAX_KEEP_ARRAY:
                return list(struct.unpack(f"<{count}{_SCALARS[etype][1]}", raw))
            return _Count(count)
        if etype in (_T_STRING, _T_ARRAY):
            if count > 1 << 28:
                raise ValueError(f"corrupt GGUF: array of {count} elements")
            for _ in range(count):
                _read_value(s, etype, False)
            return _Count(count)
        raise ValueError(f"corrupt GGUF: unknown array element type {etype}")
    raise ValueError(f"corrupt GGUF: unknown value type {vtype}")


def _tensor_bytes(name: str, dims: list[int], ggml_type: int) -> int:
    try:
        block, tsize = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(ggml_type)]
    except (ValueError, KeyError):
        raise ValueError(f"tensor {name!r}: unknown ggml type {ggml_type}") from None
    n = 1
    for d in dims:
        n *= d
    return n // block * tsize


def _parse(s: _Stream) -> tuple[dict, list[tuple[str, int]]]:
    if s.read(4) != _MAGIC:
        raise ValueError("not a GGUF file (bad magic)")
    version = s.unpack("<I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = s.unpack("<Q")
    n_kv = s.unpack("<Q")
    if n_tensors > 1_000_000 or n_kv > 1_000_000:
        raise ValueError("corrupt GGUF: implausible tensor/kv count")
    kv: dict = {}
    for _ in range(n_kv):
        key = s.string()
        vtype = s.unpack("<I")
        kv[key] = _read_value(s, vtype, key.endswith(_KEEP_ARRAYS))
    tensors: list[tuple[str, int]] = []
    for _ in range(n_tensors):
        name = s.string()
        n_dims = s.unpack("<I")
        if n_dims > 8:
            raise ValueError(f"corrupt GGUF: tensor {name!r} has {n_dims} dims")
        dims = [s.unpack("<Q") for _ in range(n_dims)]
        ggml_type = s.unpack("<I")
        s.unpack("<Q")  # data offset, not needed
        tensors.append((name, _tensor_bytes(name, dims, ggml_type)))
    return kv, tensors


def _need(kv: dict, key: str):
    if key not in kv or kv[key] is None:
        raise ValueError(f"GGUF metadata is missing {key!r}")
    return kv[key]


def _scalar_or_max(v):
    return max(v) if isinstance(v, list) else v


def _build(kv: dict, tensors: list[tuple[str, int]], file_bytes: int | None) -> ModelMeta:
    arch = _need(kv, "general.architecture")
    n_layers = int(_need(kv, f"{arch}.block_count"))
    n_embd = int(_need(kv, f"{arch}.embedding_length"))
    n_head = int(_scalar_or_max(_need(kv, f"{arch}.attention.head_count")))
    n_head_kv = int(_scalar_or_max(kv.get(f"{arch}.attention.head_count_kv") or n_head))
    head_dim = int(kv.get(f"{arch}.attention.key_length") or n_embd // max(n_head, 1))
    layer_bytes = [0] * n_layers
    expert_bytes = [0] * n_layers  # routed experts only; shared experts (_shexp) always run
    other = 0
    sizes: dict[str, int] = {}
    pat = re.compile(r"^blk\.(\d+)\.")
    for name, nbytes in tensors:
        sizes[name] = nbytes
        m = pat.match(name)
        if m:
            i = int(m.group(1))
            if i >= n_layers:
                raise ValueError(f"tensor {name!r} beyond block_count {n_layers}")
            layer_bytes[i] += nbytes
            if _EXPERTS.match(name):
                expert_bytes[i] += nbytes
        else:
            other += nbytes
    layout = _layout(kv, arch, n_layers, n_embd, n_head, head_dim)
    n_nextn = min(int(kv.get(f"{arch}.nextn_predict_layers") or 0), n_layers)
    nextn_bytes = 0
    if n_nextn and arch in _MTP_ON_DEMAND:
        # llama.cpp skips these tensors unless --spec-type draft-mtp asks for them
        nextn_bytes = sum(layer_bytes[n_layers - n_nextn:])
        for i in range(n_layers - n_nextn, n_layers):
            layer_bytes[i] = expert_bytes[i] = 0
    active = None
    n_expert = int(kv.get(f"{arch}.expert_count") or 0)
    n_used = int(kv.get(f"{arch}.expert_used_count") or 0)
    if n_expert > 0 and 0 < n_used < n_expert and any(expert_bytes):
        active = [b - e + math.ceil(e * n_used / n_expert) for b, e in zip(layer_bytes, expert_bytes)]
    out = sizes.get("output.weight", sizes.get("token_embd.weight", 0))
    out += sum(b for n, b in sizes.items() if n.startswith("output_norm."))
    # tokens is a skipped string array: _read_value leaves only its element count
    tokens = kv.get("tokenizer.ggml.tokens")
    vocab = len(tokens) if isinstance(tokens, list) else int(tokens) if isinstance(tokens, _Count) else None
    tok_model = kv.get("tokenizer.ggml.model")
    return ModelMeta(
        arch=arch, n_layers=n_layers, n_embd=n_embd, n_head=n_head, n_head_kv=n_head_kv,
        head_dim=head_dim, layer_bytes=layer_bytes, other_bytes=other, output_bytes=out,
        file_bytes=file_bytes, vocab_size=vocab,
        tokenizer_model=tok_model if isinstance(tok_model, str) else None,
        n_nextn=n_nextn if nextn_bytes else 0, nextn_bytes=nextn_bytes, active_bytes=active,
        **layout,
    )


# Routed expert tensors of a MoE layer ("ffn_gate_exps", "ffn_down_exps", "ffn_gate_up_exps"...).
_EXPERTS = re.compile(r"^blk\.\d+\.ffn_[a-z_]*_c?h?exps\.")

# Architectures whose MTP (nextn) blocks llama.cpp b11342 loads only for --spec-type draft-mtp
# (TENSOR_SKIP otherwise, src/models/*.cpp). Others with nextn layers load them always.
_MTP_ON_DEMAND = frozenset({
    "bailingmoe3", "cohere2moe", "deepseek2", "deepseek32", "deepseek4", "glm-dsa", "glm4moe",
    "glm5-next", "hy_v3", "mimo2", "nemotron_h", "qwen35", "qwen35moe", "qwen3next", "qwen4exp",
    "step35",
})

# Sliding-window layers, for architectures whose loader sets the pattern itself when the GGUF
# carries no sliding_window_pattern array: (period, dense_first) as passed to load_swa_pattern.
# Only applied when the GGUF has sliding_window > 0. Architectures not listed (or with conditions
# this table cannot see) count every layer as full attention: an over-estimate, never an OOM.
_SWA_PATTERN = {"gemma2": (2, False), "gemma3": (6, False), "gemma3n": (5, False),
                "gpt-oss": (2, False), "cohere2": (4, False), "olmo2": (4, False)}

# Hybrid models whose loader marks recurrent layers by full_attention_interval (default 4).
_INTERVAL_RECURRENT = frozenset({"qwen3next", "qwen35", "qwen35moe"})
_PURE_RECURRENT = frozenset({"mamba", "mamba2"})

# Layers from this index on reuse earlier layers' KV (n_layer_kv_from_start in the loaders).
_KV_FROM_START_FIXED = {"gemma3n": 20}


def _per_layer(v, n: int, default: int) -> list[int]:
    if isinstance(v, list) and len(v) >= n:
        return [int(x) for x in v[:n]]
    if isinstance(v, (int, float)) and not isinstance(v, _Count):
        return [int(v)] * n
    return [default] * n


def _bool_layers(v, n: int) -> list[bool] | None:
    if isinstance(v, list) and len(v) >= n:
        return [bool(x) for x in v[:n]]
    return None


def _layout(kv: dict, arch: str, n_layers: int, n_embd: int, n_head: int, head_dim: int) -> dict:
    """Per-layer KV / state layout, mirroring llama.cpp b11342 (llama-hparams.cpp,
    llama-kv-cache.cpp, llama-kv-cache-iswa.cpp and the per-architecture loaders)."""
    a = f"{arch}."
    n_nextn = min(int(kv.get(a + "nextn_predict_layers") or 0), n_layers)
    n_main = n_layers - n_nextn
    heads = _per_layer(kv.get(a + "attention.head_count"), n_layers, n_head)
    hkv = _per_layer(kv.get(a + "attention.head_count_kv"), n_layers, 0)
    hkv = [h if kv.get(a + "attention.head_count_kv") is not None else heads[i]
           for i, h in enumerate(hkv)]
    head_k = int(kv.get(a + "attention.key_length") or head_dim)
    head_v = int(kv.get(a + "attention.value_length") or head_k)
    head_k_swa = int(kv.get(a + "attention.key_length_swa") or head_k)
    head_v_swa = int(kv.get(a + "attention.value_length_swa") or head_v)
    mla = bool(kv.get(a + "attention.key_length_mla"))  # V is not cached with MLA

    n_swa = int(kv.get(a + "attention.sliding_window") or 0)
    swa = [False] * n_layers
    if n_swa > 0:
        arr = _bool_layers(kv.get(a + "attention.sliding_window_pattern"), n_layers)
        if arr is not None:
            swa = arr
        elif arch in _SWA_PATTERN:
            period, dense_first = _SWA_PATTERN[arch]
            given = kv.get(a + "attention.sliding_window_pattern")
            if isinstance(given, int) and not isinstance(given, _Count):
                period = given
            for i in range(n_main):  # llama_hparams::set_swa_pattern
                swa[i] = period == 0 or (i % period != 0 if dense_first else i % period < period - 1)
        for i in range(n_main, n_layers):
            swa[i] = False

    recr = _bool_layers(kv.get(a + "attention.recurrent_layers"), n_layers)
    if recr is None and isinstance(kv.get(a + "attention.head_count_kv"), list) \
            and any(hkv[i] == 0 for i in range(n_main)):
        # Jamba, Nemotron-H, Granite hybrid, LFM2...: a layer without KV heads is not attention
        recr = [i < n_main and hkv[i] == 0 for i in range(n_layers)]
    if recr is None and arch in _INTERVAL_RECURRENT:
        interval = int(kv.get(a + "full_attention_interval") or 4)
        recr = [i < n_main and (i + 1) % interval != 0 for i in range(n_layers)]
    if recr is None and arch in _PURE_RECURRENT:
        recr = [i < n_main for i in range(n_layers)]
    d_conv = int(kv.get(a + "ssm.conv_kernel") or 0)
    d_inner = int(kv.get(a + "ssm.inner_size") or 0)
    d_state = int(kv.get(a + "ssm.state_size") or 0)
    n_group = int(kv.get(a + "ssm.group_count") or 0)
    # llama_hparams::n_embd_r / n_embd_s (Mamba-style), f32 per sequence
    ssm_state = 4 * (max(d_conv - 1, 0) * (d_inner + 2 * n_group * d_state) + d_state * d_inner)

    kv_from = _KV_FROM_START_FIXED.get(arch)
    shared = int(kv.get(a + "attention.shared_kv_layers") or 0)
    if shared and arch == "gemma4":
        kv_from = n_layers - shared

    k_rows, v_rows, state = [0] * n_layers, [0] * n_layers, [0] * n_layers
    for i in range(n_layers):
        no_attn = (recr is not None and recr[i]) or (kv_from is not None and i >= kv_from)
        if recr is not None and recr[i]:
            if not ssm_state:  # a recurrent layout this parser does not know: keep the old rule
                k_rows[i] = v_rows[i] = (hkv[i] or max(hkv) or n_head) * head_k
                continue
            state[i] = ssm_state
        if no_attn:
            continue
        hk, hv = (head_k_swa, head_v_swa) if swa[i] else (head_k, head_v)
        k_rows[i] = hk * hkv[i]
        v_rows[i] = 0 if mla else hv * hkv[i]
    return {"kv_k": k_rows, "kv_v": v_rows, "swa": swa, "n_swa": n_swa,
            "state_bytes": state if any(state) else None}


def _load(source: str, headers: dict | None) -> tuple[dict, list[tuple[str, int]], int | None]:
    """Parse one GGUF header (local path or http(s) URL): (kv, tensors, file size or None)."""
    if source.startswith(("http://", "https://")):
        return _load_url(source, headers)
    with open(source, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        kv, tensors = _parse(_Stream(f))
    return kv, tensors, size


def _load_url(url: str, headers: dict | None) -> tuple[dict, list[tuple[str, int]], int | None]:
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0),
                      **external_sync_kwargs()) as client:
        with client.stream("GET", url, headers=headers or {}) as resp:
            resp.raise_for_status()
            cl = resp.headers.get("content-length")
            size = int(cl) if cl and cl.isdigit() else None
            kv, tensors = _parse(_Stream(resp.iter_raw(64 * 1024)))
        # leaving the stream context closes the connection: the body is not downloaded
    return kv, tensors, size


def _split_count(kv: dict) -> int:
    # gguf.Keys.Split.LLM_KV_SPLIT_COUNT; written by llama.cpp's gguf-split into every part
    v = kv.get("split.count")
    return int(v) if isinstance(v, int) else 0


def read_meta(source: str, *, headers: dict | None = None) -> ModelMeta:
    """Read ModelMeta from a local path or http(s) URL, touching only the header.

    Single-file GGUFs only: a split GGUF's part holds just some of the tensors, so its layer
    sizes would silently be too small (an under-estimate of VRAM).
    """
    kv, tensors, size = _load(source, headers)
    if _split_count(kv) > 1:
        raise ValueError(
            f"split GGUF ({_split_count(kv)} parts): its tensors are spread over several files, "
            "use read_meta_parts with every part")
    return _build(kv, tensors, size)


def read_meta_parts(sources: list[str], *, headers: dict | None = None) -> ModelMeta:
    """ModelMeta of a split GGUF: KV (architecture, block_count, ...) from the first part,
    tensors summed over all parts, file_bytes = sum of part sizes (None if any is unknown).
    sources are local paths or http(s) URLs, in part order."""
    if not sources:
        raise ValueError("no GGUF parts given")
    kv: dict = {}
    tensors: list[tuple[str, int]] = []
    total: int | None = 0
    for i, src in enumerate(sources):
        pkv, ptensors, size = _load(src, headers)
        if i == 0:
            kv = pkv
            declared = _split_count(kv)
            if declared and declared != len(sources):
                raise ValueError(f"split GGUF declares {declared} parts, got {len(sources)}")
        tensors.extend(ptensors)
        total = None if total is None or size is None else total + size
    return _build(kv, tensors, total)
