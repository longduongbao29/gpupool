"""Sequential GGUF header parser: metadata without reading the tensor data.

Own parser (not gguf.GGUFReader) because GGUFReader memory-maps the whole file;
the coordinator needs metadata of multi-GB files on disk or behind a URL while
transferring only the header.
"""
from __future__ import annotations

import os
import re
import struct
from collections.abc import Iterator
from typing import BinaryIO

import gguf
import httpx

from gpupool.common.models import ModelMeta

_MAGIC = b"GGUF"
_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_T_STRING = 8
_T_ARRAY = 9
_MAX_KEEP_ARRAY = 4096  # keep small numeric arrays (per-layer head counts); skip tokenizer data
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
            return None
        if etype in (_T_STRING, _T_ARRAY):
            if count > 1 << 28:
                raise ValueError(f"corrupt GGUF: array of {count} elements")
            for _ in range(count):
                _read_value(s, etype, False)
            return None
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
        kv[key] = _read_value(s, vtype, key.endswith(("head_count_kv", "head_count")))
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
        else:
            other += nbytes
    out = sizes.get("output.weight", sizes.get("token_embd.weight", 0))
    out += sum(b for n, b in sizes.items() if n.startswith("output_norm."))
    return ModelMeta(
        arch=arch, n_layers=n_layers, n_embd=n_embd, n_head=n_head, n_head_kv=n_head_kv,
        head_dim=head_dim, layer_bytes=layer_bytes, other_bytes=other, output_bytes=out,
        file_bytes=file_bytes,
    )


def read_meta(source: str, *, headers: dict | None = None) -> ModelMeta:
    """Read ModelMeta from a local path or http(s) URL, touching only the header."""
    if source.startswith(("http://", "https://")):
        return _read_url(source, headers)
    with open(source, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        kv, tensors = _parse(_Stream(f))
    return _build(kv, tensors, size)


def _read_url(url: str, headers: dict | None) -> ModelMeta:
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0)) as client:
        with client.stream("GET", url, headers=headers or {}) as resp:
            resp.raise_for_status()
            cl = resp.headers.get("content-length")
            size = int(cl) if cl and cl.isdigit() else None
            kv, tensors = _parse(_Stream(resp.iter_raw(64 * 1024)))
        # leaving the stream context closes the connection: the body is not downloaded
    return _build(kv, tensors, size)
