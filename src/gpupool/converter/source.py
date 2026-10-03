"""Conversion sources: Hugging Face repos (HfClient) and local folders, and `inspect_source`.

Inspecting never downloads weights: for HF the tree listing + model info + config.json are
enough, for a folder the safetensors headers are read (first 8 bytes = header length, then JSON).
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import math
import os
import re
import struct
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import httpx

from gpupool.converter.models import (
    ClusterVram, ConvertError, InspectResult, SourceFile, SourceSpec,
)
from gpupool.converter.quant import (
    QUANT_OPTIONS, estimate_bytes, estimate_vram_mb, recommend,
)

HF_BASE = "https://huggingface.co"
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_GATED_MSG = "repo is gated or private: set HF_TOKEN (and accept the license on huggingface.co)"
# Prequantized formats convert_hf_to_gguf.py (b11342) can dequantize.
PREQUANT_SUPPORTED = frozenset({"fp8", "gptq", "bitnet", "compressed-tensors", "modelopt", "mxfp4"})
_MAX_ST_HEADER = 100 * 1024 * 1024

_SKIP_DIRS = {"original", "onnx", "openvino", "coreml", "flax", "tf"}
_TOKENIZER_NAMES = {
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt",
    "merges.txt", "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
}


class HfClient:
    def __init__(self, http: httpx.AsyncClient, token: str = "", base: str = HF_BASE):
        self.http = http
        self.token = token
        self.base = base.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @staticmethod
    def _check(r: httpx.Response, what: str) -> None:
        if r.status_code in (401, 403):
            raise ConvertError(_GATED_MSG, 403)
        if r.status_code == 404:
            raise ConvertError(f"{what} not found on Hugging Face", 404)
        if r.status_code >= 400:
            raise ConvertError(f"Hugging Face returned HTTP {r.status_code} for {what}", 502)

    async def _get(self, url: str, what: str, params: dict | None = None) -> httpx.Response:
        try:
            r = await self.http.get(url, params=params, headers=self._headers(), follow_redirects=True)
        except httpx.HTTPError as e:
            raise ConvertError(f"cannot reach Hugging Face: {type(e).__name__}: {e}", 502) from e
        self._check(r, what)
        return r

    @staticmethod
    def _repo(repo: str) -> str:
        if not _REPO_RE.match(repo) or any(p in (".", "..") for p in repo.split("/")):
            raise ConvertError(f"invalid Hugging Face repo id {repo!r} (expected owner/name)", 400)
        return repo

    async def list_files(self, repo: str, revision: str = "main") -> list[SourceFile]:
        repo = self._repo(repo)
        url: str | None = f"{self.base}/api/models/{repo}/tree/{quote(revision, safe='')}"
        params: dict | None = {"recursive": "true"}
        out: list[SourceFile] = []
        for _ in range(1000):  # hard bound: a hostile Link chain must not loop forever
            r = await self._get(url, f"{repo}@{revision}", params)
            try:
                entries = r.json()
            except ValueError as e:
                raise ConvertError("Hugging Face returned an unreadable file listing", 502) from e
            for e in entries if isinstance(entries, list) else []:
                if e.get("type") == "file":
                    out.append(SourceFile(name=e["path"], bytes=int(e.get("size") or 0)))
            nxt = r.links.get("next", {}).get("url")
            if not nxt:
                return out
            url, params = nxt, None
        return out

    async def model_info(self, repo: str, revision: str = "main") -> dict:
        repo = self._repo(repo)
        r = await self._get(f"{self.base}/api/models/{repo}/revision/{quote(revision, safe='')}",
                            f"{repo}@{revision}")
        try:
            info = r.json()
        except ValueError as e:
            raise ConvertError("Hugging Face returned unreadable model info", 502) from e
        return info if isinstance(info, dict) else {}

    async def fetch_json(self, repo: str, revision: str, file: str) -> dict:
        repo = self._repo(repo)
        url = f"{self.base}/{repo}/resolve/{quote(revision, safe='')}/{quote(file)}"
        r = await self._get(url, f"{file} in {repo}")
        try:
            data = r.json()
        except ValueError as e:
            raise ConvertError(f"{file} is not valid JSON", 422) from e
        if not isinstance(data, dict):
            raise ConvertError(f"{file} is not a JSON object", 422)
        return data

    async def gguf_alternatives(self, repo: str, limit: int = 5) -> list[str]:
        try:
            r = await self.http.get(
                f"{self.base}/api/models",
                params=[("filter", f"base_model:quantized:{repo}"), ("filter", "gguf"),
                        ("sort", "downloads"), ("limit", str(limit))],
                headers=self._headers(), follow_redirects=True)
            if r.status_code != 200:
                return []
            return [m["id"] for m in r.json() if isinstance(m, dict) and m.get("id")][:limit]
        except Exception:  # advisory only: never fail an inspect over it
            return []

    async def download(self, repo: str, revision: str, file: str, dest: Path,
                       expected_bytes: int | None, on_progress: Callable[[int], None]) -> int:
        repo = self._repo(repo)
        if expected_bytes is not None:
            try:
                if dest.is_file() and dest.stat().st_size == expected_bytes:
                    on_progress(expected_bytes)
                    return expected_bytes
            except OSError:
                pass
        part = Path(str(dest) + ".part")
        url = f"{self.base}/{repo}/resolve/{quote(revision, safe='')}/{quote(file)}"
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            async with self.http.stream("GET", url, headers=self._headers(),
                                        follow_redirects=True) as r:
                self._check(r, f"{file} in {repo}")
                # With a Content-Encoding the raw bytes are compressed and the length header does
                # not match the decoded file: decode and skip that check.
                encoded = r.headers.get("content-encoding", "identity") != "identity"
                length = None if encoded else r.headers.get("content-length")
                wanted = {n for n in (int(length) if length is not None else None,
                                      None if encoded else expected_bytes) if n}
                chunks = r.aiter_bytes(1024 * 1024) if encoded else r.aiter_raw(1024 * 1024)
                got = 0
                # Writes run in a thread: this loop also serves the router, and a blocking
                # write/fsync of a multi-GB file would stall inference traffic.
                with open(part, "wb") as f:
                    async for chunk in chunks:
                        await asyncio.to_thread(f.write, chunk)
                        got += len(chunk)
                        on_progress(got)
                    await asyncio.to_thread(f.flush)
                    await asyncio.to_thread(os.fsync, f.fileno())
            if got == 0:
                raise ConvertError(f"empty download of {file}", 502)
            if any(got != n for n in wanted):
                raise ConvertError(f"truncated download of {file}: got {got} of {sorted(wanted)} bytes", 502)
            os.replace(part, dest)
            return got
        except ConvertError:
            _unlink(part)
            raise
        except asyncio.CancelledError:
            _unlink(part)
            raise
        except (httpx.HTTPError, OSError) as e:
            _unlink(part)
            raise ConvertError(f"download of {file} failed: {type(e).__name__}: {e}", 502) from e


def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError:
        pass


def select_files(files: list[SourceFile], allow_remote_code: bool = False) -> tuple[list[SourceFile], list[str]]:
    def bad_dir(name: str) -> bool:
        # Names become paths under the download cache: anything that could escape it is dropped.
        if name.startswith("/") or "\\" in name or ":" in name:
            return True
        parts = name.split("/")
        if any(p in ("", ".", "..") for p in parts):
            return True
        return any(d in _SKIP_DIRS or d.startswith(".") for d in parts[:-1])

    def is_st(base: str) -> bool:
        # consolidated.* is the Mistral-native copy of the same weights (Mistral repos ship both):
        # taking it would double the download, and the HF-format converter path does not use it.
        if base.startswith("consolidated."):
            return False
        return base.endswith(".safetensors") or base.endswith(".safetensors.index.json")

    def is_bin(base: str) -> bool:
        return fnmatch.fnmatch(base, "pytorch_model*.bin") or base == "pytorch_model.bin.index.json"

    cands = [f for f in files if not bad_dir(f.name)]
    has_st = any(is_st(f.name.rsplit("/", 1)[-1]) for f in cands)
    keep: list[SourceFile] = []
    skipped: list[str] = []
    for f in files:
        base = f.name.rsplit("/", 1)[-1]
        ok = (not bad_dir(f.name)) and (
            base in _TOKENIZER_NAMES or base.endswith(".tiktoken") or is_st(base)
            or (is_bin(base) and not has_st)
            or (allow_remote_code and base.endswith(".py")))
        (keep if ok else skipped).append(f if ok else f.name)  # type: ignore[arg-type]
    return keep, skipped


def local_files(root: Path) -> list[SourceFile]:
    out: list[SourceFile] = []
    for dirpath, _dirs, names in os.walk(root, followlinks=False):  # symlinked dirs are not entered
        for n in names:
            p = Path(dirpath) / n
            try:
                size = p.stat().st_size
                if not p.is_file():
                    continue
            except OSError:
                continue
            out.append(SourceFile(name=p.relative_to(root).as_posix(), bytes=size))
    out.sort(key=lambda f: f.name)
    return out


def _safetensors_params(path: Path) -> int:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("file too short")
        (n,) = struct.unpack("<Q", raw)
        if n > _MAX_ST_HEADER:
            raise ValueError("implausible header length")
        header = json.loads(f.read(n))
    return sum(math.prod(t["shape"]) for k, t in header.items() if k != "__metadata__")


def _flat(config: dict) -> dict:
    tc = config.get("text_config")
    return {**config, **tc} if isinstance(tc, dict) else config


def _first_int(cfg: dict, *keys: str) -> int | None:
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return None


def _base_model(card: object) -> str | None:
    bm = card.get("base_model") if isinstance(card, dict) else None
    if isinstance(bm, list):
        bm = bm[0] if bm else None
    return bm if isinstance(bm, str) and bm else None


async def inspect_source(spec: SourceSpec, *, hf: HfClient, locate_dir: Callable[[str], Path],
                         cluster: ClusterVram, supported_architectures: set[str] | None) -> InspectResult:
    warnings: list[str] = []
    gated = False
    base_model: str | None = None
    params: int | None = None
    root: Path | None = None
    alternatives: list[str] = []

    if spec.hf_repo is not None:
        info = await hf.model_info(spec.hf_repo, spec.revision)
        gated = bool(info.get("gated"))
        base_model = _base_model(info.get("cardData"))
        st = info.get("safetensors")
        if isinstance(st, dict) and isinstance(st.get("total"), int) and st["total"] > 0:
            params = st["total"]
        all_files = await hf.list_files(spec.hf_repo, spec.revision)
        try:
            config = await hf.fetch_json(spec.hf_repo, spec.revision, "config.json")
        except ConvertError as e:
            if e.status == 404:
                raise ConvertError("no config.json: not a transformers model repo", 422) from e
            raise
        if gated and not hf.token:
            warnings.append("This repo is gated: set HF_TOKEN and accept its license on huggingface.co "
                            "or the download will be refused.")
        alternatives = await hf.gguf_alternatives(spec.hf_repo)
    else:
        assert spec.path is not None
        try:
            root = locate_dir(spec.path)
        except ConvertError:
            raise
        except Exception as e:
            if hasattr(e, "message") and hasattr(e, "status"):
                raise ConvertError(e.message, e.status) from e  # LibraryError
            raise
        cfg_path = root / "config.json"
        if not cfg_path.is_file():
            raise ConvertError("no config.json in that folder: not a transformers model folder", 422)
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("not an object")
        except (OSError, ValueError) as e:
            raise ConvertError(f"config.json cannot be read: {e}", 422) from e
        all_files = await asyncio.to_thread(local_files, root)

    files, skipped = select_files(all_files, allow_remote_code=False)
    remote_code = any(f.name.endswith(".py") for f in all_files) or "auto_map" in config
    st_files = [f for f in files if f.name.endswith(".safetensors")]
    bin_files = [f for f in files if f.name.endswith(".bin")]
    weight_format = "safetensors" if st_files else "pytorch_bin" if bin_files else "none"
    if weight_format == "none":
        warnings.append("No safetensors or PyTorch .bin weights found: there is nothing to convert.")

    if root is not None and params is None:
        if st_files:
            try:
                params = await asyncio.to_thread(
                    lambda: sum(_safetensors_params(root / f.name) for f in st_files))
            except (OSError, ValueError, KeyError, TypeError) as e:
                warnings.append(f"Could not read the safetensors headers ({e}); the size is unknown.")
        elif bin_files:
            params = sum(f.bytes for f in bin_files) // 2
            warnings.append("The parameter count is estimated from the .bin file sizes (assuming 16-bit weights).")
    if params is None and spec.hf_repo is not None:
        if st_files:
            params = sum(f.bytes for f in st_files) // 2
            warnings.append("The parameter count is estimated from the file sizes (assuming 16-bit weights).")
        elif bin_files:
            params = sum(f.bytes for f in bin_files) // 2
            warnings.append("The parameter count is estimated from the .bin file sizes (assuming 16-bit weights).")

    archs = config.get("architectures")
    architecture = archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else None
    flat = _flat(config)
    model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else None
    n_layers = _first_int(flat, "num_hidden_layers", "n_layer", "num_layers")
    context = _first_int(flat, "max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")

    if supported_architectures is None:
        supported = None
        warnings.append("Conversion toolchain not installed: cannot tell whether this architecture is supported.")
    else:
        supported = architecture in supported_architectures
        if not supported:
            warnings.append(f"Architecture {architecture or 'unknown'} is not supported by the pinned converter.")

    qc = config.get("quantization_config") or flat.get("quantization_config")
    prequantized = None
    if isinstance(qc, dict):
        prequantized = str(qc.get("quant_method") or "unknown")
    prequant_supported = None if prequantized is None else prequantized in PREQUANT_SUPPORTED
    if prequantized is not None:
        msg = (f"This source is already quantized ({prequantized}); quantizing again loses extra quality")
        if prequant_supported is False:
            msg += " and the converter cannot read this format"
        if base_model:
            msg += f". Convert the original model {base_model} instead"
        warnings.append(msg + ".")
        warnings.append("The parameter count of a quantized source is approximate.")
    if remote_code:
        warnings.append("This source ships custom Python code. Without allow_remote_code it is not "
                        "downloaded or run, so a custom tokenizer may fail to load.")

    options = [o.model_copy() for o in QUANT_OPTIONS]
    for o in options:
        if params is None:
            continue
        o.est_bytes = estimate_bytes(params, o.type, config)
        o.est_vram_mb = estimate_vram_mb(o.est_bytes, config)
        if cluster.pool_mb > 0:
            o.fits_single_gpu = o.est_vram_mb <= cluster.largest_gpu_mb
            o.fits_pool = o.est_vram_mb <= cluster.pool_mb
    rec, reasons = recommend(params, options)
    for o in options:
        o.recommended = o.type == rec

    return InspectResult(
        source=spec, architecture=architecture, model_type=model_type, supported=supported,
        params=params, n_layers=n_layers, context_length=context, weight_format=weight_format,
        prequantized=prequantized, prequant_supported=prequant_supported,
        source_bytes=sum(f.bytes for f in files), files=files, skipped=skipped,
        remote_code=remote_code, gated=gated, base_model=base_model,
        gguf_alternatives=alternatives, options=options, recommended=rec,
        recommend_reasons=reasons, warnings=warnings)
