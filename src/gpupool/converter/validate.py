"""Post-conversion checks: GGUF header, tokenizer parity with Hugging Face, a short generation.

Outcome policy (the caller, jobs.py, applies it):
  - header_ok is False (header unreadable / no tokenizer; see errors) -> the conversion failed.
  - tokenizer_ok is False or generation_ok is False                  -> needs human review.
  - None values (check could not run, skipped, timed out)            -> not a failure; a warning.
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import psutil

from gpupool.converter.models import TokenizerCase, Validation
from gpupool.converter.toolchain import Toolchain

log = logging.getLogger(__name__)

# Fixed probe strings: the places where converted tokenizers usually go wrong (pre-tokenizer
# regex, whitespace handling, byte fallback, digits, non-Latin scripts).
TOKENIZER_TEXTS = [
    "The quick brown fox jumps over the lazy dog. It's 5 o'clock, isn't it?",
    "Xin chào, tôi tên là Long. Hôm nay trời đẹp quá!",
    "def add(a, b):\n    if a > b:\n        return a + b\n    return a - b  # sum",
    "3.14159 1,000,000 2026-10-03",
    "Emoji time \U0001F600\U0001F680 and a family \U0001F468\u200d\U0001F469\u200d\U0001F467",
    "  leading, trailing and   multiple   spaces  ",
    "line one\n\nline three\n\n\n\nline seven\n",
    "\u65e5\u672c\u8a9e\u3068\u4e2d\u6587 mixed with English",
]

_LIST_LINE = re.compile(r"^\s*\[[\s\d,\-]*\]\s*$")


def _ram_available() -> int:
    return int(psutil.virtual_memory().available)


def _read_header(path: Path) -> dict:
    """Header facts as plain values. Everything derived from the memory map is dropped before
    returning: on Windows a lingering map would block moving the file afterwards."""
    import gguf

    out: dict = {}
    reader = gguf.GGUFReader(str(path))
    try:
        fields = reader.fields

        def scalar(key: str):
            f = fields.get(key)
            if f is None:
                return None
            try:
                return f.contents()
            except Exception:
                return None

        arch = scalar("general.architecture")
        out["architecture"] = str(arch) if arch is not None else None
        if out["architecture"]:
            n = scalar(f"{out['architecture']}.block_count")
            out["n_layers"] = int(n) if isinstance(n, int) else None
        out["tokenizer_model"] = scalar("tokenizer.ggml.model")
        toks = fields.get("tokenizer.ggml.tokens")
        out["vocab_size"] = len(toks.data) if toks is not None else 0
        out["chat_template"] = "tokenizer.chat_template" in fields
        del scalar, fields, toks
    finally:
        del reader
    gc.collect()
    return out


async def _check_header(path: Path, v: Validation) -> bool:
    try:
        h = await asyncio.to_thread(_read_header, path)
    except Exception as e:
        v.header_ok = False
        v.errors.append(f"the converted file is not a readable GGUF: {type(e).__name__}: {e}")
        return False
    v.architecture = h.get("architecture")
    v.n_layers = h.get("n_layers")
    v.vocab_size = h.get("vocab_size") or 0
    v.chat_template = bool(h.get("chat_template"))
    if not v.architecture:
        v.errors.append("GGUF header has no general.architecture")
    elif v.n_layers is None:
        v.warnings.append(f"GGUF header has no {v.architecture}.block_count")
    if not h.get("tokenizer_model") or not v.vocab_size:
        v.errors.append("GGUF has no tokenizer (tokenizer.ggml.model / tokenizer.ggml.tokens "
                        "missing): the model cannot be served")
    v.header_ok = not v.errors
    return v.header_ok


async def _hf_ids(src_dir: Path, tc: Toolchain, allow_remote_code: bool) -> list[list[int]] | str:
    """Token ids per probe text from Hugging Face, or an error string."""
    cmd = tc.hf_tokenize_cmd(src_dir, allow_remote_code)
    payload = json.dumps(TOKENIZER_TEXTS).encode("utf-8")
    try:
        res = await tc.run(cmd, env=tc.convert_env(), stdin_data=payload, capture=True,
                           merge_stderr=False, timeout=600)
    except Exception as e:
        return f"could not run the Hugging Face tokenizer: {e}"
    if res.timed_out:
        return "the Hugging Face tokenizer timed out"
    try:
        data = json.loads(res.output.strip().splitlines()[-1])
    except (ValueError, IndexError):
        tail = (res.stderr or res.output).strip().splitlines()[-1:] or ["no output"]
        return f"the Hugging Face tokenizer printed no result (exit {res.code}): {tail[0]}"
    if "error" in data:
        return str(data["error"])
    ids = data.get("ids")
    if (not isinstance(ids, list) or len(ids) != len(TOKENIZER_TEXTS)
            or not all(isinstance(x, list) for x in ids)):
        return "the Hugging Face tokenizer returned an unexpected result"
    return ids


async def _gguf_ids(gguf_path: Path, tc: Toolchain, workdir: Path) -> list[list[int]] | str:
    ids: list[list[int]] = []
    for i, text in enumerate(TOKENIZER_TEXTS):
        pf = workdir / f"probe{i}.txt"
        await asyncio.to_thread(pf.write_bytes, text.encode("utf-8"))
        res = await tc.run(tc.tokenize_cmd(gguf_path, pf), capture=True, merge_stderr=False,
                           timeout=300)
        if res.timed_out:
            return "llama-tokenize timed out"
        if res.code != 0:
            tail = (res.stderr or res.output).strip().splitlines()[-1:] or [""]
            return f"llama-tokenize exited with code {res.code}: {tail[0]}".rstrip(": ")
        line = next((ln for ln in reversed(res.output.splitlines()) if _LIST_LINE.match(ln)), None)
        if line is None:
            return "llama-tokenize printed no token list"
        ids.append([int(x) for x in json.loads(line)])
    return ids


async def _check_tokenizer(gguf_path: Path, src_dir: Path, tc: Toolchain,
                           allow_remote_code: bool, v: Validation) -> None:
    hf = await _hf_ids(src_dir, tc, allow_remote_code)
    if isinstance(hf, str):
        v.tokenizer_ok = None
        v.warnings.append(f"tokenizer comparison skipped: {hf}")
        return
    tmp = Path(tempfile.mkdtemp(prefix=".probe-", dir=str(gguf_path.parent)))
    try:
        gg = await _gguf_ids(gguf_path, tc, tmp)
    finally:
        await asyncio.to_thread(_rm, tmp)
    if isinstance(gg, str):
        v.tokenizer_ok = None
        v.warnings.append(f"tokenizer comparison skipped: {gg}")
        return
    v.tokenizer_cases = [TokenizerCase(text=t, hf=h, gguf=g, match=h == g)
                         for t, h, g in zip(TOKENIZER_TEXTS, hf, gg)]
    bad = [c for c in v.tokenizer_cases if not c.match]
    v.tokenizer_ok = not bad
    if bad:
        v.warnings.append(f"tokenizer differs from Hugging Face on {len(bad)} of "
                          f"{len(v.tokenizer_cases)} probe texts: the model may behave differently "
                          "(odd outputs, broken multilingual text)")


def _rm(path: Path) -> None:
    import shutil
    shutil.rmtree(path, ignore_errors=True)


async def _check_generation(gguf_path: Path, tc: Toolchain, ram_available: Callable[[], int],
                            timeout_s: float | None, v: Validation) -> None:
    size = os.path.getsize(gguf_path)
    avail = ram_available()
    if size * 1.2 > avail:
        v.generation_ok = None
        v.warnings.append(
            f"generation test skipped: the model needs about {size * 1.2 / 2**30:.1f} GB of RAM "
            f"for a CPU run, {avail / 2**30:.1f} GB is available")
        return
    timeout = timeout_s if timeout_s is not None else 120 + 60 * (size / 2**30)
    res = await tc.run(tc.simple_cmd(gguf_path, "The capital of France is", 16), capture=True,
                       merge_stderr=False, timeout=timeout)
    if res.timed_out:
        v.generation_ok = None
        v.warnings.append(f"generation test timed out after {timeout:.0f} s (slow CPU, not "
                          "necessarily a broken model)")
        return
    text = res.output.strip()
    if res.code != 0:
        tail = (res.stderr or res.output).strip().splitlines()[-3:]
        v.generation_ok = False
        v.errors.append(f"llama-simple failed (exit {res.code}): " + " | ".join(tail))
        return
    if not text:
        v.generation_ok = False
        v.errors.append("llama-simple ran but generated no text")
        return
    v.generation_ok = True
    v.generation_sample = text[:400]


async def validate(gguf_path: Path, src_dir: Path, toolchain: Toolchain, *, generation: bool,
                   allow_remote_code: bool,
                   ram_available: Callable[[], int] = _ram_available,
                   generation_timeout_s: float | None = None) -> Validation:
    """Run every check. Never raises for a bad model: problems land in the result.

    A failed generation also puts a message in `errors`, but only header_ok False is a hard
    failure; a failed generation or tokenizer mismatch means needs_review."""
    gguf_path = Path(gguf_path)
    v = Validation()
    if not await _check_header(gguf_path, v):
        return v
    await _check_tokenizer(gguf_path, Path(src_dir), toolchain, allow_remote_code, v)
    if generation:
        await _check_generation(gguf_path, toolchain, ram_available, generation_timeout_s, v)
    return v
