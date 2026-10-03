"""validate(): GGUF header, tokenizer parity, generation, with fake tools (see make_toolkit)."""
from __future__ import annotations

import json
import sys

import gguf
import numpy as np
import pytest

from gpupool.converter.hf_tokenize import run as hf_run
from gpupool.converter.toolchain import run_tool
from gpupool.converter.validate import TOKENIZER_TEXTS, validate
from tests.test_converter_toolchain import make_toolkit

BIG_RAM = lambda: 1 << 40  # noqa: E731


def write_gguf(path, *, tokenizer=True, arch="llama", blocks=True, template=True):
    w = gguf.GGUFWriter(str(path), arch)
    if blocks:
        w.add_block_count(3)
    if tokenizer:
        w.add_tokenizer_model("gpt2")
        w.add_token_list(["a", "b", "c", "d"])
    if template:
        w.add_chat_template("{{ x }}")
    w.add_tensor("t", np.zeros((4, 4), dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture
def tc(tmp_path):
    return make_toolkit(tmp_path / "kit")


@pytest.fixture
def model(tmp_path):
    (tmp_path / "src").mkdir()
    return write_gguf(tmp_path / "m.gguf"), tmp_path / "src"


async def test_all_good(tc, model):
    gg, src = model
    v = await validate(gg, src, tc, generation=True, allow_remote_code=False, ram_available=BIG_RAM)
    assert v.header_ok and v.errors == []
    assert (v.architecture, v.n_layers, v.vocab_size, v.chat_template) == ("llama", 3, 4, True)
    assert v.tokenizer_ok is True and len(v.tokenizer_cases) == len(TOKENIZER_TEXTS)
    assert all(c.match for c in v.tokenizer_cases)
    assert v.generation_ok is True and "Paris" in v.generation_sample
    # the memory map must be gone: on Windows an open map blocks moving the file
    gg.replace(gg.with_name("moved.gguf"))


async def test_probe_texts_cover_required_cases():
    joined = "\n".join(TOKENIZER_TEXTS)
    assert "Xin chào, tôi tên là Long. Hôm nay trời đẹp quá!" in TOKENIZER_TEXTS
    assert "3.14159 1,000,000 2026-10-03" in TOKENIZER_TEXTS
    assert "日本語と中文" in joined and "    if" in joined and "   multiple   " in joined


async def test_unreadable_header_is_hard_failure(tc, tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf at all" * 10)
    v = await validate(bad, tmp_path, tc, generation=True, allow_remote_code=False)
    assert v.header_ok is False and v.errors
    assert v.tokenizer_ok is None and v.generation_ok is None  # nothing else ran


async def test_missing_tokenizer_is_hard_failure(tc, tmp_path):
    gg = write_gguf(tmp_path / "nt.gguf", tokenizer=False)
    v = await validate(gg, tmp_path, tc, generation=False, allow_remote_code=False)
    assert v.header_ok is False and any("tokenizer" in e for e in v.errors)


async def test_missing_block_count_only_warns(tc, tmp_path):
    gg = write_gguf(tmp_path / "nb.gguf", blocks=False)
    (tmp_path / "src").mkdir()
    v = await validate(gg, tmp_path / "src", tc, generation=False, allow_remote_code=False)
    assert v.header_ok and v.n_layers is None and any("block_count" in w for w in v.warnings)


async def test_tokenizer_mismatch(tc, model, monkeypatch):
    gg, src = model
    monkeypatch.setenv("FAKE_TOK_MISMATCH", "1")
    v = await validate(gg, src, tc, generation=False, allow_remote_code=False)
    assert v.header_ok and v.tokenizer_ok is False
    bad = [c for c in v.tokenizer_cases if not c.match]
    assert len(bad) == len(TOKENIZER_TEXTS) and bad[0].hf != bad[0].gguf
    assert any("differs" in w for w in v.warnings)
    assert v.generation_ok is None  # generation disabled


async def test_hf_side_error_is_unknown_not_failure(tc, model, monkeypatch):
    gg, src = model
    monkeypatch.setenv("FAKE_HF_ERROR", "1")
    v = await validate(gg, src, tc, generation=False, allow_remote_code=False)
    assert v.tokenizer_ok is None and v.tokenizer_cases == []
    assert any("no tokenizer" in w for w in v.warnings)


async def test_llama_tokenize_failure_is_unknown(tc, model, monkeypatch):
    gg, src = model
    monkeypatch.setenv("FAKE_TOK_FAIL", "1")
    v = await validate(gg, src, tc, generation=False, allow_remote_code=False)
    assert v.tokenizer_ok is None and any("llama-tokenize" in w for w in v.warnings)


async def test_hf_tokenize_gets_trust_flag(tc, model, monkeypatch):
    gg, src = model
    seen = []
    orig = tc.hf_tokenize_cmd
    tc.hf_tokenize_cmd = lambda d, trust: (seen.append(trust), orig(d, trust))[1]
    await validate(gg, src, tc, generation=False, allow_remote_code=True)
    await validate(gg, src, tc, generation=False, allow_remote_code=False)
    assert seen == [True, False]


async def test_generation_skipped_when_ram_short(tc, model):
    gg, src = model
    size = gg.stat().st_size
    v = await validate(gg, src, tc, generation=True, allow_remote_code=False,
                       ram_available=lambda: int(size * 1.1))
    assert v.generation_ok is None and any("skipped" in w for w in v.warnings)
    v = await validate(gg, src, tc, generation=True, allow_remote_code=False,
                       ram_available=lambda: int(size * 1.3))
    assert v.generation_ok is True


async def test_generation_crash(tc, model, monkeypatch):
    gg, src = model
    monkeypatch.setenv("FAKE_SIMPLE", "crash")
    v = await validate(gg, src, tc, generation=True, allow_remote_code=False, ram_available=BIG_RAM)
    assert v.generation_ok is False and v.header_ok is True
    assert any("unable to load model" in e for e in v.errors)


async def test_generation_empty_output(tc, model, monkeypatch):
    gg, src = model
    monkeypatch.setenv("FAKE_SIMPLE", "empty")
    v = await validate(gg, src, tc, generation=True, allow_remote_code=False, ram_available=BIG_RAM)
    assert v.generation_ok is False and any("no text" in e for e in v.errors)


async def test_generation_timeout_is_unknown(tc, model, monkeypatch):
    gg, src = model
    monkeypatch.setenv("FAKE_SIMPLE", "slow")
    v = await validate(gg, src, tc, generation=True, allow_remote_code=False, ram_available=BIG_RAM,
                       generation_timeout_s=1.5)
    assert v.generation_ok is None and any("timed out" in w for w in v.warnings)
    assert v.errors == []


async def test_probe_files_do_not_leak(tc, model):
    gg, src = model
    await validate(gg, src, tc, generation=False, allow_remote_code=False)
    assert [p.name for p in gg.parent.iterdir() if p.name.startswith(".probe")] == []


# ---- hf_tokenize.py (standalone script) ---------------------------------------------------------


def install_fake_transformers(monkeypatch, *, fail=False):
    import types

    calls = {}

    class Tok:
        def encode(self, text, add_special_tokens=True):
            calls["add_special_tokens"] = add_special_tokens
            return [ord(c) for c in text]

    class Auto:
        @staticmethod
        def from_pretrained(path, trust_remote_code=False):
            calls["path"], calls["trust"] = path, trust_remote_code
            if fail:
                raise OSError("no such tokenizer")
            return Tok()

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = Auto
    monkeypatch.setitem(sys.modules, "transformers", mod)
    return calls


def test_hf_tokenize_run_ok(monkeypatch):
    calls = install_fake_transformers(monkeypatch)
    out = hf_run(["/m"], json.dumps(["ab", "é"]))
    assert out == {"ids": [[97, 98], [233]]}
    assert calls == {"path": "/m", "trust": False, "add_special_tokens": False}
    hf_run(["/m", "1"], "[]")
    assert calls["trust"] is True


def test_hf_tokenize_run_errors_never_raise(monkeypatch):
    install_fake_transformers(monkeypatch, fail=True)
    assert "no such tokenizer" in hf_run(["/m"], '["x"]')["error"]
    assert "error" in hf_run(["/m"], "not json")
    assert "error" in hf_run(["/m"], '[1, 2]')
    assert "error" in hf_run([], "[]")


async def test_hf_tokenize_script_end_to_end_without_transformers(tmp_path):
    # Real subprocess run of the shipped script; transformers is absent here, so the contract
    # "print {error} and exit 0" is what is verified.
    from gpupool.converter import hf_tokenize

    res = await run_tool([sys.executable, hf_tokenize.__file__, str(tmp_path), "0"],
                         stdin_data=b'["hi"]', capture=True, merge_stderr=False)
    assert res.code == 0
    data = json.loads(res.output.strip().splitlines()[-1])
    assert "ids" in data or "error" in data
