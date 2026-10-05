"""Toolchain: discovery, command builders, supported architectures, the run helper.

Also home of `make_toolkit`: fake llama.cpp tools (plain python scripts, run through the
Toolchain script_tools test seam) shared by the validate and jobs tests. Fake behaviour is
steered with FAKE_* environment variables, which run_tool passes through to the tool.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import psutil
import pytest

from gpupool.converter.models import ConvertError
from gpupool.converter.toolchain import ImatrixProgress, Toolchain, run_tool

FAKE_CONVERT = r'''
import json, os, sys, time
args = sys.argv[1:]
if "--print-supported-models" in args:
    if os.environ.get("FAKE_PRINT_FAIL"):
        sys.stderr.write("boom\n"); sys.exit(2)
    sys.stderr.write("INFO:hf-to-gguf:TEXT models:\n")
    for n in ("LlamaForCausalLM", "Qwen2ForCausalLM"):
        sys.stderr.write("INFO:hf-to-gguf:  - %s\n" % n)
    sys.stderr.write("INFO:hf-to-gguf:MMPROJ models:\n")
    sys.stderr.write("INFO:hf-to-gguf:  - LlavaForConditionalGeneration\n")
    sys.exit(0)
src = args[0]
out = args[args.index("--outfile") + 1]
outtype = args[args.index("--outtype") + 1]
if os.environ.get("FAKE_SEEN_FILE"):
    seen = sorted(os.path.relpath(os.path.join(d, f), src).replace(os.sep, "/")
                  for d, _, fs in os.walk(src) for f in fs)
    with open(os.environ["FAKE_SEEN_FILE"], "a") as fh:
        fh.write(json.dumps({"seen": seen, "outtype": outtype,
                             "offline": os.environ.get("HF_HUB_OFFLINE"),
                             "no_local_gguf": os.environ.get("NO_LOCAL_GGUF")}) + "\n")
mode = os.environ.get("FAKE_CONVERT_MODE", "ok")
if mode == "fail":
    sys.stderr.write("INFO:hf-to-gguf:Loading model: foo\n")
    sys.stderr.write("ERROR:hf-to-gguf:Model Foo is not supported\n")
    sys.exit(1)
if os.environ.get("FAKE_PID_FILE"):
    open(os.environ["FAKE_PID_FILE"], "w").write(str(os.getpid()))
import numpy as np
import gguf
w = gguf.GGUFWriter(out, "llama")
w.add_block_count(2)
if mode != "notok":
    w.add_tokenizer_model("gpt2")
    w.add_token_list(["a", "b", "c"])
w.add_chat_template("{{ messages }}")
w.add_tensor("t", np.zeros((8, 8), dtype=np.float32))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
for pct in (0, 50, 100):
    sys.stderr.write("\rWriting: %3d%%|#####     | %d/2 [00:00<?, ?it/s]" % (pct, pct // 50))
    sys.stderr.flush()
    time.sleep(0.05)
sys.stderr.write("\n")
if mode == "slow":
    time.sleep(120)
'''

FAKE_QUANTIZE = r'''
import json, os, shutil, sys
args = sys.argv[1:]
if os.environ.get("FAKE_ARGV_LOG"):
    with open(os.environ["FAKE_ARGV_LOG"], "a") as fh:
        fh.write(json.dumps(args) + "\n")
pos = [a for a in args if not a.startswith("--")]
# flags with a value: drop their values from the positional list
for flag in ("--output-tensor-type", "--token-embedding-type", "--imatrix"):
    if flag in args:
        pos.remove(args[args.index(flag) + 1])
src, dst = pos[0], pos[1]
# Mirrors llama-quantize b11342: these types are refused without an importance matrix.
NEEDS = {"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS"}
if pos[2] in NEEDS and "--imatrix" not in args:
    sys.stderr.write("============================================================\n")
    sys.stderr.write("Missing importance matrix for tensor blk.0.attn_k.weight in a very low-bit quantization\n")
    sys.stderr.write("The result will be garbage, so bailing out\n")
    sys.exit(1)
if "--imatrix" in args and not os.path.isfile(args[args.index("--imatrix") + 1]):
    sys.stderr.write("failed to load imatrix file\n"); sys.exit(1)
if os.environ.get("FAKE_QUANT_MODE") == "fail":
    sys.stderr.write("llama_model_quantize: failed to quantize: boom\n"); sys.exit(1)
for i in (1, 2):
    print("[%4d/%4d] blk.%d.weight - [8, 8], type = f32, size = 0.000 MB" % (i, 2, i), flush=True)
shutil.copyfile(src, dst)
'''

FAKE_IMATRIX = r'''
import json, os, sys, time
args = sys.argv[1:]
model = args[args.index("-m") + 1]
calib = args[args.index("-f") + 1]
out = args[args.index("-o") + 1]
chunks = int(args[args.index("--chunks") + 1])
if os.environ.get("FAKE_IMATRIX_LOG"):
    with open(os.environ["FAKE_IMATRIX_LOG"], "a") as fh:
        fh.write(json.dumps({"args": args, "calib": open(calib, encoding="utf-8").read(),
                             "model_exists": os.path.isfile(model)}) + "\n")
mode = os.environ.get("FAKE_IMATRIX_MODE", "ok")
if mode == "fail":
    sys.stderr.write("compute_imatrix: the data file you provided tokenizes to only 3 tokens\n")
    sys.exit(1)
sys.stderr.write("compute_imatrix: computing over %d chunks, n_ctx=512, batch_size=2048, n_seq=4\n" % chunks)
sys.stderr.write("compute_imatrix: 0.05 seconds per pass - ETA 0.02 minutes\n")
sys.stderr.flush()
if mode == "slow":
    time.sleep(120)
if mode != "nofile":
    with open(out, "wb") as fh:
        fh.write(b"GGUF-imatrix")
'''

FAKE_TOKENIZE = r'''
import os, sys
args = sys.argv[1:]
data = open(args[args.index("-f") + 1], "rb").read()
ids = list(data)
if os.environ.get("FAKE_TOK_MISMATCH") and ids:
    ids[0] += 1
if os.environ.get("FAKE_TOK_FAIL"):
    sys.stderr.write("error: could not load model\n"); sys.exit(1)
sys.stderr.write("llama_model_loader: loaded meta data\n")
print("[" + ", ".join(str(i) for i in ids) + "]")
'''

FAKE_SIMPLE = r'''
import os, sys, time
mode = os.environ.get("FAKE_SIMPLE", "ok")
sys.stderr.write("main: decoded 16 tokens\n")
if mode == "crash":
    sys.stderr.write("main: error: unable to load model\n"); sys.exit(1)
if mode == "slow":
    time.sleep(60)
if mode == "empty":
    sys.exit(0)
sys.stdout.write(sys.argv[-1] + " Paris, a city of light\n")
'''

FAKE_HF_TOKENIZE = r'''
import json, os, sys
texts = json.loads(sys.stdin.read())
if os.environ.get("FAKE_HF_ERROR"):
    print(json.dumps({"error": "OSError: no tokenizer"})); sys.exit(0)
print(json.dumps({"ids": [list(t.encode("utf-8")) for t in texts]}))
'''


def make_toolkit(root: Path, imatrix: bool = True) -> Toolchain:
    cdir = root / "llama.cpp"
    (cdir / "conversion").mkdir(parents=True)
    (cdir / "convert_hf_to_gguf.py").write_text(FAKE_CONVERT, encoding="utf-8")
    tools = root / "bin"
    tools.mkdir()
    for name, src in (("llama-quantize", FAKE_QUANTIZE), ("llama-tokenize", FAKE_TOKENIZE),
                      ("llama-simple", FAKE_SIMPLE),
                      *((("llama-imatrix", FAKE_IMATRIX),) if imatrix else ())):
        (tools / f"{name}.py").write_text(src, encoding="utf-8")
    hf = root / "hf_tokenize_fake.py"
    hf.write_text(FAKE_HF_TOKENIZE, encoding="utf-8")
    return Toolchain(cdir, sys.executable, tools, script_tools=True, hf_tokenize_script=hf)


# ---- problem() -----------------------------------------------------------------------------


def test_problem_none_when_complete(tmp_path):
    assert make_toolkit(tmp_path).problem() is None


def test_problem_unconfigured():
    assert "GPUPOOL_CONVERT_DIR" in Toolchain(None).problem()


def test_problem_missing_pieces(tmp_path):
    tc = make_toolkit(tmp_path)
    (tc.tools_dir / "llama-simple.py").unlink()
    assert "llama-simple" in tc.problem()
    (tc.convert_dir / "conversion").rmdir()
    assert "conversion" in tc.problem()
    (tc.convert_dir / "convert_hf_to_gguf.py").unlink()
    assert "convert_hf_to_gguf.py" in tc.problem()


def test_problem_bad_python(tmp_path):
    tc = make_toolkit(tmp_path)
    tc.python = str(tmp_path / "no-such-python")
    assert "not runnable" in tc.problem()


def test_script_tools_are_off_by_default(tmp_path):
    tc = make_toolkit(tmp_path)
    tc.script_tools = False
    assert "llama-quantize" in tc.problem()


def test_exe_suffix_accepted(tmp_path):
    tc = make_toolkit(tmp_path)
    tc.script_tools = False
    for n in ("llama-quantize", "llama-tokenize", "llama-simple"):
        (tc.tools_dir / f"{n}.exe").write_bytes(b"")
    assert tc.problem() is None


# ---- supported architectures -----------------------------------------------------------------


async def test_supported_architectures_parses_text_section_only(tmp_path):
    tc = make_toolkit(tmp_path)
    archs = await tc.supported_architectures()
    assert archs == {"LlamaForCausalLM", "Qwen2ForCausalLM"}
    # cached: the converter is not run again
    (tc.convert_dir / "convert_hf_to_gguf.py").write_text("raise SystemExit(9)")
    assert await tc.supported_architectures() == archs


async def test_supported_architectures_none_on_failure(tmp_path, monkeypatch):
    tc = make_toolkit(tmp_path)
    monkeypatch.setenv("FAKE_PRINT_FAIL", "1")
    assert await tc.supported_architectures() is None
    monkeypatch.delenv("FAKE_PRINT_FAIL")
    assert await tc.supported_architectures() == {"LlamaForCausalLM", "Qwen2ForCausalLM"}


async def test_supported_architectures_none_when_unusable():
    assert await Toolchain(None).supported_architectures() is None


# ---- command builders -------------------------------------------------------------------------


def test_convert_cmd_and_env(tmp_path):
    tc = make_toolkit(tmp_path)
    cmd = tc.convert_cmd(tmp_path / "src", tmp_path / "o.gguf", "bf16")
    assert cmd[0] == sys.executable and cmd[1].endswith("convert_hf_to_gguf.py")
    assert cmd[-4:] == ["--outfile", str(tmp_path / "o.gguf"), "--outtype", "bf16"]
    env = tc.convert_env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert "NO_LOCAL_GGUF" not in env
    with pytest.raises(ConvertError):
        tc.convert_cmd(tmp_path, tmp_path / "o.gguf", "f16; rm -rf /")


def test_quantize_cmd_maps_advanced_flags(tmp_path):
    tc = make_toolkit(tmp_path)
    cmd = tc.quantize_cmd(Path("in.gguf"), Path("out.gguf"), "Q4_K_M", threads=6,
                          leave_output_tensor=True, pure=True, output_tensor_type="q8_0",
                          token_embedding_type="Q6_K")
    i = cmd.index("--leave-output-tensor")
    assert cmd[i:i + 7] == ["--leave-output-tensor", "--pure", "--output-tensor-type", "q8_0",
                            "--token-embedding-type", "Q6_K", "in.gguf"]
    assert cmd[-3:] == ["out.gguf", "Q4_K_M", "6"]


def test_quantize_cmd_minimal_and_no_thread_arg(tmp_path):
    tc = make_toolkit(tmp_path)
    cmd = tc.quantize_cmd(Path("a"), Path("b"), "Q8_0")
    assert cmd[-3:] == ["a", "b", "Q8_0"] and not any(c.startswith("--") for c in cmd[2:])


@pytest.mark.parametrize("bad", ["q8_0 --evil", "--pure", "Q4_K_M\n", "", "nope", "q4_k_m"])
def test_quantize_cmd_rejects_unknown_types(tmp_path, bad):
    tc = make_toolkit(tmp_path)
    for kw in ({"output_tensor_type": bad}, {"token_embedding_type": bad}):
        if not bad:
            continue  # empty = option not given, never reaches the command line
        with pytest.raises(ConvertError):
            tc.quantize_cmd(Path("a"), Path("b"), "Q4_K_M", **kw)


def test_tokenize_and_simple_cmds(tmp_path):
    tc = make_toolkit(tmp_path)
    t = tc.tokenize_cmd(Path("m.gguf"), Path("p.txt"))
    assert {"--ids", "--no-bos", "--no-parse-special", "--no-escape"} <= set(t)
    assert t[t.index("-f") + 1] == "p.txt" and t[t.index("-m") + 1] == "m.gguf"
    s = tc.simple_cmd(Path("m.gguf"), "Hello", 16)
    assert s[-1] == "Hello" and s[s.index("-n") + 1] == "16" and s[s.index("-ngl") + 1] == "0"


# ---- run_tool -------------------------------------------------------------------------------


async def test_run_tool_splits_cr_and_lf_and_captures():
    code = ("import sys\n"
            "sys.stderr.write('a\\rb\\rc\\nlast\\n')\n"
            "sys.stdout.write('out1\\n')\n")
    lines: list[str] = []
    res = await run_tool([sys.executable, "-c", code], on_line=lines.append, capture=True)
    assert res.code == 0
    assert {"a", "b", "c", "last", "out1"} <= set(lines)
    assert lines.index("a") < lines.index("b") < lines.index("c")


async def test_run_tool_separate_stderr():
    code = "import sys; print('o'); sys.stderr.write('e\\n')"
    res = await run_tool([sys.executable, "-c", code], capture=True, merge_stderr=False)
    assert res.output.strip() == "o" and res.stderr.strip() == "e"


async def test_run_tool_stdin_and_exit_code():
    code = "import sys; d=sys.stdin.read(); print(len(d)); sys.exit(3)"
    res = await run_tool([sys.executable, "-c", code], stdin_data=b"x" * 200000, capture=True)
    assert res.code == 3 and res.output.strip() == "200000"


async def test_run_tool_timeout_kills(tmp_path):
    pidf = tmp_path / "pid"
    code = f"import os,time; open(r'{pidf}','w').write(str(os.getpid())); time.sleep(60)"
    t0 = time.monotonic()
    res = await run_tool([sys.executable, "-c", code], timeout=1.5, capture=True)
    assert res.timed_out and res.code is None and time.monotonic() - t0 < 20
    assert not psutil.pid_exists(int(pidf.read_text()))


async def test_run_tool_cancel_kills_process_tree(tmp_path):
    pidf = tmp_path / "pids"
    child = ("import os,time,subprocess,sys;"
             f"p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
             f"open(r'{pidf}','w').write(str(os.getpid())+','+str(p.pid));time.sleep(60)")
    task = asyncio.create_task(run_tool([sys.executable, "-c", child]))
    for _ in range(100):
        await asyncio.sleep(0.1)
        if pidf.exists() and "," in pidf.read_text():
            break
    parent, kid = (int(x) for x in pidf.read_text().split(","))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    assert not psutil.pid_exists(parent)
    # The grandchild is reparented to PID 1 once its parent dies. Under an init that does not
    # reap (some container runtimes) it stays a zombie: killed, but its pid still exists.
    assert not psutil.pid_exists(kid) or psutil.Process(kid).status() == psutil.STATUS_ZOMBIE


async def test_run_tool_survives_broken_callback():
    def bad(_line):
        raise RuntimeError("x")

    res = await run_tool([sys.executable, "-c", "print('hi')"], on_line=bad, capture=True)
    assert res.code == 0 and res.output.strip() == "hi"


async def test_run_tool_env_passthrough(monkeypatch):
    res = await run_tool([sys.executable, "-c", "import os;print(os.environ['ZZ'])"],
                         env={"ZZ": "1"}, capture=True)
    assert res.output.strip() == "1"
    assert json.loads(json.dumps(res.code)) == 0


async def test_run_tool_flushes_unterminated_last_line():
    res = await run_tool([sys.executable, "-c", "import sys; sys.stdout.write('no newline')"],
                         capture=True)
    assert res.output == "no newline"


# ---- llama-imatrix --------------------------------------------------------------------------


def test_imatrix_is_optional(tmp_path):
    tc = make_toolkit(tmp_path)
    assert tc.has_imatrix() and tc.problem() is None
    lean = make_toolkit(tmp_path / "lean", imatrix=False)
    assert not lean.has_imatrix()
    assert lean.problem() is None  # missing llama-imatrix never disables conversion
    with pytest.raises(ConvertError) as e:
        lean.imatrix_cmd(Path("m.gguf"), Path("c.txt"), Path("o.gguf"))
    assert e.value.status == 503 and "llama-imatrix is not installed" in e.value.message
    assert not Toolchain(None).has_imatrix()


def test_imatrix_cmd_flags(tmp_path):
    tc = make_toolkit(tmp_path)
    cmd = tc.imatrix_cmd(Path("m.gguf"), Path("c.txt"), Path("o.gguf"), chunks=7, threads=3)
    flags = cmd[2:]  # after [python, script]
    assert flags[:6] == ["-m", "m.gguf", "-f", "c.txt", "-o", "o.gguf"]
    assert flags[flags.index("--chunks") + 1] == "7" and "--no-ppl" in flags
    assert flags[flags.index("-ngl") + 1] == "0" and flags[flags.index("-c") + 1] == "512"
    assert flags[flags.index("-t") + 1] == "3"
    assert "-t" not in tc.imatrix_cmd(Path("m"), Path("c"), Path("o"))
    assert tc.imatrix_cmd(Path("m"), Path("c"), Path("o"))[2:][flags.index("--chunks")] == "--chunks"


def test_quantize_cmd_with_imatrix(tmp_path):
    tc = make_toolkit(tmp_path)
    cmd = tc.quantize_cmd(Path("i.gguf"), Path("o.gguf"), "IQ2_M", imatrix=Path("im.gguf"), threads=2)
    args = cmd[2:]
    assert args[:2] == ["--imatrix", "im.gguf"] and args[-4:] == ["i.gguf", "o.gguf", "IQ2_M", "2"]
    assert "--imatrix" not in tc.quantize_cmd(Path("i"), Path("o"), "Q4_K_M")


def test_imatrix_progress_parsing():
    p = ImatrixProgress()
    assert p.fraction(0) is None
    p.feed("compute_imatrix: computing over 100 chunks, n_ctx=512, batch_size=2048, n_seq=4", 10.0)
    assert p.chunks == 100 and p.fraction(10.0) is None
    p.feed("compute_imatrix: 5.00 seconds per pass - ETA 2.00 minutes", 15.0)
    assert p.fraction(15.0) == pytest.approx(5 / 120)
    assert p.fraction(75.0) == pytest.approx(65 / 120)
    assert p.fraction(10_000.0) == 0.99  # the clock alone never claims completion
    h = ImatrixProgress()
    h.feed("computing over 10 chunks, n_ctx=512", 0)
    h.feed("compute_imatrix: 1.0 seconds per pass - ETA 1 hours 2.50 minutes", 0)
    assert h.fraction(1.0 + 3750 / 2) == pytest.approx(0.5, abs=0.01)
    h.feed("[3]5.1234,[4]5.0001,", 1.0)  # per-chunk entries (perplexity on) win when present
    assert h.fraction(5.0) == pytest.approx(0.4)
