"""The llama.cpp conversion toolchain: where the tools are, how to run them, how to stop them.

Layout (see the Docker image): `convert_dir` holds convert_hf_to_gguf.py, conversion/ and
gguf-py/; `python` is an interpreter with the converter's dependencies (torch, transformers);
`tools_dir` holds the CPU build of llama-quantize, llama-tokenize and llama-simple.

Test seam (explicit, off by default): `Toolchain(..., script_tools=True)` additionally accepts
`<tool>.py` scripts in tools_dir, run with the converter's python. Real deployments never set it;
tests use it to run fake tools on Windows and Linux. The hf_tokenize script path can likewise be
replaced with `hf_tokenize_script=`.
"""
from __future__ import annotations

import asyncio
import codecs
import logging
import os
import re
import shutil
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

log = logging.getLogger(__name__)

_EXE = ".exe" if os.name == "nt" else ""
TOOL_NAMES = ("llama-quantize", "llama-tokenize", "llama-simple")

# ggml type names llama-quantize accepts for --output-tensor-type / --token-embedding-type
# (it matches case-insensitively). Whitelisted so a user-supplied value can never become an
# extra command-line argument.
GGML_TYPES = frozenset({
    "f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_k", "q3_k", "q4_k", "q5_k",
    "q6_k", "iq4_nl", "iq4_xs", "iq3_xxs", "iq3_s", "iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m",
    "tq1_0", "tq2_0", "mxfp4",
})
CONVERT_OUTTYPES = frozenset({"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"})

_ARCH_LINE = re.compile(r"^(?:[A-Z]+:[\w.\-]+:)?\s*-\s+(\S+)\s*$")
_MAX_CAPTURE = 1 << 20


def check_ggml_type(value: str, what: str) -> str:
    if value.strip().lower() not in GGML_TYPES:
        from gpupool.converter.models import ConvertError
        raise ConvertError(f"{what}: unknown ggml type {value!r}", 400)
    return value.strip()


@dataclass
class RunResult:
    code: int | None  # None when the tool was killed by the timeout
    output: str  # captured stdout (stdout+stderr when merged); "" unless capture=True
    stderr: str = ""  # only when merge_stderr=False and capture=True
    timed_out: bool = False


def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the tool and everything it spawned (the converter forks helpers)."""
    pid = proc.pid
    if pid is None or proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            try:
                for child in psutil.Process(pid).children(recursive=True):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            proc.kill()
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except ProcessLookupError:
        pass


async def _pump(stream: asyncio.StreamReader, on_line: Callable[[str], None] | None,
                sink: list[str] | None) -> None:
    """Read a stream, calling on_line for every line. tqdm redraws with "\\r", so both \\r and
    \\n end a line; a huge unterminated chunk is flushed instead of buffered forever."""
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    captured = 0

    def emit(line: str) -> None:
        nonlocal captured
        if sink is not None and captured < _MAX_CAPTURE:
            sink.append(line)
            captured += len(line) + 1
        if on_line is not None:
            try:
                on_line(line)
            except Exception:  # a broken callback must never break the pipe reader
                log.exception("line callback failed")

    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        buf += dec.decode(chunk)
        parts = re.split(r"[\r\n]", buf)
        buf = parts.pop()
        for p in parts:
            if p.strip():
                emit(p)
        if len(buf) > 65536:
            emit(buf)
            buf = ""
    buf += dec.decode(b"", final=True)
    if buf.strip():
        emit(buf)


async def run_tool(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None,
                   on_line: Callable[[str], None] | None = None, stdin_data: bytes | None = None,
                   timeout: float | None = None, capture: bool = False,
                   merge_stderr: bool = True) -> RunResult:
    """Run a tool at low priority in its own process group, stream its output by line.

    Cancelling the awaiting task kills the whole process tree and waits for it to exit, then
    re-raises CancelledError, so a cancelled job leaves nothing running and no open file handles.
    A timeout kills the tree and returns timed_out=True (never raises)."""
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", **(env or {})}
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        kwargs["start_new_session"] = True
        kwargs["preexec_fn"] = lambda: os.nice(10)  # keep inference on this host responsive
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(cwd) if cwd else None, env=full_env,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.PIPE,
        **kwargs)
    out: list[str] = []
    err: list[str] = []
    readers = [asyncio.ensure_future(_pump(proc.stdout, on_line, out if capture else None))]
    if not merge_stderr:
        readers.append(asyncio.ensure_future(_pump(proc.stderr, on_line, err if capture else None)))

    async def feed() -> None:
        if stdin_data is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(stdin_data)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    async def finish() -> int:
        await feed()
        code = await proc.wait()
        await asyncio.gather(*readers)
        return code

    waiter = asyncio.ensure_future(finish())
    try:
        code = await asyncio.wait_for(asyncio.shield(waiter), timeout)
        return RunResult(code, "\n".join(out), "\n".join(err))
    except asyncio.TimeoutError:
        _kill_tree(proc)
        await asyncio.gather(waiter, return_exceptions=True)
        return RunResult(None, "\n".join(out), "\n".join(err), timed_out=True)
    except asyncio.CancelledError:
        _kill_tree(proc)
        await asyncio.gather(waiter, return_exceptions=True)
        raise
    finally:
        for r in readers:
            if not r.done():
                r.cancel()


class Toolchain:
    def __init__(self, convert_dir: Path | None, python: str = "python3",
                 tools_dir: Path | None = None, *, script_tools: bool = False,
                 hf_tokenize_script: Path | None = None):
        self.convert_dir = Path(convert_dir) if convert_dir else None
        self.python = python
        self.tools_dir = Path(tools_dir) if tools_dir else None
        self.script_tools = script_tools  # test seam, see module docstring
        self.hf_tokenize_script = (Path(hf_tokenize_script) if hf_tokenize_script
                                   else Path(__file__).with_name("hf_tokenize.py"))
        self._archs: set[str] | None = None

    # ---- discovery ---------------------------------------------------------------------

    def _tool_cmd(self, name: str) -> list[str] | None:
        if self.tools_dir is None:
            return None
        for cand in (self.tools_dir / f"{name}{_EXE}", self.tools_dir / name,
                     self.tools_dir / f"{name}.exe"):
            if cand.is_file():
                return [str(cand)]
        if self.script_tools:
            script = self.tools_dir / f"{name}.py"
            if script.is_file():
                return [self.python, str(script)]
        return None

    def problem(self) -> str | None:
        """None when conversion can run, else a human explanation of what is missing."""
        if self.convert_dir is None:
            return ("conversion is not set up: GPUPOOL_CONVERT_DIR is not configured "
                    "(the Docker image ships it at /opt/llama.cpp)")
        script = self.convert_dir / "convert_hf_to_gguf.py"
        if not script.is_file():
            return f"{script} not found (GPUPOOL_CONVERT_DIR must hold llama.cpp's converter)"
        if not (self.convert_dir / "conversion").is_dir():
            return f"{self.convert_dir / 'conversion'} not found: the converter needs its conversion/ package"
        if not (shutil.which(self.python) or Path(self.python).is_file()):
            return f"converter python {self.python!r} is not runnable (GPUPOOL_CONVERT_PYTHON)"
        if self.tools_dir is None:
            return "GPUPOOL_LLAMA_TOOLS_DIR is not configured (llama-quantize, llama-tokenize, llama-simple)"
        missing = [n for n in TOOL_NAMES if self._tool_cmd(n) is None]
        if missing:
            return f"{', '.join(missing)} not found in {self.tools_dir}"
        return None

    async def supported_architectures(self) -> set[str] | None:
        """Architectures the pinned converter knows (TEXT models), cached. None = unavailable."""
        if self._archs is not None:
            return self._archs
        if self.problem() is not None:
            return None
        cmd = [self.python, str(self.convert_dir / "convert_hf_to_gguf.py"), "--print-supported-models"]
        try:
            res = await run_tool(cmd, cwd=self.convert_dir, env=self.convert_env(), capture=True,
                                 timeout=300)
        except Exception as e:
            log.warning("could not list supported architectures: %s", e)
            return None
        if res.code != 0:
            tail = " | ".join(res.output.splitlines()[-3:])
            log.warning("--print-supported-models exited %s (%s)", res.code,
                        "timed out" if res.timed_out else tail)
            return None
        archs = self._parse_supported(res.output)
        if not archs:
            log.warning("--print-supported-models printed no architectures: %s",
                        " | ".join(res.output.splitlines()[-3:]))
            return None
        self._archs = archs
        return archs

    @staticmethod
    def _parse_supported(text: str) -> set[str]:
        archs: set[str] = set()
        in_text = False
        for line in text.splitlines():
            if "TEXT models" in line:
                in_text = True
                continue
            if "MMPROJ models" in line:
                break
            if in_text:
                m = _ARCH_LINE.match(line)
                if m:
                    archs.add(m.group(1))
        return archs

    # ---- command builders --------------------------------------------------------------

    @staticmethod
    def convert_env() -> dict[str, str]:
        # Offline: the converter must not fetch tokenizer code or files. NO_LOCAL_GGUF stays
        # unset so the gguf-py shipped next to the converter (same version) is the one used.
        return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    def convert_cmd(self, src_dir: Path, outfile: Path, outtype: str) -> list[str]:
        if outtype not in CONVERT_OUTTYPES:
            from gpupool.converter.models import ConvertError
            raise ConvertError(f"unknown converter output type {outtype!r}", 400)
        assert self.convert_dir is not None
        return [self.python, str(self.convert_dir / "convert_hf_to_gguf.py"), str(src_dir),
                "--outfile", str(outfile), "--outtype", outtype]

    def quantize_cmd(self, src: Path, dst: Path, qtype: str, *, threads: int = 0,
                     leave_output_tensor: bool = False, pure: bool = False,
                     output_tensor_type: str | None = None,
                     token_embedding_type: str | None = None) -> list[str]:
        base = self._tool_cmd("llama-quantize")
        assert base is not None
        cmd = list(base)
        if leave_output_tensor:
            cmd.append("--leave-output-tensor")
        if pure:
            cmd.append("--pure")
        if output_tensor_type:
            cmd += ["--output-tensor-type", check_ggml_type(output_tensor_type, "output_tensor_type")]
        if token_embedding_type:
            cmd += ["--token-embedding-type",
                    check_ggml_type(token_embedding_type, "token_embedding_type")]
        cmd += [str(src), str(dst), qtype]
        if threads > 0:
            cmd.append(str(threads))
        return cmd

    def tokenize_cmd(self, model: Path, prompt_file: Path) -> list[str]:
        """llama-tokenize at b11342: -f reads the file verbatim (--stdin/-p do too, but only -f
        keeps every byte on all platforms); --ids prints "[1, 2, 3]"; --no-escape because the
        default processes \\n-style escapes, which would alter code samples."""
        base = self._tool_cmd("llama-tokenize")
        assert base is not None
        return [*base, "-m", str(model), "-f", str(prompt_file), "--ids", "--no-bos",
                "--no-parse-special", "--no-escape"]

    def simple_cmd(self, model: Path, prompt: str, n_predict: int = 16) -> list[str]:
        """llama-simple at b11342: `-m model [-n N] [-ngl N] [prompt...]`; generated text goes to
        stdout, timing/log lines to stderr. -ngl 0: validation always runs on the CPU."""
        base = self._tool_cmd("llama-simple")
        assert base is not None
        return [*base, "-m", str(model), "-n", str(n_predict), "-ngl", "0", prompt]

    def hf_tokenize_cmd(self, model_dir: Path, trust_remote_code: bool) -> list[str]:
        return [self.python, str(self.hf_tokenize_script), str(model_dir),
                "1" if trust_remote_code else "0"]

    # ---- running -----------------------------------------------------------------------

    async def run(self, cmd: list[str], **kw) -> RunResult:
        return await run_tool(cmd, **kw)
