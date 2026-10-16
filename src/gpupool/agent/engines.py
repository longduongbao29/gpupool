"""Engines other than llama.cpp: vLLM and SGLang.

They serve Hugging Face checkpoints directly (no GGUF), download them themselves into the agent's
Hugging Face cache, and shard a model over the GPUs of this server by tensor parallelism. Both
speak the OpenAI API (/v1/chat/completions, /v1/completions, /v1/embeddings...), answer /health
and export Prometheus /metrics, so the coordinator treats them like a llama-server head.

The agent does not install them: each one is a command (GPUPOOL_VLLM_CMD, GPUPOOL_SGLANG_CMD)
that the agent runs if it resolves on PATH. A virtualenv's bin/vllm, `uvx vllm` or a wrapper
script that runs a container all work.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from gpupool.common.models import HF_ENGINES, EngineSpec

ENGINE_FEATURE = "engine:"  # NodeReport.features entry: "engine:vllm", "engine:sglang"

DEFAULT_COMMANDS = {
    "vllm": "vllm serve",
    "sglang": "python3 -m sglang.launch_server",
}


def engine_feature(engine: str) -> str:
    return ENGINE_FEATURE + engine


def split_command(cmd: str) -> list[str]:
    return shlex.split(cmd, posix=os.name != "nt")


def _resolves(argv: list[str]) -> bool:
    if not argv:
        return False
    exe = argv[0]
    if os.sep in exe or (os.altsep and os.altsep in exe):
        return Path(exe).is_file() and os.access(exe, os.X_OK)
    return shutil.which(exe) is not None


def engine_argv(engine: str, commands: dict[str, str]) -> list[str]:
    return split_command(commands.get(engine) or DEFAULT_COMMANDS[engine])


_FIND_SPEC = "import importlib.util, sys; sys.exit(importlib.util.find_spec(sys.argv[1]) is None)"


def _module_present(python: str, module: str) -> bool:
    """`python -m pkg.mod`: pkg is installed for that interpreter. find_spec locates the package
    without importing it (importing sglang or vllm takes seconds)."""
    try:
        r = subprocess.run([python, "-c", _FIND_SPEC, module.split(".")[0]], capture_output=True,
                           timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def available_engines(commands: dict[str, str]) -> list[str]:
    """Engines whose command resolves on this machine (and, for `python -m <module>`, whose
    package that interpreter has)."""
    out = []
    for engine in sorted(HF_ENGINES):
        argv = engine_argv(engine, commands)
        if not _resolves(argv):
            continue
        if len(argv) >= 3 and argv[1] == "-m" and not _module_present(argv[0], argv[2]):
            continue
        out.append(engine)
    return out


def cuda_visible_devices(spec: EngineSpec) -> str:
    """CUDA_VISIBLE_DEVICES for the engine: the GPUs' UUIDs when known (stable whatever the
    agent's own view), else the indices of "CUDA<i>" (PCI bus order, like the agent's probe)."""
    if spec.device_uuids and len(spec.device_uuids) == len(spec.devices):
        return ",".join(spec.device_uuids)
    idx = []
    for d in spec.devices:
        if not d.startswith("CUDA") or not d[4:].isdigit():
            raise ValueError(f"{spec.engine} runs on CUDA devices only, got {d!r}")
        idx.append(d[4:])
    return ",".join(idx)


def build_hf_command(spec: EngineSpec, base: list[str], bind_host: str) -> list[str]:
    """The command line of a vLLM or SGLang server for `spec`."""
    if not spec.hf_repo:
        raise ValueError(f"{spec.engine} engine needs hf_repo")
    tp = str(len(spec.devices))
    name = spec.model or spec.hf_repo
    if spec.engine == "vllm":
        cmd = [*base, spec.hf_repo, "--host", bind_host, "--port", str(spec.port),
               "--served-model-name", name, "--tensor-parallel-size", tp,
               "--max-model-len", str(spec.ctx_size), "--max-num-seqs", str(spec.parallel),
               "--disable-uvicorn-access-log"]
        if spec.gpu_mem_fraction is not None:
            cmd += ["--gpu-memory-utilization", f"{spec.gpu_mem_fraction:.4f}"]
    elif spec.engine == "sglang":
        cmd = [*base, "--model-path", spec.hf_repo, "--host", bind_host, "--port", str(spec.port),
               "--served-model-name", name, "--tp-size", tp,
               "--context-length", str(spec.ctx_size),
               "--max-running-requests", str(spec.parallel), "--enable-metrics"]
        if spec.gpu_mem_fraction is not None:
            cmd += ["--mem-fraction-static", f"{spec.gpu_mem_fraction:.4f}"]
    else:
        raise ValueError(f"unknown engine {spec.engine!r}")
    if spec.hf_revision:
        cmd += ["--revision", spec.hf_revision]
    if spec.trust_remote_code:
        cmd += ["--trust-remote-code"]
    return cmd + list(spec.engine_args)


def hf_env(spec: EngineSpec, hf_home: Path | None) -> dict[str, str]:
    """Environment additions for an engine process."""
    env = {"CUDA_VISIBLE_DEVICES": cuda_visible_devices(spec), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    if hf_home is not None and not os.environ.get("HF_HOME"):
        env["HF_HOME"] = str(hf_home)  # keep downloads in the agent's cache, not ~/.cache
    return env
