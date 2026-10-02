from __future__ import annotations

import socket
import time
from pathlib import Path

import psutil
import pytest

from gpupool.agent.procs import ProcessManager, llama_version
from gpupool.common.models import EngineSpec

LLAMA_DIR = Path(__file__).resolve().parents[1] / ".cache" / "llama" / "b11342-cuda12.4"

pytestmark = pytest.mark.real


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_real_version():
    assert llama_version(LLAMA_DIR) == "b11342"


def test_real_rpc_server_lifecycle(tmp_path):
    pm = ProcessManager(LLAMA_DIR, tmp_path / "logs", "127.0.0.1")
    port = free_port()
    spec = EngineSpec(engine_id="real-rpc", kind="rpc", port=port, devices=["CPU"])
    try:
        st = pm.start(spec)
        pid = st.pid
        end = time.time() + 60
        while time.time() < end and pm.get("real-rpc").state == "starting":
            time.sleep(0.2)
        st = pm.get("real-rpc")
        assert st.state == "running", st.log_tail
        st = pm.stop("real-rpc")
        assert st.state == "exited"
        assert not psutil.pid_exists(pid)
    finally:
        pm.stop_all()


def test_real_split_command_is_accepted_by_llama_server(tmp_path):
    # Feed the exact split command build_command produces to the real argument parser.
    # Regression: "--device CUDA0,RPC0" before "--rpc" made llama-server exit with a usage
    # error, which every mocked test missed and the first real multi-node launch hit.
    import subprocess

    from gpupool.agent.procs import build_command, find_binaries

    pm = ProcessManager(LLAMA_DIR, tmp_path / "logs", "127.0.0.1")
    port = free_port()
    pm.start(EngineSpec(engine_id="rpc", kind="rpc", port=port, devices=["CPU"]))
    try:
        end = time.time() + 30
        while time.time() < end and pm.get("rpc").state != "running":
            time.sleep(0.2)
        spec = EngineSpec(engine_id="head", kind="server", port=free_port(),
                          devices=["CUDA0", "RPC0"], model="m",
                          rpc_endpoints=[f"127.0.0.1:{port}"], tensor_split=[1, 1])
        cmd = build_command(spec, find_binaries(LLAMA_DIR), "127.0.0.1", "unused.gguf")
        r = subprocess.run(cmd + ["--list-devices"], capture_output=True, text=True, timeout=60)
        out = r.stdout + r.stderr
        assert r.returncode == 0 and "RPC0" in out, out[-2000:]
    finally:
        pm.stop_all()


def test_real_gpu_telemetry():
    from gpupool.agent.gpu import probe_devices
    from gpupool.common.config import AgentConfig
    devs = probe_devices(AgentConfig(node_id="n", llama_dir=LLAMA_DIR, include_cpu=False))
    if not devs:
        pytest.skip("no NVML devices")
    d = devs[0]
    print("REAL", d.model_dump())
    assert d.temp_c is not None and 0 < d.temp_c < 120
    assert d.driver and d.cuda and "." in d.cuda
    assert d.power_w is None or d.power_w >= 0  # laptops may not report power
    assert isinstance(d.processes, list)
