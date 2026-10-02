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
