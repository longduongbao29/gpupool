from __future__ import annotations

import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from gpupool.agent import procs
from gpupool.agent.gpu import probe_devices
from gpupool.agent.models_cache import ensure_model, list_models
from gpupool.agent.procs import EngineExists, PortInUse, ProcessManager, build_command
from gpupool.common.config import AgentConfig
from gpupool.common.models import EngineSpec

BINS = {"server": Path("llama-server"), "rpc": Path("ggml-rpc-server")}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def make_cfg(tmp_path, **kw) -> AgentConfig:
    return AgentConfig(node_id="n1", llama_dir=tmp_path, cache_dir=tmp_path / "cache",
                       log_dir=tmp_path / "logs", **kw)


# ---------- build_command ----------

def test_build_rpc():
    spec = EngineSpec(engine_id="r-rpc-CUDA0", kind="rpc", port=9001, devices=["CUDA0"])
    assert build_command(spec, BINS, "10.0.0.5", None) == [
        "ggml-rpc-server", "-H", "10.0.0.5", "-p", "9001", "-d", "CUDA0", "-c"]


def test_build_server_single():
    spec = EngineSpec(engine_id="r-head", kind="server", port=9000, devices=["CUDA0"],
                      model="m", ctx_size=2048, parallel=2, extra_args=["--foo"])
    cmd = build_command(spec, BINS, "127.0.0.1", "/m.gguf")
    assert cmd == ["llama-server", "-m", "/m.gguf", "--host", "127.0.0.1", "--port", "9000",
                   "--alias", "m", "-c", "2048", "-np", "2", "-ngl", "999", "--device", "CUDA0",
                   "--split-mode", "layer", "--cache-reuse", "256", "--metrics", "--fit", "off",
                   "--foo"]


def test_build_server_split():
    spec = EngineSpec(engine_id="r-head", kind="server", port=9000,
                      devices=["CUDA0", "RPC0", "RPC1"], model="m",
                      rpc_endpoints=["a:1", "b:2"], tensor_split=[10, 0.5, 2.25])
    cmd = build_command(spec, BINS, "h", "/m.gguf")
    assert cmd[cmd.index("--device") + 1] == "CUDA0,RPC0,RPC1"
    assert cmd[cmd.index("--rpc") + 1] == "a:1,b:2"
    assert cmd[cmd.index("--tensor-split") + 1] == "10,0.5,2.25"
    # llama.cpp resolves --device names at parse time: RPC devices exist only after --rpc
    assert cmd.index("--rpc") < cmd.index("--device")


def test_build_server_requires_model_path():
    spec = EngineSpec(engine_id="x", kind="server", port=1, devices=["CUDA0"], model="m")
    with pytest.raises(ValueError):
        build_command(spec, BINS, "h", None)


def test_find_binaries_fallback_and_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found in llama dir"):
        procs.find_binaries(tmp_path)
    ext = procs._EXE
    (tmp_path / f"llama-server{ext}").write_text("x")
    (tmp_path / f"rpc-server{ext}").write_text("x")
    assert procs.find_binaries(tmp_path)["rpc"].name == f"rpc-server{ext}"


def test_llama_version_unknown(tmp_path):
    assert procs.llama_version(tmp_path) == "unknown"


# ---------- gpu ----------

def test_fake_devices_margin_and_budget(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("GPUPOOL_FAKE_DEVICES", json.dumps([
        {"device_id": "CUDA0", "kind": "cuda", "name": "a", "total_mb": 10000, "free_mb": 9000},
        {"device_id": "CUDA1", "kind": "cuda", "name": "b", "total_mb": 2000, "free_mb": 600},
        {"device_id": "CUDA2", "kind": "cuda", "name": "c", "total_mb": 10000, "free_mb": 9000},
        {"device_id": "CPU", "kind": "cpu", "name": "cpu", "total_mb": 16000, "free_mb": 8000},
    ]))
    cfg = make_cfg(tmp_path, budget_mb={"CUDA2": 1200, "CPU": 2000}, include_cpu=True)
    d = {x.device_id: x for x in probe_devices(cfg)}
    assert d["CUDA0"].usable_mb == 9000 - 1000  # 10% beats the 512 floor
    assert d["CUDA1"].usable_mb == 88  # floor 512 beats 10% of 2000
    assert d["CUDA2"].usable_mb == 1200  # budget caps
    assert d["CPU"].usable_mb == 2000  # 8000-512 capped by budget


def test_fake_devices_never_negative(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUPOOL_FAKE_DEVICES",
                       '[{"device_id":"CUDA0","kind":"cuda","name":"a","total_mb":1000,"free_mb":100}]')
    assert probe_devices(make_cfg(tmp_path))[0].usable_mb == 0


def test_probe_without_nvml_returns_cpu_only(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUPOOL_FAKE_DEVICES", raising=False)
    import gpupool.agent.gpu as g

    def boom(cfg):
        raise RuntimeError("no driver")
    monkeypatch.setattr(g, "_cuda_devices", boom)
    devs = probe_devices(make_cfg(tmp_path, include_cpu=True))
    assert [x.device_id for x in devs] == ["CPU"]
    assert probe_devices(make_cfg(tmp_path)) == []


# ---------- ProcessManager with a stand-in process ----------

LISTENER = """
import socket, sys, time
print("hello", flush=True)
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", {port})); s.listen()
for i in range(100): print("line", i, flush=True)
time.sleep(60)
"""


def wait_state(pm, eid, state, timeout=10):
    end = time.time() + timeout
    st = pm.get(eid)
    while time.time() < end:
        st = pm.get(eid)
        if st.state == state:
            return st
        time.sleep(0.05)
    raise AssertionError(f"state {st.state}, wanted {state}: {st.log_tail}")


@pytest.fixture
def pm(tmp_path):
    m = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1")
    yield m
    m.stop_all()


def patch_cmd(monkeypatch, script):
    monkeypatch.setattr(procs, "build_command",
                        lambda spec, bins, host, mp: [sys.executable, "-c", script])
    monkeypatch.setattr(ProcessManager, "_binaries", lambda self: BINS)


def test_lifecycle(pm, monkeypatch):
    port = free_port()
    patch_cmd(monkeypatch, LISTENER.format(port=port))
    spec = EngineSpec(engine_id="e1", kind="rpc", port=port, devices=["CPU"])
    st = pm.start(spec)
    assert st.state in ("starting", "running") and st.pid
    st = wait_state(pm, "e1", "running")
    assert len(st.log_tail) == 50 and st.log_tail[-1] == "line 99"
    with pytest.raises(EngineExists):
        pm.start(spec)
    with pytest.raises(PortInUse):  # another engine id, same (now busy) port
        pm.start(EngineSpec(engine_id="e2", kind="rpc", port=port, devices=["CPU"]))
    st = pm.stop("e1")
    assert st.state == "exited" and st.exit_code is not None
    assert pm.list()[0].engine_id == "e1"  # dead engines stay visible
    assert pm.stop("nope") is None and pm.get("nope") is None


def test_failed_exit_code_and_replace(pm, monkeypatch):
    patch_cmd(monkeypatch, "import sys; print('boom'); sys.exit(3)")
    spec = EngineSpec(engine_id="e1", kind="rpc", port=free_port(), devices=["CPU"])
    pm.start(spec)
    st = wait_state(pm, "e1", "failed")
    assert st.exit_code == 3 and st.log_tail == ["boom"]
    patch_cmd(monkeypatch, "print('ok')")
    pm.start(spec)  # replaces the dead entry
    st = wait_state(pm, "e1", "exited")
    assert st.exit_code == 0 and st.log_tail == ["ok"]


def test_kill_when_terminate_ignored(pm, monkeypatch):
    port = free_port()
    script = ("import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
              + LISTENER.format(port=port))
    patch_cmd(monkeypatch, script)
    pm.start(EngineSpec(engine_id="e1", kind="rpc", port=port, devices=["CPU"]))
    wait_state(pm, "e1", "running")
    st = pm.stop("e1", timeout=1.0)
    assert st.state == "exited"


# ---------- models_cache ----------

class _Handler(BaseHTTPRequestHandler):
    hits: list[str] = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Handler.hits.append(self.path)
        if "bad" in self.path:
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"x" * 10)
            self.wfile.flush()
            self.close_connection = True
            return
        if self.path.startswith("/files/") and self.headers.get("Authorization") != "Bearer tok":
            self.send_response(401)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b"GGUF" + b"0" * 4096
        time.sleep(0.2)
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def http_server():
    _Handler.hits = []
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ensure_local_path_untouched(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"abc")
    p, n = ensure_model("m", str(f), tmp_path / "cache", "http://x", "t")
    assert p == f and n == 3 and not (tmp_path / "cache").exists()


def test_ensure_http_download_and_cache(tmp_path, http_server):
    cache = tmp_path / "cache"
    p, n = ensure_model("m", f"{http_server}/dir/model.gguf", cache, "x", "t")
    assert p == cache / "model.gguf" and n == 4100 and p.read_bytes()[:4] == b"GGUF"
    assert not list(cache.glob("*.part"))
    ensure_model("m", f"{http_server}/dir/model.gguf", cache, "x", "t")
    assert len(_Handler.hits) == 1  # cache hit
    assert list_models(cache) == ["model.gguf"]


def test_ensure_truncated_leaves_nothing(tmp_path, http_server):
    cache = tmp_path / "cache"
    with pytest.raises(Exception):
        ensure_model("m", f"{http_server}/bad.gguf", cache, "x", "t")
    assert list(cache.iterdir()) == []


def test_ensure_coordinator_scheme_and_auth(tmp_path, http_server):
    p, n = ensure_model("m", "coordinator://c.gguf", tmp_path / "cache", http_server, "tok")
    assert p.name == "c.gguf" and n == 4100
    with pytest.raises(httpx.HTTPStatusError):
        ensure_model("m", "coordinator://d.gguf", tmp_path / "cache", http_server, "wrong")
    assert list_models(tmp_path / "cache") == ["c.gguf"]


def test_ensure_concurrent_downloads_once(tmp_path, http_server):
    res = []
    ts = [threading.Thread(target=lambda: res.append(
        ensure_model("m", f"{http_server}/z.gguf", tmp_path / "cache", "x", "t"))) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(res) == 4 and len(_Handler.hits) == 1


def test_ensure_missing_local(tmp_path):
    with pytest.raises(FileNotFoundError):
        ensure_model("m", str(tmp_path / "nope.gguf"), tmp_path / "c", "x", "t")


def test_new_manager_reaps_orphans_of_a_crashed_agent(tmp_path, monkeypatch):
    # Agent 1 starts an engine and dies without stopping it (no stop_all, like kill -9).
    port = free_port()
    patch_cmd(monkeypatch, LISTENER.format(port=port))
    crashed = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1")
    pid = crashed.start(EngineSpec(engine_id="orphan", kind="rpc", port=port, devices=["CPU"])).pid
    wait_state(crashed, "orphan", "running")
    import psutil
    assert psutil.pid_exists(pid)
    # Agent 2 on the same log_dir stops it on construction and frees the port.
    fresh = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1")
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert not (tmp_path / "logs" / "orphan.pid").exists()
    assert fresh._port_free(port)


def test_stale_pid_file_of_a_reused_pid_is_ignored(tmp_path):
    # A pid file whose pid now belongs to an unrelated process (different create_time)
    # must not kill that process.
    import os
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "old.pid").write_text(f"{os.getpid()} 12345.0\n")
    ProcessManager(tmp_path, logs, "127.0.0.1")
    assert not (logs / "old.pid").exists()  # we are still alive to assert this
