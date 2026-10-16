"""vLLM / SGLang on the agent: command lines, discovery, process groups, /engines validation."""
from __future__ import annotations

import json
import os
import socket
import sys
import time

import httpx
import psutil
import pytest

from gpupool.agent import engines, procs
from gpupool.agent.app import create_app
from gpupool.agent.engines import available_engines, build_hf_command, cuda_visible_devices, hf_env
from gpupool.agent.procs import ProcessManager
from gpupool.common.config import AgentConfig
from gpupool.common.models import EngineSpec

H = {"Authorization": "Bearer secret"}


def hf_spec(**kw) -> EngineSpec:
    base = dict(engine_id="r1-head", kind="server", port=9100, engine="vllm", devices=["CUDA0", "CUDA1"],
                model="qwen-vl", hf_repo="Qwen/Qwen2.5-VL-7B-Instruct", ctx_size=32768, parallel=8,
                gpu_mem_fraction=0.8123)
    return EngineSpec(**{**base, **kw})


def test_vllm_command():
    cmd = build_hf_command(hf_spec(hf_revision="abc123", trust_remote_code=True,
                                   engine_args=["--kv-cache-dtype", "fp8"]), ["vllm", "serve"], "10.0.0.1")
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-VL-7B-Instruct"]
    flags = dict(zip(cmd[3::2], cmd[4::2]))
    assert flags["--host"] == "10.0.0.1" and flags["--port"] == "9100"
    assert flags["--served-model-name"] == "qwen-vl" and flags["--tensor-parallel-size"] == "2"
    assert flags["--max-model-len"] == "32768" and flags["--max-num-seqs"] == "8"
    assert "--gpu-memory-utilization" in cmd and cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.8123"
    assert cmd[cmd.index("--revision") + 1] == "abc123" and "--trust-remote-code" in cmd
    assert cmd[-2:] == ["--kv-cache-dtype", "fp8"]  # engine flags last: they win over ours


def test_sglang_command():
    cmd = build_hf_command(hf_spec(engine="sglang", devices=["CUDA3"]),
                           ["python3", "-m", "sglang.launch_server"], "0.0.0.0")
    assert cmd[:5] == ["python3", "-m", "sglang.launch_server", "--model-path", "Qwen/Qwen2.5-VL-7B-Instruct"]
    i = cmd.index
    assert cmd[i("--tp-size") + 1] == "1" and cmd[i("--context-length") + 1] == "32768"
    assert cmd[i("--max-running-requests") + 1] == "8" and cmd[i("--mem-fraction-static") + 1] == "0.8123"
    assert "--enable-metrics" in cmd and "--revision" not in cmd and "--trust-remote-code" not in cmd


def test_visible_devices_prefer_uuids_and_reject_cpu():
    assert cuda_visible_devices(hf_spec()) == "0,1"
    assert cuda_visible_devices(hf_spec(device_uuids=["GPU-a", "GPU-b"])) == "GPU-a,GPU-b"
    assert cuda_visible_devices(hf_spec(device_uuids=["GPU-a"])) == "0,1"  # partial: positions
    with pytest.raises(ValueError):
        cuda_visible_devices(hf_spec(devices=["CPU"]))


def test_hf_env_keeps_a_user_hf_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    assert hf_env(hf_spec(), tmp_path)["HF_HOME"] == str(tmp_path)
    monkeypatch.setenv("HF_HOME", "/elsewhere")
    assert "HF_HOME" not in hf_env(hf_spec(), tmp_path)


def test_available_engines(tmp_path):
    exe = tmp_path / "vllm"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    found = available_engines({"vllm": f"{exe} serve", "sglang": f"{sys.executable} -m no_such_pkg_xyz.launch"})
    assert found == ["vllm"]
    # a python -m command counts once the interpreter has the package (json stands in for sglang)
    assert available_engines({"vllm": str(tmp_path / "missing"), "sglang": f"{sys.executable} -m json.tool"}) == ["sglang"]


# A stand-in for `vllm serve`: records its argv and environment, forks a worker like vLLM's
# tensor-parallel workers, then listens on --port.
FAKE_ENGINE = r'''
import json, os, socket, subprocess, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
print(json.dumps({"argv": sys.argv[1:], "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "hf_home": os.environ.get("HF_HOME"), "child": child.pid}), flush=True)
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", port)); s.listen()
time.sleep(120)
'''


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")
def test_process_manager_runs_and_stops_the_whole_group(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    script = tmp_path / "fake_vllm.py"
    script.write_text(FAKE_ENGINE)
    pm = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1", engine_commands={"vllm": f"{sys.executable} {script}"},
                        hf_home=tmp_path / "hf")
    spec = hf_spec(port=free_port(), device_uuids=["GPU-a", "GPU-b"])
    pm.start(spec)
    deadline = time.time() + 15
    while pm.get(spec.engine_id).state != "running" and time.time() < deadline:
        time.sleep(0.05)
    st = pm.get(spec.engine_id)
    assert st.state == "running", st.log_tail
    info = json.loads(next(ln for ln in st.log_tail if ln.startswith("{")))
    assert info["argv"][0] == "Qwen/Qwen2.5-VL-7B-Instruct" and info["cvd"] == "GPU-a,GPU-b"
    assert info["hf_home"] == str(tmp_path / "hf")
    pm.stop(spec.engine_id, timeout=2)
    deadline = time.time() + 5
    while psutil.pid_exists(info["child"]) and time.time() < deadline:
        try:
            if psutil.Process(info["child"]).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.05)
    alive = psutil.pid_exists(info["child"]) and psutil.Process(info["child"]).status() != psutil.STATUS_ZOMBIE
    assert not alive, "the worker outlived its engine"


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUPOOL_FAKE_DEVICES", json.dumps(
        [{"device_id": "CUDA0", "kind": "cuda", "name": "g", "total_mb": 24000, "free_mb": 23000}]))
    monkeypatch.setattr(procs, "llama_version", lambda d: "b1")
    import gpupool.agent.app as appmod
    monkeypatch.setattr(appmod, "llama_version", lambda d: "b1")
    exe = tmp_path / "vllm"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    return AgentConfig(node_id="n1", port=7071, host="127.0.0.1", llama_dir=tmp_path, cluster_token="secret",
                       cache_dir=tmp_path / "cache", log_dir=tmp_path / "logs", coordinator_url="http://coord:8080",
                       vllm_cmd=f"{exe} serve", sglang_cmd=str(tmp_path / "no-python"))


async def test_agent_reports_and_validates_engines(cfg, monkeypatch):
    seen = []
    monkeypatch.setattr(ProcessManager, "start", lambda self, spec, mp=None: seen.append(spec) or
                        procs.EngineStatus(engine_id=spec.engine_id, kind=spec.kind, state="starting", port=spec.port))
    app = create_app(cfg, start_heartbeat=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert "engine:vllm" in (await c.get("/report", headers=H)).json()["features"]
        body = hf_spec(devices=["CUDA0"]).model_dump(mode="json")
        assert (await c.post("/engines", json=body, headers=H)).status_code == 200
        r = await c.post("/engines", json={**body, "engine": "sglang"}, headers=H)
        assert r.status_code == 422 and "not installed" in r.text
        r = await c.post("/engines", json={**body, "engine_args": ["--foo"]}, headers=H)
        assert r.status_code == 422 and "GPUPOOL_ALLOW_ENGINE_ARGS" in r.text
        r = await c.post("/engines", json={**body, "hf_repo": None}, headers=H)
        assert r.status_code == 422
        llama = {"engine_id": "x", "kind": "rpc", "port": 9200, "devices": ["CUDA0"], "engine_args": ["--foo"]}
        assert (await c.post("/engines", json=llama, headers=H)).status_code == 422
    assert [s.engine for s in seen] == ["vllm"]


async def test_agent_accepts_engine_args_when_allowed(cfg, monkeypatch):
    seen = []
    monkeypatch.setattr(ProcessManager, "start", lambda self, spec, mp=None: seen.append(spec) or
                        procs.EngineStatus(engine_id=spec.engine_id, kind=spec.kind, state="starting", port=spec.port))
    app = create_app(cfg.model_copy(update={"allow_engine_args": True}), start_heartbeat=False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        body = hf_spec(devices=["CUDA0"], engine_args=["--enforce-eager"]).model_dump(mode="json")
        assert (await c.post("/engines", json=body, headers=H)).status_code == 200
    assert seen[0].engine_args == ["--enforce-eager"]


def test_engine_feature_name():
    assert engines.engine_feature("vllm") == "engine:vllm"
