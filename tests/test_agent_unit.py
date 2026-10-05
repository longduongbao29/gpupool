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


def test_build_rpc_serves_several_devices_from_one_process():
    spec = EngineSpec(engine_id="r-rpc-CUDA0", kind="rpc", port=9001, devices=["CUDA0", "CUDA1"])
    cmd = build_command(spec, BINS, "10.0.0.5", None)
    assert cmd[cmd.index("-d") + 1] == "CUDA0,CUDA1"


def test_build_server_single():
    spec = EngineSpec(engine_id="r-head", kind="server", port=9000, devices=["CUDA0"],
                      model="m", ctx_size=2048, parallel=2, extra_args=["--foo"])
    cmd = build_command(spec, BINS, "127.0.0.1", "/m.gguf")
    assert cmd == ["llama-server", "-m", "/m.gguf", "--host", "127.0.0.1", "--port", "9000",
                   "--alias", "m", "-c", "2048", "-np", "2", "-ngl", "999", "-lv", "4", "--device", "CUDA0",
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


_BASE = ["llama-server", "-m", "/m.gguf", "--host", "h", "--port", "1", "--alias", "m",
         "-c", "4096", "-np", "1", "-ngl", "999", "-lv", "4", "--device", "CUDA0", "--split-mode", "layer",
         "--cache-reuse", "256", "--metrics", "--fit", "off"]


def _srv(**kw):
    return EngineSpec(engine_id="x", kind="server", port=1, devices=["CUDA0"], model="m", **kw)


def test_build_default_has_no_kv_or_spec_flags():
    assert build_command(_srv(), BINS, "h", "/m.gguf") == _BASE


def test_build_cache_type():
    cmd = build_command(_srv(cache_type="q8_0", extra_args=["--foo"]), BINS, "h", "/m.gguf")
    assert cmd == _BASE + ["-ctk", "q8_0", "-ctv", "q8_0", "--foo"]


def test_build_ngram():
    cmd = build_command(_srv(spec_type="ngram"), BINS, "h", "/m.gguf")
    assert cmd == _BASE + ["--spec-type", "ngram-mod"]


def test_build_kv_unified():
    assert build_command(_srv(kv_unified=True), BINS, "h", "/m.gguf")[-1] == "-kvu"
    assert "-kvu" not in build_command(_srv(), BINS, "h", "/m.gguf")


@pytest.mark.parametrize("build, on", [(None, False), (11342, False), (11413, True), (12000, True)])
@pytest.mark.parametrize("spec_type", ["draft", "mtp"])
def test_probabilistic_draft_sampling_only_where_llama_has_it(spec_type, build, on):
    kw = {"draft_model_path": "/d.gguf", "draft_device": "CUDA0"} if spec_type == "draft" else {}
    cmd = build_command(_srv(spec_type=spec_type, **kw), BINS, "h", "/m.gguf", build)
    assert (cmd[-2:] == ["--spec-draft-sampling", "probabilistic"]) is on


def test_no_probabilistic_sampling_for_ngram():
    assert "--spec-draft-sampling" not in build_command(_srv(spec_type="ngram"), BINS, "h", "/m.gguf", 11413)


def test_build_mtp():
    cmd = build_command(_srv(spec_type="mtp", draft_n_max=3), BINS, "h", "/m.gguf")
    assert cmd == _BASE + ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]


def test_build_draft():
    cmd = build_command(_srv(spec_type="draft", draft_model_path="/d.gguf",
                             draft_device="CUDA0", draft_n_max=6), BINS, "h", "/m.gguf")
    assert cmd == _BASE + ["--spec-type", "draft-simple", "-md", "/d.gguf", "-devd", "CUDA0",
                           "-ngld", "999", "--spec-draft-n-max", "6"]


def test_build_draft_with_quantized_cache():
    cmd = build_command(_srv(spec_type="draft", draft_model_path="/d.gguf", draft_device="CUDA0",
                             cache_type="q4_0", extra_args=["--foo"]), BINS, "h", "/m.gguf")
    assert cmd == _BASE + ["-ctk", "q4_0", "-ctv", "q4_0", "--spec-type", "draft-simple",
                           "-md", "/d.gguf", "-devd", "CUDA0", "-ngld", "999",
                           "--spec-draft-n-max", "4", "-ctkd", "q4_0", "-ctvd", "q4_0", "--foo"]


@pytest.mark.parametrize("kw", [{"draft_device": "CUDA0"}, {"draft_model_path": "/d.gguf"}, {}])
def test_build_draft_requires_model_and_device(kw):
    with pytest.raises(ValueError):
        build_command(_srv(spec_type="draft", **kw), BINS, "h", "/m.gguf")


def test_build_rpc_ignores_kv_and_spec():
    spec = EngineSpec(engine_id="r", kind="rpc", port=9001, devices=["CUDA0"], cache_type="q8_0")
    assert "-ctk" not in build_command(spec, BINS, "h", None)


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
    # The cap itself is reported so the coordinator can subtract its own replicas from it.
    assert (d["CUDA0"].budget_mb, d["CUDA2"].budget_mb, d["CPU"].budget_mb) == (None, 1200, 2000)


def test_cpu_and_cuda_probes_report_budget(tmp_path, monkeypatch):
    import gpupool.agent.gpu as g
    cfg = make_cfg(tmp_path, budget_mb={"CPU": 3000}, include_cpu=True)
    monkeypatch.delenv("GPUPOOL_FAKE_DEVICES", raising=False)
    monkeypatch.setattr(g, "_cuda_devices", lambda c: [])
    assert probe_devices(cfg)[0].budget_mb == 3000
    assert probe_devices(make_cfg(tmp_path, include_cpu=True))[0].budget_mb is None


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
                        lambda spec, bins, host, mp, *_: [sys.executable, "-c", script])
    monkeypatch.setattr(ProcessManager, "_binaries", lambda self: BINS)


def test_lifecycle(pm, monkeypatch):
    port = free_port()
    patch_cmd(monkeypatch, LISTENER.format(port=port))
    spec = EngineSpec(engine_id="e1", kind="rpc", port=port, devices=["CPU"])
    st = pm.start(spec)
    assert st.state in ("starting", "running") and st.pid
    st = wait_state(pm, "e1", "running")
    assert len(st.log_tail) == 50 and st.log_tail[-1] == "line 99"  # get() keeps the tail
    assert [s.log_tail for s in pm.list() if s.state == "running"] == [[]]  # list() drops it
    with pytest.raises(EngineExists):
        pm.start(spec)
    with pytest.raises(PortInUse):  # another engine id, same (now busy) port
        pm.start(EngineSpec(engine_id="e2", kind="rpc", port=port, devices=["CPU"]))
    st = pm.stop("e1")
    assert st.state == "exited" and st.exit_code is not None
    assert pm.list() == [] and pm.get("e1") is None  # a stopped engine is forgotten
    assert (pm.log_dir / "e1.log").is_file()  # ...but its log stays on disk
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


def test_list_keeps_tail_of_failed_engine(pm, monkeypatch):
    patch_cmd(monkeypatch, "import sys; print('boom'); sys.exit(3)")
    pm.start(EngineSpec(engine_id="e1", kind="rpc", port=free_port(), devices=["CPU"]))
    wait_state(pm, "e1", "failed")
    (st,) = pm.list()
    assert st.state == "failed" and st.log_tail == ["boom"]  # crash events read this


def test_list_keeps_tail_of_starting_engine(pm, monkeypatch):
    port = free_port()
    patch_cmd(monkeypatch, "import time; print('loading', flush=True); time.sleep(60)")
    pm.start(EngineSpec(engine_id="e1", kind="rpc", port=port, devices=["CPU"]))
    end = time.time() + 10
    while time.time() < end and pm.list()[0].log_tail != ["loading"]:
        time.sleep(0.05)
    (st,) = pm.list()
    assert st.state == "starting" and st.log_tail == ["loading"]  # launch diagnostics


def test_crashed_engine_stays_listed_until_stopped(pm, monkeypatch):
    # The coordinator detects a crash by seeing exited/failed in the report, then calls stop.
    patch_cmd(monkeypatch, "import sys; sys.exit(3)")
    pm.start(EngineSpec(engine_id="e1", kind="rpc", port=free_port(), devices=["CPU"]))
    wait_state(pm, "e1", "failed")
    assert [s.engine_id for s in pm.list()] == ["e1"]
    assert pm.stop("e1").state == "failed"
    assert pm.list() == [] and pm.get("e1") is None


def test_stop_does_not_forget_a_replacement(pm, monkeypatch):
    patch_cmd(monkeypatch, "import sys; sys.exit(0)")
    pm.start(EngineSpec(engine_id="e1", kind="rpc", port=free_port(), devices=["CPU"]))
    old = pm._engines["e1"]
    wait_state(pm, "e1", "exited")
    new_spec = EngineSpec(engine_id="e1", kind="rpc", port=free_port(), devices=["CPU"])
    orig_status = pm._status

    def status_then_replace(eng):
        st = orig_status(eng)
        if eng is old:  # a concurrent start() lands between the status and the removal
            pm.start(new_spec)
        return st
    monkeypatch.setattr(pm, "_status", status_then_replace)
    pm.stop("e1")
    assert pm._engines["e1"] is not old


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
    # download + one size probe on the cache hit
    assert _Handler.hits == ["/dir/model.gguf"] * 2
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
    # one download; the three waiters find the file and only probe its size
    assert len(res) == 4 and len(_Handler.hits) == 4
    assert {r[1] for r in res} == {4100}


def _split_names(stem, total):
    return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


def test_ensure_split_downloads_all_parts(tmp_path, http_server):
    cache = tmp_path / "cache"
    names = _split_names("model", 3)
    p, n = ensure_model("m", f"coordinator://{names[0]}", cache, http_server, "tok")
    assert p == cache / names[0]
    assert n == 3 * 4100
    assert all((cache / x).stat().st_size == 4100 for x in names)
    assert sorted(_Handler.hits) == sorted(f"/files/{x}" for x in names)
    assert _Handler.hits[-1] == f"/files/{names[0]}"  # part 1 last
    _Handler.hits.clear()
    p, n = ensure_model("m", f"coordinator://{names[0]}", cache, http_server, "tok")
    assert (p, n) == (cache / names[0], 3 * 4100)
    assert sorted(_Handler.hits) == sorted(f"/files/{x}" for x in names)  # probes only


def test_ensure_split_fetches_only_missing_part(tmp_path, http_server):
    cache = tmp_path / "cache"
    names = _split_names("model", 3)
    ensure_model("m", f"coordinator://{names[0]}", cache, http_server, "tok")
    (cache / names[1]).unlink()
    _Handler.hits.clear()
    p, n = ensure_model("m", f"coordinator://{names[0]}", cache, http_server, "tok")
    assert n == 3 * 4100
    # part 2 downloaded; parts 3 and 1 only probed (part 1 last)
    assert _Handler.hits == [f"/files/{names[1]}", f"/files/{names[2]}", f"/files/{names[0]}"]
    assert (cache / names[1]).read_bytes()[:4] == b"GGUF"


def test_ensure_split_http_source_swaps_last_segment(tmp_path, http_server):
    names = _split_names("model", 2)
    p, n = ensure_model("m", f"{http_server}/dir/{names[0]}", tmp_path / "cache", "x", "t")
    assert n == 2 * 4100
    assert sorted(_Handler.hits) == sorted(f"/dir/{x}" for x in names)


def test_ensure_split_failed_part_leaves_no_part_files(tmp_path, http_server):
    cache = tmp_path / "cache"
    names = _split_names("bad", 2)  # the stub truncates every path containing "bad"
    with pytest.raises(Exception):
        ensure_model("m", f"coordinator://{names[0]}", cache, http_server, "tok")
    assert list(cache.glob("*.part")) == []
    assert not (cache / names[0]).exists()  # part 1 never looks complete


def test_ensure_non_first_part_is_a_plain_file(tmp_path, http_server):
    name = _split_names("model", 3)[1]
    p, n = ensure_model("m", f"coordinator://{name}", tmp_path / "cache", http_server, "tok")
    assert n == 4100 and _Handler.hits == [f"/files/{name}"]


def test_ensure_same_size_cache_is_not_redownloaded(tmp_path, http_server):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.gguf").write_bytes(b"S" * 4100)  # same size as the server's file
    p, n = ensure_model("m", f"{http_server}/model.gguf", cache, "x", "t")
    assert n == 4100 and p.read_bytes() == b"S" * 4100  # sentinel survived: body not fetched


def test_ensure_different_size_cache_is_replaced(tmp_path, http_server):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.gguf").write_bytes(b"old quant")
    p, n = ensure_model("m", f"{http_server}/model.gguf", cache, "x", "t")
    assert n == 4100 and p.read_bytes()[:4] == b"GGUF"
    assert not list(cache.glob("*.part"))


def test_ensure_stale_cache_in_use_fails_instead_of_serving_it(tmp_path, http_server, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "model.gguf").write_bytes(b"old quant")
    real_unlink = Path.unlink

    def locked(self, *a, **kw):
        if self.name == "model.gguf":
            raise PermissionError("in use")
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(OSError, match="stale"):
        ensure_model("m", f"{http_server}/model.gguf", cache, "x", "t")


def test_ensure_split_stale_part_is_replaced(tmp_path, http_server):
    cache = tmp_path / "cache"
    cache.mkdir()
    names = _split_names("model", 2)
    (cache / names[1]).write_bytes(b"x" * 10)  # stale part 2 from another file
    p, n = ensure_model("m", f"coordinator://{names[0]}", cache, http_server, "tok")
    assert n == 2 * 4100 and (cache / names[1]).stat().st_size == 4100


def test_ensure_probe_failure_keeps_cached_file(tmp_path, http_server):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "c.gguf").write_bytes(b"cached")
    # wrong token -> 401 on the probe; the node must still start with what it has
    p, n = ensure_model("m", "coordinator://c.gguf", cache, http_server, "wrong")
    assert n == 6 and p.read_bytes() == b"cached"
    # unreachable source: same
    (cache / "d.gguf").write_bytes(b"cached")
    p, n = ensure_model("m", "http://127.0.0.1:1/d.gguf", cache, "x", "t")
    assert n == 6


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


# ---------- NVML telemetry mapping (stubbed pynvml) ----------

class _NvmlErr(Exception):
    pass


@pytest.fixture(autouse=True)
def _reset_nvml_state():
    from gpupool.agent import gpu
    gpu._nvml_mod = None
    yield
    gpu._nvml_mod = None


def _stub_nvml(monkeypatch, fail=(), procs_compute=(), procs_graphics=()):
    import types
    from types import SimpleNamespace as NS

    def guard(name, value):
        def f(*a, **k):
            if name in fail:
                raise _NvmlErr("NOT_SUPPORTED")
            return value
        return f

    m = types.ModuleType("pynvml")
    m.NVML_TEMPERATURE_GPU = 0
    m.NVML_VALUE_NOT_AVAILABLE = 2**64 - 1
    m.nvmlInit = lambda: None
    m.nvmlShutdown = lambda: None
    m.nvmlDeviceGetCount = lambda: 1
    m.nvmlDeviceGetHandleByIndex = lambda i: "h"
    m.nvmlDeviceGetPciInfo = lambda h: NS(busId="0000:01:00.0")
    m.nvmlDeviceGetName = lambda h: b"GPU X"
    m.nvmlDeviceGetMemoryInfo = lambda h: NS(total=4000 * 1024 * 1024, free=3000 * 1024 * 1024)
    m.nvmlDeviceGetUtilizationRates = lambda h: NS(gpu=7)
    m.nvmlDeviceGetTemperature = guard("temp", 61)
    m.nvmlDeviceGetPowerUsage = guard("power", 25400)
    m.nvmlSystemGetDriverVersion = guard("driver", b"535.154.05")
    m.nvmlSystemGetCudaDriverVersion_v2 = guard("cuda", 12020)
    m.nvmlDeviceGetComputeRunningProcesses = guard("compute", list(procs_compute))
    m.nvmlDeviceGetGraphicsRunningProcesses = guard("graphics", list(procs_graphics))
    monkeypatch.setitem(sys.modules, "pynvml", m)
    monkeypatch.delenv("GPUPOOL_FAKE_DEVICES", raising=False)
    return m


def test_nvml_fields_mapped(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import gpupool.agent.gpu as g
    na = 2**64 - 1
    monkeypatch.setattr(g.psutil, "Process", lambda pid: NS(name=lambda: f"p{pid}"))
    _stub_nvml(monkeypatch,
               procs_compute=[NS(pid=1, usedGpuMemory=512 * 1024 * 1024), NS(pid=2, usedGpuMemory=na)],
               procs_graphics=[NS(pid=1, usedGpuMemory=1), NS(pid=3, usedGpuMemory=None)])
    d = g._cuda_devices(make_cfg(tmp_path))[0]
    assert (d.temp_c, d.power_w, d.driver, d.cuda, d.util_pct) == (61, 25, "535.154.05", "12.2", 7)
    assert [(p.pid, p.name, p.used_mb) for p in d.processes] == [
        (1, "p1", 512), (2, "p2", None), (3, "p3", None)]


@pytest.mark.parametrize("fail", ["temp", "power", "driver", "cuda", "compute", "graphics"])
def test_nvml_each_call_best_effort(tmp_path, monkeypatch, fail):
    import gpupool.agent.gpu as g
    _stub_nvml(monkeypatch, fail=(fail,))
    d = g._cuda_devices(make_cfg(tmp_path))
    assert len(d) == 1 and d[0].total_mb == 4000
    field = {"temp": "temp_c", "power": "power_w", "driver": "driver", "cuda": "cuda"}.get(fail)
    if field:
        assert getattr(d[0], field) is None
    assert d[0].processes == []


def test_nvml_hidden_process_name_blank(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import gpupool.agent.gpu as g

    def boom(pid):
        raise PermissionError("denied")
    monkeypatch.setattr(g.psutil, "Process", boom)
    _stub_nvml(monkeypatch, procs_compute=[NS(pid=9, usedGpuMemory=1024 * 1024)])
    p = g._cuda_devices(make_cfg(tmp_path))[0].processes
    assert [(x.pid, x.name, x.used_mb) for x in p] == [(9, "", 1)]


def test_fake_devices_passthrough(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("GPUPOOL_FAKE_DEVICES", json.dumps([{
        "device_id": "CUDA0", "total_mb": 100, "free_mb": 90, "temp_c": 50, "power_w": 70,
        "driver": "1.2", "cuda": "12.4", "processes": [{"pid": 5, "name": "x", "used_mb": 3}]}]))
    d = probe_devices(make_cfg(tmp_path))[0]
    assert (d.temp_c, d.power_w, d.driver, d.cuda) == (50, 70, "1.2", "12.4")
    assert d.processes[0].pid == 5


def test_lost_gpu_is_skipped_alone(tmp_path, monkeypatch):
    # Two GPUs; the second fell off the bus (every call raises GPU_IS_LOST). The healthy one
    # must still be reported, otherwise the coordinator fails replicas on it too.
    from types import SimpleNamespace as NS
    import gpupool.agent.gpu as g
    m = _stub_nvml(monkeypatch)
    m.nvmlDeviceGetCount = lambda: 2
    m.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"

    def pci(h):
        if h == "h1":
            raise _NvmlErr("GPU_IS_LOST")
        return NS(busId="0000:01:00.0")
    m.nvmlDeviceGetPciInfo = pci
    devs = g._cuda_devices(make_cfg(tmp_path))
    assert [d.device_id for d in devs] == ["CUDA0"]
    assert devs[0].usable_mb == 0  # numbering is no longer trustworthy, see below


def _three_gpus(monkeypatch):
    from types import SimpleNamespace as NS
    m = _stub_nvml(monkeypatch)
    m.nvmlDeviceGetCount = lambda: 3
    m.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"
    m.nvmlDeviceGetPciInfo = lambda h: NS(busId=f"0000:0{h[1]}:00.0")
    return m


def test_all_healthy_gpus_are_usable(tmp_path, monkeypatch):
    import gpupool.agent.gpu as g
    _three_gpus(monkeypatch)
    devs = g._cuda_devices(make_cfg(tmp_path))
    assert [d.device_id for d in devs] == ["CUDA0", "CUDA1", "CUDA2"]
    assert all(d.usable_mb > 0 for d in devs)


def test_lost_gpu_in_handle_query_blocks_node(tmp_path, monkeypatch, caplog):
    import gpupool.agent.gpu as g
    m = _three_gpus(monkeypatch)

    def handle(i):
        if i == 1:
            raise _NvmlErr("GPU_IS_LOST")
        return f"h{i}"
    m.nvmlDeviceGetHandleByIndex = handle
    with caplog.at_level("ERROR"):
        devs = g._cuda_devices(make_cfg(tmp_path))
    assert len(devs) == 2 and all(d.usable_mb == 0 for d in devs)
    assert sum(r.levelname == "ERROR" for r in caplog.records) == 1


def test_lost_gpu_in_pci_query_blocks_node(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import gpupool.agent.gpu as g
    m = _three_gpus(monkeypatch)

    def pci(h):
        if h == "h1":
            raise _NvmlErr("GPU_IS_LOST")
        return NS(busId=f"0000:0{h[1]}:00.0")
    m.nvmlDeviceGetPciInfo = pci
    devs = g._cuda_devices(make_cfg(tmp_path))
    assert len(devs) == 2 and all(d.usable_mb == 0 for d in devs)


def test_lost_gpu_in_memory_query_blocks_node(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    import gpupool.agent.gpu as g
    m = _three_gpus(monkeypatch)

    def mem(h):
        if h == "h1":
            raise _NvmlErr("GPU_IS_LOST")
        return NS(total=4000 * 1024 * 1024, free=3000 * 1024 * 1024)
    m.nvmlDeviceGetMemoryInfo = mem
    devs = g._cuda_devices(make_cfg(tmp_path))
    assert len(devs) == 2 and all(d.usable_mb == 0 for d in devs)


def _count_init(m):
    calls = {"init": 0, "shutdown": 0}

    def init():
        calls["init"] += 1

    def shutdown():
        calls["shutdown"] += 1
    m.nvmlInit, m.nvmlShutdown = init, shutdown
    return calls


def test_nvml_initialised_once_across_probes(tmp_path, monkeypatch):
    m = _stub_nvml(monkeypatch)
    calls = _count_init(m)
    cfg = make_cfg(tmp_path)
    for _ in range(3):
        assert [d.device_id for d in probe_devices(cfg) if d.kind == "cuda"] == ["CUDA0"]
    assert calls == {"init": 1, "shutdown": 0}


def test_nvml_init_failure_retried_next_probe(tmp_path, monkeypatch):
    m = _stub_nvml(monkeypatch)
    state = {"n": 0}

    def init():
        state["n"] += 1
        if state["n"] == 1:
            raise _NvmlErr("DRIVER_NOT_LOADED")
    m.nvmlInit = init
    cfg = make_cfg(tmp_path)
    assert [d for d in probe_devices(cfg) if d.kind == "cuda"] == []  # never raises
    assert [d.device_id for d in probe_devices(cfg) if d.kind == "cuda"] == ["CUDA0"]
    assert [d.device_id for d in probe_devices(cfg) if d.kind == "cuda"] == ["CUDA0"]
    assert state["n"] == 2  # re-init only after the failure


def test_nvml_uninitialized_error_triggers_reinit(tmp_path, monkeypatch):
    m = _stub_nvml(monkeypatch)
    calls = _count_init(m)

    class NVMLError_Uninitialized(Exception):
        pass
    count = {"fail": True}

    def get_count():
        if count["fail"]:
            count["fail"] = False
            raise NVMLError_Uninitialized("Uninitialized")
        return 1
    m.nvmlDeviceGetCount = get_count
    cfg = make_cfg(tmp_path)
    assert [d for d in probe_devices(cfg) if d.kind == "cuda"] == []
    assert [d.device_id for d in probe_devices(cfg) if d.kind == "cuda"] == ["CUDA0"]
    assert calls["init"] == 2


def test_nvml_lost_gpu_does_not_reinit(tmp_path, monkeypatch):
    # GPU_IS_LOST is a per-device fault, not a library fault.
    import gpupool.agent.gpu as g
    m = _stub_nvml(monkeypatch)
    calls = _count_init(m)

    def mem(h):
        raise _NvmlErr("GPU_IS_LOST")
    m.nvmlDeviceGetMemoryInfo = mem
    cfg = make_cfg(tmp_path)
    g._cuda_devices(cfg)
    g._cuda_devices(cfg)
    assert calls["init"] == 1


def test_nvml_uuid_and_pci_filled(tmp_path, monkeypatch):
    import gpupool.agent.gpu as g
    m = _stub_nvml(monkeypatch)
    m.nvmlDeviceGetUUID = lambda h: b"GPU-8f2c0000-aaaa"
    d = g._cuda_devices(make_cfg(tmp_path))[0]
    assert (d.uuid, d.pci_bus_id) == ("GPU-8f2c0000-aaaa", "0000:01:00.0")
    m.nvmlDeviceGetUUID = lambda h: "GPU-str"  # newer pynvml returns str
    assert g._cuda_devices(make_cfg(tmp_path))[0].uuid == "GPU-str"


def test_nvml_uuid_failure_keeps_device(tmp_path, monkeypatch):
    import gpupool.agent.gpu as g
    m = _stub_nvml(monkeypatch)

    def boom(h):
        raise _NvmlErr("NOT_SUPPORTED")
    m.nvmlDeviceGetUUID = boom
    d = g._cuda_devices(make_cfg(tmp_path))
    assert len(d) == 1 and d[0].uuid is None
    assert d[0].usable_mb > 0 and d[0].pci_bus_id == "0000:01:00.0"


def test_nvml_bandwidth_from_bus_width_and_mem_clock(tmp_path, monkeypatch):
    import gpupool.agent.gpu as g
    m = _stub_nvml(monkeypatch)
    m.NVML_CLOCK_MEM = 2
    m.nvmlDeviceGetMemoryBusWidth = lambda h: 128
    m.nvmlDeviceGetMaxClockInfo = lambda h, kind: 5001 if kind == 2 else 0
    assert g._cuda_devices(make_cfg(tmp_path))[0].bandwidth_gbps == 160.0


@pytest.mark.parametrize("fail", ["width", "clock", "missing"])
def test_nvml_bandwidth_failure_keeps_device(tmp_path, monkeypatch, fail):
    import gpupool.agent.gpu as g
    m = _stub_nvml(monkeypatch)
    m.NVML_CLOCK_MEM = 2

    def boom(*a):
        raise _NvmlErr("NOT_SUPPORTED")
    m.nvmlDeviceGetMemoryBusWidth = boom if fail == "width" else (lambda h: 128)
    m.nvmlDeviceGetMaxClockInfo = boom if fail == "clock" else (lambda h, kind: 5001)
    if fail == "missing":
        del m.nvmlDeviceGetMaxClockInfo
    d = g._cuda_devices(make_cfg(tmp_path))
    assert len(d) == 1 and d[0].bandwidth_gbps is None and d[0].usable_mb > 0


def test_fake_devices_bandwidth_passthrough(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("GPUPOOL_FAKE_DEVICES", json.dumps([
        {"device_id": "CUDA0", "total_mb": 100, "free_mb": 90, "bandwidth_gbps": 320.5},
        {"device_id": "CUDA1", "total_mb": 100, "free_mb": 90}]))
    d = probe_devices(make_cfg(tmp_path))
    assert (d[0].bandwidth_gbps, d[1].bandwidth_gbps) == (320.5, None)


def test_fake_devices_identity_passthrough(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("GPUPOOL_FAKE_DEVICES", json.dumps([{
        "device_id": "CUDA0", "total_mb": 100, "free_mb": 90,
        "uuid": "GPU-fake", "pci_bus_id": "00000000:02:00.0"}]))
    d = probe_devices(make_cfg(tmp_path))[0]
    assert (d.uuid, d.pci_bus_id) == ("GPU-fake", "00000000:02:00.0")


def test_disable_core_dumps_sets_rlimit_core_to_zero(monkeypatch):
    # Engines inherit the agent's limit: a llama.cpp abort must not write a core of 0.1-0.5 GB.
    import gpupool.agent.procs as procs_mod
    resource = pytest.importorskip("resource")  # POSIX only; a no-op elsewhere
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda what, lim: calls.append((what, lim)))
    procs_mod.disable_core_dumps()
    assert calls == [(resource.RLIMIT_CORE, (0, 0))]


def test_disable_core_dumps_is_a_no_op_without_resource(monkeypatch):
    import builtins
    import gpupool.agent.procs as procs_mod
    real_import = builtins.__import__

    def no_resource(name, *a, **kw):
        if name == "resource":
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_resource)
    procs_mod.disable_core_dumps()  # must not raise (Windows)


# ---------- attention / batching flags, CUDA architectures ----------

def test_build_flash_attn_and_batches():
    assert build_command(_srv(flash_attn="auto", batch=2048, ubatch=512), BINS, "h", "/m.gguf") == _BASE
    cmd = build_command(_srv(flash_attn="on", ubatch=1024), BINS, "h", "/m.gguf")
    assert cmd == _BASE + ["-fa", "on", "-ub", "1024"]
    # a micro-batch above the batch raises the batch (llama.cpp would cap the micro-batch instead)
    cmd = build_command(_srv(flash_attn="off", batch=1024, ubatch=4096), BINS, "h", "/m.gguf")
    assert cmd == _BASE + ["-fa", "off", "-b", "4096", "-ub", "4096"]


def test_llama_cuda_archs_file_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("GPUPOOL_CUDA_ARCHS", raising=False)
    assert procs.llama_cuda_archs(tmp_path) is None
    (tmp_path / "cuda-archs.txt").write_text("120 61 86 junk 70\n")
    assert procs.llama_cuda_archs(tmp_path) == ["61", "70", "86", "120"]
    monkeypatch.setenv("GPUPOOL_CUDA_ARCHS", "75;89")
    assert procs.llama_cuda_archs(tmp_path) == ["75", "89"]


def test_nvml_compute_cap_reported(tmp_path, monkeypatch):
    import gpupool.agent.gpu as g
    monkeypatch.delenv("GPUPOOL_CUDA_ARCHS", raising=False)
    m = _stub_nvml(monkeypatch)
    m.nvmlDeviceGetCudaComputeCapability = lambda h: (8, 6)
    d = g._cuda_devices(make_cfg(tmp_path))[0]
    assert d.compute_cap == "8.6" and d.kernels_ok is None and d.usable_mb > 0  # build archs unknown


@pytest.mark.parametrize("archs,cc,ok", [("61 86", (8, 9), True), ("86", (6, 1), False), ("61", (7, 5), True)])
def test_nvml_gpu_without_kernels_is_not_offered(tmp_path, monkeypatch, archs, cc, ok):
    import gpupool.agent.gpu as g
    monkeypatch.delenv("GPUPOOL_CUDA_ARCHS", raising=False)
    (tmp_path / "cuda-archs.txt").write_text(archs)
    m = _stub_nvml(monkeypatch)
    m.nvmlDeviceGetCudaComputeCapability = lambda h: cc
    d = g._cuda_devices(make_cfg(tmp_path))[0]
    assert d.kernels_ok is ok and (d.usable_mb > 0) is ok


def test_nvml_compute_cap_failure_keeps_device(tmp_path, monkeypatch):
    import gpupool.agent.gpu as g
    _stub_nvml(monkeypatch)  # no nvmlDeviceGetCudaComputeCapability (very old pynvml)
    d = g._cuda_devices(make_cfg(tmp_path))[0]
    assert d.compute_cap is None and d.kernels_ok is None and d.usable_mb > 0


ENV_PRINTER = """
import os
print("LLAMA_CACHE=" + os.environ.get("LLAMA_CACHE", "<unset>"), flush=True)
"""


@pytest.mark.parametrize("preset", [None, "/user/choice"])
def test_engines_get_a_persistent_llama_cache(tmp_path, monkeypatch, preset):
    # ggml-rpc-server -c caches weights under $LLAMA_CACHE/rpc: it must land in the agent's
    # data dir, not in the container's writable layer; a LLAMA_CACHE the user set wins.
    if preset:
        monkeypatch.setenv("LLAMA_CACHE", preset)
    else:
        monkeypatch.delenv("LLAMA_CACHE", raising=False)
    patch_cmd(monkeypatch, ENV_PRINTER)
    m = ProcessManager(tmp_path, tmp_path / "logs", "127.0.0.1", llama_cache=tmp_path / "lc")
    try:
        m.start(EngineSpec(engine_id="e1", kind="rpc", port=free_port(), devices=["CPU"]))
        st = wait_state(m, "e1", "exited")
    finally:
        m.stop_all()
    assert st.log_tail == [f"LLAMA_CACHE={preset or tmp_path / 'lc'}"]
