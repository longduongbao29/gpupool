"""ConvertManager end to end with fake tools (make_toolkit), a fake HF client and fake library.

inspect_source is monkeypatched (gpupool.converter.jobs.inspect_source): these tests are about
the job pipeline, not about reading config.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import types
from pathlib import Path

import psutil
import pytest

from gpupool.converter import jobs as jobs_mod
from gpupool.converter.jobs import ConvertManager
from gpupool.converter.models import (
    ConvertAdvanced,
    ConvertError,
    ConvertRequest,
    InspectResult,
    SourceFile,
    SourceSpec,
)
from gpupool.converter.quant import QUANT_OPTIONS
from tests.test_converter_toolchain import make_toolkit

REPO = "acme/tiny-model"
CONFIG = json.dumps({"architectures": ["LlamaForCausalLM"], "torch_dtype": "bfloat16"})
REPO_FILES = {
    "config.json": CONFIG.encode(),
    "model.safetensors": b"W" * 5000,
    "tokenizer.json": b"{}",
    "README.md": b"# readme",
    "modeling_tiny.py": b"raise SystemExit('remote code must never run')",
}


class FakeHf:
    """Stands in for HfClient: list_files + download with the real cache-hit contract."""

    def __init__(self, files: dict[str, bytes] | None = None):
        self.files = dict(files or REPO_FILES)
        self.network: list[str] = []  # files really "downloaded"
        self.slow = False

    async def list_files(self, repo, revision="main"):
        return [SourceFile(name=n, bytes=len(b)) for n, b in self.files.items()]

    async def download(self, repo, revision, file, dest, expected_bytes, on_progress):
        if expected_bytes is not None and dest.is_file() and dest.stat().st_size == expected_bytes:
            on_progress(expected_bytes)
            return expected_bytes
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.files[file]
        self.network.append(file)
        dest.write_bytes(data)
        on_progress(len(data))
        await asyncio.sleep(0)
        return len(data)


class FakeLibrary:
    def __init__(self):
        self.names: set[str] = set()
        self.added: list[tuple[str, Path, str | None]] = []

    def name_taken(self, name):
        return name in self.names

    def add_converted(self, name, path, hf_repo):
        assert path.is_file(), "the manager must move the file before registering it"
        self.names.add(name)
        self.added.append((name, path, hf_repo))
        return types.SimpleNamespace(name=name)

    def locate_dir(self, path):
        p = Path(path)
        if not p.is_dir():
            raise LibErr(f"no such folder: {path}", 404)
        return p

    def locate_file(self, path):
        p = Path(path)
        if not p.is_file():
            raise LibErr(f"no such file: {path}", 404)
        return p


class LibErr(Exception):
    """Like LibraryError: carries .message and .status."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def fake_inspect_factory(**over):
    calls = []

    async def fake(spec, *, hf, locate_dir, cluster, supported_architectures):
        calls.append(supported_architectures)
        opts = [o.model_copy(update={"est_bytes": 3000}) for o in QUANT_OPTIONS]
        data = dict(source=spec, architecture="LlamaForCausalLM", model_type="llama", supported=True,
                    params=2000, weight_format="safetensors", options=opts, source_bytes=5000)
        data.update(over)
        return InspectResult(**data)

    fake.calls = calls
    return fake


@pytest.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_mod, "inspect_source", fake_inspect_factory())
    tc = make_toolkit(tmp_path / "kit")
    lib = FakeLibrary()
    hf = FakeHf()
    models = tmp_path / "models"
    mgr = ConvertManager(tmp_path / "jobs.db", models, tc, lib, hf, threads=2)
    mgr.start()
    yield types.SimpleNamespace(mgr=mgr, tc=tc, lib=lib, hf=hf, models=models, tmp=tmp_path,
                                db=tmp_path / "jobs.db")
    await mgr.shutdown()


def hf_req(**kw) -> ConvertRequest:
    adv = kw.pop("advanced", {})
    return ConvertRequest(source=SourceSpec(hf_repo=REPO), advanced=ConvertAdvanced(**adv), **kw)


async def wait_for(mgr, job_id, states, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = mgr.get(job_id)
        if job.state in states:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job stuck in {mgr.get(job_id).state}: {mgr.get(job_id).error}")


async def run_job(env, req, states=("done", "failed", "needs_review", "cancelled")):
    job = await env.mgr.submit(req)
    return await wait_for(env.mgr, job.id, states)


def work_dirs(env):
    root = env.models / ".convert"
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


# ---- happy paths -------------------------------------------------------------------------------


async def test_hf_job_quantized_end_to_end(env, monkeypatch):
    seen, argv = env.tmp / "seen.jsonl", env.tmp / "argv.jsonl"
    monkeypatch.setenv("FAKE_SEEN_FILE", str(seen))
    monkeypatch.setenv("FAKE_ARGV_LOG", str(argv))
    req = hf_req(quant="Q4_K_M", advanced={"pure": True, "output_tensor_type": "q8_0", "threads": 3})
    job = await env.mgr.submit(req)
    assert job.state == "queued" and job.output_name == "tiny-model-Q4_K_M.gguf"
    assert job.est_output_bytes == 3000
    states = []
    while True:
        j = env.mgr.get(job.id)
        if not states or states[-1] != j.state:
            states.append(j.state)
        if j.state in ("done", "failed"):
            break
        await asyncio.sleep(0.01)
    assert j.state == "done", j.error
    assert states[0] in ("queued", "downloading") and states[-1] == "done"
    assert j.validation.header_ok and j.validation.tokenizer_ok is True
    assert j.validation.generation_ok is True
    assert j.output_bytes == (env.models / j.output_name).stat().st_size
    assert j.finished_at is not None and j.started_at is not None
    # library registration with the moved file
    assert env.lib.added == [(j.output_name, env.models / j.output_name, REPO)]
    # converter saw only whitelisted files, offline, bf16 intermediate (bfloat16 source)
    rec = json.loads(seen.read_text().splitlines()[0])
    assert rec["seen"] == ["config.json", "model.safetensors", "tokenizer.json"]
    assert rec["outtype"] == "bf16" and rec["offline"] == "1" and rec["no_local_gguf"] is None
    qargs = json.loads(argv.read_text().splitlines()[0])
    assert "--pure" in qargs and qargs[qargs.index("--output-tensor-type") + 1] == "q8_0"
    assert qargs[-2:] == ["Q4_K_M", "3"]
    # nothing left behind
    assert work_dirs(env) == [] and not (env.models / ".hf" / f"acme__tiny-model@main").exists()
    assert sorted(p.name for p in env.models.iterdir() if p.is_file()) == [j.output_name]
    # the toolchain's supported set reached inspect
    assert jobs_mod.inspect_source.calls[0] == {"LlamaForCausalLM", "Qwen2ForCausalLM"}


async def test_direct_q8_0_skips_quantize(env, monkeypatch):
    argv = env.tmp / "argv.jsonl"
    monkeypatch.setenv("FAKE_ARGV_LOG", str(argv))
    j = await run_job(env, hf_req(quant="Q8_0", name="direct.gguf"))
    assert j.state == "done", j.error
    assert not argv.exists()
    assert (env.models / "direct.gguf").is_file()


async def test_f16_intermediate_for_non_bf16_source(env, monkeypatch):
    env.hf.files["config.json"] = json.dumps({"torch_dtype": "float16"}).encode()
    seen = env.tmp / "seen.jsonl"
    monkeypatch.setenv("FAKE_SEEN_FILE", str(seen))
    j = await run_job(env, hf_req(quant="Q5_K_M"))
    assert j.state == "done", j.error
    assert json.loads(seen.read_text().splitlines()[0])["outtype"] == "f16"


async def test_job_survives_manager_restart_in_db(env):
    j = await run_job(env, hf_req(quant="Q8_0"))
    assert j.state == "done"
    await env.mgr.shutdown()
    mgr2 = ConvertManager(env.db, env.models, env.tc, env.lib, env.hf)
    j2 = mgr2.get(j.id)
    assert j2.state == "done" and j2.request == j.request and j2.validation == j.validation
    assert j2.output_name == j.output_name and [x.id for x in mgr2.list()] == [j.id]
    await mgr2.shutdown()


async def test_list_is_newest_first(env):
    a = await env.mgr.submit(hf_req(quant="Q8_0", name="a.gguf"))
    b = await env.mgr.submit(hf_req(quant="Q8_0", name="b.gguf"))
    assert [j.id for j in env.mgr.list()] == [b.id, a.id]
    await wait_for(env.mgr, b.id, ("done",))


# ---- submit policy -----------------------------------------------------------------------------


async def test_submit_503_without_toolchain(env):
    (env.tc.convert_dir / "convert_hf_to_gguf.py").unlink()
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req())
    assert ei.value.status == 503 and "convert_hf_to_gguf.py" in ei.value.message
    assert env.mgr.available() is not None


async def test_submit_unsupported_architecture_422(env, monkeypatch):
    monkeypatch.setattr(jobs_mod, "inspect_source",
                        fake_inspect_factory(supported=False, architecture="FooForCausalLM"))
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req())
    assert ei.value.status == 422
    assert ei.value.message == ("architecture FooForCausalLM is not supported by "
                                "llama.cpp b11342's converter")


async def test_submit_unsupported_prequant_suggests_base_model(env, monkeypatch):
    monkeypatch.setattr(jobs_mod, "inspect_source", fake_inspect_factory(
        prequantized="awq", prequant_supported=False, base_model="acme/base"))
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req())
    assert ei.value.status == 422 and "awq" in ei.value.message and "acme/base" in ei.value.message


async def test_submit_supported_prequant_is_allowed(env, monkeypatch):
    monkeypatch.setattr(jobs_mod, "inspect_source", fake_inspect_factory(
        prequantized="fp8", prequant_supported=True))
    j = await run_job(env, hf_req(quant="Q8_0"))
    assert j.state == "done", j.error


async def test_submit_no_weights_422(env, monkeypatch):
    monkeypatch.setattr(jobs_mod, "inspect_source", fake_inspect_factory(weight_format="none"))
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req())
    assert ei.value.status == 422


async def test_submit_bad_output_name_400(env):
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req(name="../evil.gguf"))
    assert ei.value.status == 400


async def test_name_collisions_409(env, monkeypatch):
    env.lib.names.add("taken.gguf")
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req(name="taken.gguf"))
    assert ei.value.status == 409
    # an untracked file in models_dir must not be overwritten either
    (env.models / "stray.gguf").write_bytes(b"x")
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req(name="stray.gguf"))
    assert ei.value.status == 409
    # another non-terminal job with the same output name
    monkeypatch.setenv("FAKE_CONVERT_MODE", "slow")
    first = await env.mgr.submit(hf_req(quant="Q8_0", name="same.gguf"))
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req(quant="Q8_0", name="same.gguf"))
    assert ei.value.status == 409 and first.id in ei.value.message
    await env.mgr.cancel(first.id)
    # terminal jobs release the name
    again = await env.mgr.submit(hf_req(quant="Q8_0", name="same.gguf"))
    await env.mgr.cancel(again.id)


# ---- failure modes -----------------------------------------------------------------------------


async def test_converter_failure_surfaces_tool_message_and_cleans_up(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_MODE", "fail")
    j = await run_job(env, hf_req())
    assert j.state == "failed"
    assert "Model Foo is not supported" in j.error and "exit code 1" in j.error
    assert work_dirs(env) == [] and not list(env.models.glob("*.gguf"))
    assert env.lib.added == []
    assert any("not supported" in ln for ln in j.log_tail)
    # the download cache survives a failure so a retry does not download again
    assert (env.models / ".hf" / "acme__tiny-model@main" / "model.safetensors").is_file()


async def test_retry_reuses_cache_and_succeeds(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_MODE", "fail")
    j = await run_job(env, hf_req(quant="Q8_0"))
    assert j.state == "failed"
    downloads = len(env.hf.network)
    monkeypatch.setenv("FAKE_CONVERT_MODE", "ok")
    r = await env.mgr.retry(j.id)
    assert r.state == "queued" and r.error is None
    done = await wait_for(env.mgr, j.id, ("done", "failed"))
    assert done.state == "done", done.error
    assert len(env.hf.network) == downloads  # all cache hits


async def test_retry_wrong_state_and_name_conflict(env, monkeypatch):
    j = await run_job(env, hf_req(quant="Q8_0", name="r.gguf"))
    with pytest.raises(ConvertError) as ei:
        await env.mgr.retry(j.id)
    assert ei.value.status == 409
    monkeypatch.setenv("FAKE_CONVERT_MODE", "fail")
    f = await run_job(env, hf_req(quant="Q8_0", name="f.gguf"))
    assert f.state == "failed"
    env.lib.names.add("f.gguf")
    with pytest.raises(ConvertError) as ei:
        await env.mgr.retry(f.id)
    assert ei.value.status == 409


async def test_quantizer_failure_message(env, monkeypatch):
    monkeypatch.setenv("FAKE_QUANT_MODE", "fail")
    j = await run_job(env, hf_req(quant="Q4_K_M"))
    assert j.state == "failed" and "llama-quantize failed" in j.error and "boom" in j.error
    assert work_dirs(env) == []


async def test_missing_tokenizer_in_output_is_failure(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_MODE", "notok")
    j = await run_job(env, hf_req(quant="Q8_0"))
    assert j.state == "failed" and "tokenizer" in j.error
    assert j.validation.header_ok is False
    assert work_dirs(env) == [] and not list(env.models.glob("*.gguf"))


async def test_disk_check_failure(env, monkeypatch):
    # the early check at submit is covered below; this is the worker's own per-stage check
    async def no_early_check(*a, **k):
        return None

    monkeypatch.setattr(env.mgr, "_early_disk_check", no_early_check)
    usage = types.SimpleNamespace(total=10**12, used=10**12 - 1_000_000, free=1_000_000)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage)
    j = await run_job(env, hf_req())
    assert j.state == "failed"
    assert str(env.models) in j.error and "GB" in j.error and "free" in j.error
    assert env.hf.network == []  # refused before downloading anything
    assert work_dirs(env) == []


async def test_source_without_config_or_weights_fails(env):
    env.hf.files.pop("config.json")
    j = await run_job(env, hf_req())
    assert j.state == "failed" and "config.json" in j.error


async def test_unsafe_file_names_are_refused_even_if_select_files_lets_them_through(env, monkeypatch):
    # Defense in depth: select_files filters unsafe names today; the manager must not depend on it.
    real = jobs_mod.select_files

    def lax(files, allow_remote_code=False):
        keep, skipped = real(files, allow_remote_code)
        return [*keep, SourceFile(name="../escape.safetensors", bytes=1)], skipped

    monkeypatch.setattr(jobs_mod, "select_files", lax)
    j = await run_job(env, hf_req())
    assert j.state == "failed" and "unsafe" in j.error
    assert not (env.models / ".hf" / "escape.safetensors").exists()
    assert not (env.models / "escape.safetensors").exists()


async def test_locate_dir_failure_at_run_time(env, monkeypatch):
    monkeypatch.setattr(jobs_mod, "inspect_source", fake_inspect_factory())
    j = await run_job(env, ConvertRequest(source=SourceSpec(path=str(env.tmp / "gone"))))
    assert j.state == "failed" and "no such folder" in j.error


# ---- cancel / shutdown / restart -----------------------------------------------------------------


async def test_cancel_running_job_kills_tool_and_cleans(env, monkeypatch):
    pidf = env.tmp / "pid"
    monkeypatch.setenv("FAKE_CONVERT_MODE", "slow")
    monkeypatch.setenv("FAKE_PID_FILE", str(pidf))
    j = await env.mgr.submit(hf_req(quant="Q4_K_M"))
    waiting = await env.mgr.submit(hf_req(quant="Q8_0", name="next.gguf"))
    for _ in range(300):
        if pidf.exists() and env.mgr.get(j.id).stage_progress == 1.0:
            break
        await asyncio.sleep(0.05)
    running = env.mgr.get(j.id)
    assert running.state == "converting" and running.stage_progress == 1.0  # tqdm "100%" parsed
    assert any("Writing" in ln for ln in running.log_tail)
    pid = int(pidf.read_text())
    monkeypatch.setenv("FAKE_CONVERT_MODE", "ok")  # the next job runs normally
    c = await env.mgr.cancel(j.id)
    assert c.state == "cancelled" and c.finished_at is not None
    await asyncio.sleep(0.3)
    assert not psutil.pid_exists(pid)
    assert env.lib.added == [] or env.lib.added[0][0] == "next.gguf"
    nxt = await wait_for(env.mgr, waiting.id, ("done", "failed"))
    assert nxt.state == "done", nxt.error  # the worker moved on
    assert work_dirs(env) == []
    assert env.mgr.get(j.id).state == "cancelled"


async def test_cancel_queued_job(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_MODE", "slow")
    first = await env.mgr.submit(hf_req(quant="Q8_0", name="a.gguf"))
    second = await env.mgr.submit(hf_req(quant="Q8_0", name="b.gguf"))
    c = await env.mgr.cancel(second.id)
    assert c.state == "cancelled"
    with pytest.raises(ConvertError) as ei:
        await env.mgr.cancel(second.id)
    assert ei.value.status == 409
    await env.mgr.cancel(first.id)
    await asyncio.sleep(0.2)
    assert env.mgr.get(second.id).state == "cancelled"  # the worker never ran it


async def test_cancel_unknown_job_404(env):
    with pytest.raises(ConvertError) as ei:
        await env.mgr.cancel("nope")
    assert ei.value.status == 404


async def test_shutdown_keeps_job_active_and_kills_tool(env, monkeypatch):
    pidf = env.tmp / "pid"
    monkeypatch.setenv("FAKE_CONVERT_MODE", "slow")
    monkeypatch.setenv("FAKE_PID_FILE", str(pidf))
    j = await env.mgr.submit(hf_req(quant="Q8_0"))
    for _ in range(300):
        if pidf.exists():
            break
        await asyncio.sleep(0.05)
    pid = int(pidf.read_text())
    await env.mgr.shutdown()
    await asyncio.sleep(0.3)
    assert not psutil.pid_exists(pid)
    mgr2 = ConvertManager(env.db, env.models, env.tc, env.lib, env.hf)
    assert mgr2.get(j.id).state == "converting"  # still active on disk: start() will requeue it
    await mgr2.shutdown()


async def test_restart_requeues_interrupted_jobs_and_drops_partials(env, monkeypatch):
    j = await env.mgr.submit(hf_req(quant="Q8_0", name="interrupted.gguf"))
    await wait_for(env.mgr, j.id, ("done",))  # let the worker idle, then fabricate a crash state
    k = await env.mgr.submit(hf_req(quant="Q8_0", name="crashed.gguf"))
    await wait_for(env.mgr, k.id, ("done",))
    await env.mgr.shutdown()
    # a new manager over the same db, with a job row stuck mid-conversion and a partial output
    mgr2 = ConvertManager(env.db, env.models, env.tc, env.lib, env.hf)
    stuck = await_submit_sync(mgr2, env)
    mgr2._set_state(mgr2._jobs[stuck], "converting", started_at=1.0)
    partial = env.models / ".convert" / stuck
    partial.mkdir(parents=True)
    (partial / "f16.gguf").write_bytes(b"partial")
    orphan = env.models / ".convert" / "orphan"
    orphan.mkdir()
    (orphan / "x").write_bytes(b"x")
    await mgr2.shutdown()

    mgr3 = ConvertManager(env.db, env.models, env.tc, env.lib, env.hf)
    mgr3.start()
    assert mgr3.get(stuck).state == "queued"  # requeued synchronously by start()
    done = await wait_for(mgr3, stuck, ("done", "failed"))
    assert done.state == "done", done.error
    assert not partial.exists() and not orphan.exists()
    await mgr3.shutdown()


def await_submit_sync(mgr, env):
    """Insert a queued job without a running worker (the manager is not started)."""
    from gpupool.converter.models import ConvertJob

    job = ConvertJob(id="stuck0000001", request=hf_req(quant="Q8_0", name="stuck.gguf"),
                     state="queued", output_name="stuck.gguf", est_output_bytes=3000,
                     created_at=time.time())
    mgr._seq += 1
    mgr._jobs[job.id] = job
    mgr._plans[job.id] = {"params": 2000, "source_bytes": 5000}
    mgr._persist(job, seq=mgr._seq)
    return job.id


# ---- validation outcomes: needs_review / accept / delete ---------------------------------------


async def test_needs_review_keeps_only_the_output_then_accept(env, monkeypatch):
    monkeypatch.setenv("FAKE_TOK_MISMATCH", "1")
    j = await run_job(env, hf_req(quant="Q4_K_M"))
    assert j.state == "needs_review"
    assert j.validation.tokenizer_ok is False and "tokenizer" in j.error
    assert env.lib.added == [] and not (env.models / j.output_name).exists()
    kept = env.models / ".convert" / j.id
    assert [p.name for p in kept.iterdir()] == [j.output_name]
    assert not (env.models / ".hf" / "acme__tiny-model@main").exists()  # source no longer needed
    # cannot retry/delete-active; accept moves it into the library
    with pytest.raises(ConvertError) as ei:
        await env.mgr.retry(j.id)
    assert ei.value.status == 409
    a = await env.mgr.accept(j.id)
    assert a.state == "done" and a.error is None
    assert env.lib.added == [(j.output_name, env.models / j.output_name, REPO)]
    assert (env.models / j.output_name).is_file() and not kept.exists()
    with pytest.raises(ConvertError) as ei:
        await env.mgr.accept(j.id)
    assert ei.value.status == 409


async def test_generation_failure_needs_review_and_delete_removes_output(env, monkeypatch):
    monkeypatch.setenv("FAKE_SIMPLE", "crash")
    j = await run_job(env, hf_req(quant="Q8_0"))
    assert j.state == "needs_review" and j.validation.generation_ok is False
    assert "unable to load model" in j.error
    await env.mgr.delete(j.id)
    assert env.mgr.get(j.id) is None and work_dirs(env) == []
    assert env.lib.added == [] and not list(env.models.glob("*.gguf"))
    with pytest.raises(ConvertError) as ei:
        await env.mgr.delete(j.id)
    assert ei.value.status == 404


async def test_needs_review_keeps_name_reserved_and_accept_rechecks(env, monkeypatch):
    monkeypatch.setenv("FAKE_TOK_MISMATCH", "1")
    j = await run_job(env, hf_req(quant="Q8_0", name="held.gguf"))
    assert j.state == "needs_review"
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req(quant="Q8_0", name="held.gguf"))
    assert ei.value.status == 409
    env.lib.names.add("held.gguf")  # someone registered the name meanwhile
    with pytest.raises(ConvertError) as ei:
        await env.mgr.accept(j.id)
    assert ei.value.status == 409


async def test_generation_not_run_when_disabled(env):
    j = await run_job(env, hf_req(quant="Q8_0", advanced={"validate_generation": False}))
    assert j.state == "done" and j.validation.generation_ok is None


async def test_delete_rules(env, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_MODE", "slow")
    running = await env.mgr.submit(hf_req(quant="Q8_0", name="run.gguf"))
    queued = await env.mgr.submit(hf_req(quant="Q8_0", name="q.gguf"))
    for j in (running, queued):
        with pytest.raises(ConvertError) as ei:
            await env.mgr.delete(j.id)
        assert ei.value.status == 409
    await env.mgr.cancel(queued.id)
    await env.mgr.cancel(running.id)
    # deleting a cancelled job removes its row and its download cache, nothing else
    (env.models / "library-file.gguf").write_bytes(b"precious")
    await env.mgr.delete(queued.id)
    await env.mgr.delete(running.id)
    assert env.mgr.list() == []
    assert (env.models / "library-file.gguf").read_bytes() == b"precious"
    assert not (env.models / ".hf" / "acme__tiny-model@main").exists()


async def test_delete_done_job_never_touches_library_file(env):
    j = await run_job(env, hf_req(quant="Q8_0", name="keep.gguf"))
    assert j.state == "done"
    await env.mgr.delete(j.id)
    assert env.mgr.get(j.id) is None
    assert (env.models / "keep.gguf").is_file()  # the registered model stays


# ---- download cache ----------------------------------------------------------------------------


async def test_cache_shared_by_two_jobs_of_same_repo_then_removed(env):
    a = await env.mgr.submit(hf_req(quant="Q8_0", name="a.gguf"))
    b = await env.mgr.submit(hf_req(quant="Q8_0", name="b.gguf"))
    ra = await wait_for(env.mgr, a.id, ("done", "failed"))
    rb = await wait_for(env.mgr, b.id, ("done", "failed"))
    assert ra.state == rb.state == "done", (ra.error, rb.error)
    assert sorted(env.hf.network) == ["config.json", "model.safetensors", "tokenizer.json"]
    assert not (env.models / ".hf" / "acme__tiny-model@main").exists()


async def test_keep_source_keeps_cache_and_next_job_reuses_it(env):
    a = await run_job(env, hf_req(quant="Q8_0", name="a.gguf", keep_source=True))
    cache = env.models / ".hf" / "acme__tiny-model@main"
    assert a.state == "done" and (cache / "model.safetensors").is_file()
    n = len(env.hf.network)
    b = await run_job(env, hf_req(quant="Q8_0", name="b.gguf"))
    assert b.state == "done" and len(env.hf.network) == n
    assert not cache.exists()  # b did not ask to keep it


async def test_remote_code_only_downloaded_when_allowed(env, monkeypatch):
    seen = env.tmp / "seen.jsonl"
    monkeypatch.setenv("FAKE_SEEN_FILE", str(seen))
    j = await run_job(env, hf_req(quant="Q8_0", name="rc.gguf", keep_source=True,
                                  advanced={"allow_remote_code": True}))
    assert j.state == "done", j.error
    assert "modeling_tiny.py" in json.loads(seen.read_text().splitlines()[0])["seen"]
    # the cache now holds the .py, but a job that does not allow remote code must not see it
    j2 = await run_job(env, hf_req(quant="Q8_0", name="norc.gguf"))
    assert j2.state == "done", j2.error
    assert "modeling_tiny.py" not in json.loads(seen.read_text().splitlines()[1])["seen"]


# ---- path sources --------------------------------------------------------------------------------


def make_local_model(root: Path) -> Path:
    d = root / "local-model"
    d.mkdir()
    (d / "config.json").write_text(CONFIG)
    (d / "model.safetensors").write_bytes(b"W" * 5000)
    (d / "tokenizer.json").write_text("{}")
    (d / "modeling_local.py").write_text("raise SystemExit('never')")
    (d / "notes.txt").write_text("keep me")
    return d


async def test_path_source_staging_excludes_py_and_never_touches_source(env, monkeypatch):
    d = make_local_model(env.tmp)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()}
    seen = env.tmp / "seen.jsonl"
    monkeypatch.setenv("FAKE_SEEN_FILE", str(seen))
    req = ConvertRequest(source=SourceSpec(path=str(d)), quant="Q8_0")
    j = await run_job(env, req)
    assert j.state == "done", j.error
    assert j.output_name == "local-model-Q8_0.gguf"
    assert json.loads(seen.read_text().splitlines()[0])["seen"] == [
        "config.json", "model.safetensors", "tokenizer.json"]
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in d.iterdir()} == before
    assert env.lib.added[0][2] is None  # no hf_repo for a local source
    assert not (env.models / ".hf").exists() or not any((env.models / ".hf").iterdir())


async def test_path_source_remote_code_when_allowed(env, monkeypatch):
    d = make_local_model(env.tmp)
    seen = env.tmp / "seen.jsonl"
    monkeypatch.setenv("FAKE_SEEN_FILE", str(seen))
    req = ConvertRequest(source=SourceSpec(path=str(d)), quant="Q8_0",
                         advanced=ConvertAdvanced(allow_remote_code=True))
    j = await run_job(env, req)
    assert j.state == "done", j.error
    assert "modeling_local.py" in json.loads(seen.read_text().splitlines()[0])["seen"]
    assert (d / "modeling_local.py").exists()


async def test_staging_falls_back_to_copy_when_links_fail(env, monkeypatch):
    d = make_local_model(env.tmp)

    def nope(*a, **k):
        raise OSError("links not supported")

    monkeypatch.setattr(os, "symlink", nope)
    monkeypatch.setattr(os, "link", nope)
    j = await run_job(env, ConvertRequest(source=SourceSpec(path=str(d)), quant="Q8_0"))
    assert j.state == "done", j.error
    assert (d / "model.safetensors").read_bytes() == b"W" * 5000


# ---- internals worth pinning ---------------------------------------------------------------------


def test_rmtree_refuses_paths_outside_converter_folders(env, tmp_path):
    outside = tmp_path / "models" / "important"
    outside.mkdir(parents=True)
    (outside / "f").write_text("x")
    env.mgr._rmtree(outside)
    env.mgr._rmtree(env.models / ".convert")  # the root itself is not removable either
    env.mgr._rmtree(env.models)
    assert (outside / "f").exists()


def test_cache_dir_rejects_traversal(env):
    with pytest.raises(ConvertError):
        env.mgr._cache_dir("../../etc/passwd", "main")
    p = env.mgr._cache_dir("a/b", "refs/pr/1")
    assert p.parent == env.models / ".hf" and "/" not in p.name


async def test_progress_persisted_at_most_once_per_second(env, monkeypatch):
    j = await env.mgr.submit(hf_req(quant="Q8_0"))
    await wait_for(env.mgr, j.id, ("done",))
    writes = []
    orig = env.mgr._persist
    monkeypatch.setattr(env.mgr, "_persist", lambda job, **k: (writes.append(1), orig(job, **k))[1])
    job = env.mgr._jobs[j.id]
    for i in range(500):
        env.mgr._progress(job, stage_progress=i / 500)
    assert len(writes) <= 1
    assert env.mgr.get(j.id).stage_progress == pytest.approx(499 / 500)  # live value is current


async def test_log_tail_keeps_last_lines_and_collapses_progress(env):
    j = await env.mgr.submit(hf_req(quant="Q8_0"))
    await wait_for(env.mgr, j.id, ("done",))
    job = env.mgr._jobs[j.id]
    for i in range(300):
        env.mgr._log(job, f"line {i}")
    for pct in range(10):
        env.mgr._log(job, f"Writing: {pct}%|##| 1/2")
    tail = env.mgr.get(j.id).log_tail
    assert len(tail) == 200 and tail[-1] == "Writing: 9%|##| 1/2"
    assert sum("Writing" in t for t in tail) == 1
    env.mgr._persist(job)
    await env.mgr.shutdown()
    mgr2 = ConvertManager(env.db, env.models, env.tc, env.lib, env.hf)
    assert len(mgr2.get(j.id).log_tail) == 50
    await mgr2.shutdown()


# ---- early disk check ----------------------------------------------------------------------------


async def test_submit_507_when_disk_cannot_hold_the_job(env, monkeypatch):
    usage = types.SimpleNamespace(total=10**12, used=10**12 - 1_000_000, free=1_000_000)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage)
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req())
    assert ei.value.status == 507
    assert str(env.models) in ei.value.message and "GB" in ei.value.message and "free" in ei.value.message
    assert env.mgr.list() == [] and env.hf.network == []


async def test_early_disk_check_counts_what_is_cached_and_the_intermediate(env, monkeypatch):
    seen = []
    monkeypatch.setattr(env.mgr, "_check_disk", lambda need, what: seen.append((need, what)))
    files = [SourceFile(name="model.safetensors", bytes=5000)]
    monkeypatch.setattr(jobs_mod, "inspect_source", fake_inspect_factory(files=files))
    margin = jobs_mod._DISK_MARGIN
    first = await env.mgr.submit(hf_req(quant="Q4_K_M", name="a.gguf"))
    need, what = seen[0]
    # 5000 download + 2000 params * 2 bytes of 16-bit intermediate + 3000 estimated output
    assert need == 5000 + 4000 + 3000 + margin and REPO in what
    # a direct type has no intermediate; a cached file is not downloaded again
    cached = env.models / ".hf" / "acme__tiny-model@main"
    cached.mkdir(parents=True, exist_ok=True)
    (cached / "model.safetensors").write_bytes(b"W" * 5000)
    second = await env.mgr.submit(hf_req(quant="Q8_0", name="b.gguf"))
    assert seen[1][0] == 3000 + margin
    for j in (first, second):
        await env.mgr.cancel(j.id)


# ---- importance matrix ---------------------------------------------------------------------------


@pytest.fixture
def calib(env, monkeypatch):
    f = env.tmp / "builtin-calibration.txt"
    f.write_text("builtin calibration text", encoding="utf-8")
    monkeypatch.setattr(jobs_mod, "BUILTIN_CALIBRATION", f)
    monkeypatch.setenv("FAKE_IMATRIX_LOG", str(env.tmp / "imatrix.jsonl"))
    monkeypatch.setenv("FAKE_ARGV_LOG", str(env.tmp / "argv.jsonl"))
    return f


def imatrix_log(env):
    f = env.tmp / "imatrix.jsonl"
    return [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []


async def test_auto_mode_decides_per_type(env, calib, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_MODE", "slow")  # jobs only need to be submitted
    for quant, expect in (("Q6_K", False), ("Q4_K_M", False), ("Q3_K_M", False), ("Q3_K_S", True),
                          ("Q2_K", True), ("IQ3_S", True), ("IQ2_M", True), ("IQ1_S", True),
                          ("Q8_0", False), ("F16", False)):
        job = await env.mgr.submit(hf_req(quant=quant, name=f"{quant}.gguf"))
        assert job.imatrix_used is expect, quant
        await env.mgr.cancel(job.id)


async def test_imatrix_job_runs_calibrating_stage_and_feeds_quantize(env, calib):
    job = await env.mgr.submit(hf_req(quant="IQ2_M", advanced={"imatrix_chunks": 7, "threads": 3}))
    assert job.imatrix_used is True
    states = []
    while True:
        j = env.mgr.get(job.id)
        if not states or states[-1] != j.state:
            states.append(j.state)
        if j.state in ("done", "failed"):
            break
        await asyncio.sleep(0.005)
    assert j.state == "done", j.error
    order = [x for x in states if x in ("converting", "calibrating", "quantizing", "validating")]
    assert order == ["converting", "calibrating", "quantizing", "validating"]
    (rec,) = imatrix_log(env)
    a = rec["args"]
    assert a[a.index("--chunks") + 1] == "7" and a[a.index("-t") + 1] == "3" and "--no-ppl" in a
    assert rec["calib"] == "builtin calibration text" and rec["model_exists"] is True
    qargs = json.loads((env.tmp / "argv.jsonl").read_text().splitlines()[0])
    assert qargs[0] == "--imatrix" and qargs[1].endswith("imatrix.gguf")
    assert j.imatrix_used is True and j.failed_stage is None
    assert work_dirs(env) == []


async def test_default_chunks_and_user_calibration_text_is_copied(env, calib):
    user = env.tmp / "mine.txt"
    user.write_text("my own text", encoding="utf-8")
    j = await run_job(env, hf_req(quant="Q2_K", advanced={"calibration_path": str(user)}))
    assert j.state == "done", j.error
    (rec,) = imatrix_log(env)
    a = rec["args"]
    assert a[a.index("--chunks") + 1] == "100"
    assert rec["calib"] == "my own text"
    assert Path(a[a.index("-f") + 1]) != user  # a snapshot in the work dir, not the user's file
    assert a[a.index("-t") + 1] == "2"  # coordinator default threads


async def test_imatrix_on_for_a_normal_type_and_off_skips_it(env, calib):
    j = await run_job(env, hf_req(quant="Q5_K_M", advanced={"imatrix": "on"}))
    assert j.state == "done" and j.imatrix_used is True and len(imatrix_log(env)) == 1
    j = await run_job(env, hf_req(quant="Q2_K", name="off.gguf", advanced={"imatrix": "off"}))
    assert j.state == "done" and j.imatrix_used is False and len(imatrix_log(env)) == 1


async def test_direct_types_ignore_imatrix_on(env, calib):
    job = await env.mgr.submit(hf_req(quant="Q8_0", advanced={"imatrix": "on"}))
    assert job.imatrix_used is False
    j = await wait_for(env.mgr, job.id, ("done", "failed"))
    assert j.state == "done" and imatrix_log(env) == []


async def test_needs_imatrix_with_off_is_422(env, calib):
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req(quant="IQ1_S", advanced={"imatrix": "off"}))
    assert ei.value.status == 422 and "importance matrix" in ei.value.message
    assert env.mgr.list() == []


async def test_missing_imatrix_tool(env, calib, tmp_path):
    env.mgr.toolchain = make_toolkit(tmp_path / "lean", imatrix=False)
    assert env.mgr.imatrix_available() is False
    for quant, adv in (("IQ2_XS", {}), ("Q4_K_M", {"imatrix": "on"})):
        with pytest.raises(ConvertError) as ei:
            await env.mgr.submit(hf_req(quant=quant, advanced=adv))
        assert ei.value.status == 503 and "llama-imatrix is not installed" in ei.value.message
    # auto on a type that merely benefits from one goes without; types that never use one are fine
    j = await run_job(env, hf_req(quant="Q2_K", name="noim.gguf"))
    assert j.state == "done" and j.imatrix_used is False
    j = await run_job(env, hf_req(quant="Q8_0", name="q8.gguf", advanced={"imatrix": "on"}))
    assert j.state == "done"


async def test_calibration_path_validation(env, calib):
    big = env.tmp / "big.txt"
    big.write_bytes(b"x" * (jobs_mod._CALIBRATION_MAX + 1))
    wrong = env.tmp / "notes.md"
    wrong.write_text("x")
    empty = env.tmp / "empty.txt"
    empty.write_text("")
    for path, word in ((str(big), "limit"), (str(wrong), ".txt"), (str(empty), "empty"),
                       (str(env.tmp / "nope.txt"), "no such file")):
        with pytest.raises(ConvertError) as ei:
            await env.mgr.submit(hf_req(quant="IQ2_M", advanced={"calibration_path": path}))
        assert ei.value.status == 422 and word in ei.value.message, path
    # irrelevant when no matrix is computed
    job = await env.mgr.submit(hf_req(quant="Q8_0", advanced={"calibration_path": str(wrong)}))
    await env.mgr.cancel(job.id)


async def test_missing_builtin_calibration_is_503(env, monkeypatch):
    monkeypatch.setattr(jobs_mod, "BUILTIN_CALIBRATION", env.tmp / "absent.txt")
    with pytest.raises(ConvertError) as ei:
        await env.mgr.submit(hf_req(quant="IQ2_M"))
    assert ei.value.status == 503 and "calibration" in ei.value.message


async def test_imatrix_failure_reports_stage(env, calib, monkeypatch):
    monkeypatch.setenv("FAKE_IMATRIX_MODE", "fail")
    j = await run_job(env, hf_req(quant="IQ2_M"))
    assert j.state == "failed" and j.failed_stage == "calibrating"
    assert "llama-imatrix failed" in j.error and "tokenizes to only" in j.error
    assert work_dirs(env) == []
    monkeypatch.setenv("FAKE_IMATRIX_MODE", "nofile")
    r = await env.mgr.retry(j.id)
    assert r.failed_stage is None and r.started_at is None and r.finished_at is None
    j2 = await wait_for(env.mgr, j.id, ("failed",))
    assert j2.failed_stage == "calibrating" and "wrote no importance matrix" in j2.error


async def test_failed_stage_for_other_stages_and_cancel(env, calib, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_MODE", "fail")
    j = await run_job(env, hf_req(quant="Q8_0"))
    assert j.state == "failed" and j.failed_stage == "converting"
    monkeypatch.setenv("FAKE_CONVERT_MODE", "ok")
    monkeypatch.setenv("FAKE_QUANT_MODE", "fail")
    j = await run_job(env, hf_req(quant="Q4_K_M", name="q.gguf"))
    assert j.failed_stage == "quantizing"
    created = j.created_at
    monkeypatch.setenv("FAKE_QUANT_MODE", "ok")
    r = await env.mgr.retry(j.id)
    assert r.failed_stage is None and r.created_at == created
    await wait_for(env.mgr, j.id, ("done",))
    # cancel mid-calibration
    monkeypatch.setenv("FAKE_IMATRIX_MODE", "slow")
    job = await env.mgr.submit(hf_req(quant="IQ2_M", name="c.gguf"))
    await wait_for(env.mgr, job.id, ("calibrating",))
    c = await env.mgr.cancel(job.id)
    assert c.state == "cancelled" and c.failed_stage == "calibrating"
    assert work_dirs(env) == []


async def test_failed_stage_and_imatrix_survive_a_restart(env, calib, monkeypatch):
    monkeypatch.setenv("FAKE_IMATRIX_MODE", "fail")
    j = await run_job(env, hf_req(quant="IQ2_M"))
    await env.mgr.shutdown()
    again = ConvertManager(env.db, env.models, env.tc, env.lib, env.hf)
    got = again.get(j.id)
    assert got.failed_stage == "calibrating" and got.imatrix_used is True
    await again.shutdown()
    env.mgr = ConvertManager(env.db, env.models, env.tc, env.lib, env.hf)  # teardown closes it
