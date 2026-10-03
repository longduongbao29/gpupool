"""Conversion router and its wiring in create_app, against a fake ConvertManager."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from gpupool.converter.models import (
    ClusterVram, ConvertError, ConvertJob, ConvertRequest, InspectResult, SourceSpec,
)
from gpupool.coordinator.app import create_app
from gpupool.coordinator.library import Library
from gpupool.coordinator.store import ServerRecord, Store
from tests.test_coordinator_helpers import FakeClient, dev, make_cfg, node

AD = {"Authorization": "Bearer adm"}
REQ = {"source": {"hf_repo": "acme/model"}, "quant": "Q4_K_M"}


def job(jid="j1", state="queued") -> ConvertJob:
    return ConvertJob(id=jid, request=ConvertRequest(source=SourceSpec(hf_repo="acme/model")),
                      state=state, output_name="model-Q4_K_M.gguf", created_at=1.0)


class FakeManager:
    def __init__(self):
        self.problem: str | None = None
        self.imatrix = True
        self.jobs = {"j1": job("j1"), "j2": job("j2", "failed")}
        self.fail: ConvertError | None = None
        self.started = self.stopped = 0
        self.calls: list[tuple] = []

    def start(self):
        self.started += 1

    async def shutdown(self):
        self.stopped += 1

    def available(self):
        return self.problem

    def imatrix_available(self):
        return self.imatrix

    def list(self):
        return list(reversed(self.jobs.values()))

    def get(self, job_id):
        return self.jobs.get(job_id)

    def _op(self, name, job_id):
        self.calls.append((name, job_id))
        if self.fail:
            raise self.fail
        if job_id not in self.jobs:
            raise ConvertError(f"no conversion job {job_id}", 404)
        return self.jobs[job_id]

    async def submit(self, req):
        self.calls.append(("submit", req))
        if self.fail:
            raise self.fail
        return job("new")

    async def cancel(self, job_id):
        return self._op("cancel", job_id)

    async def retry(self, job_id):
        return self._op("retry", job_id)

    async def accept(self, job_id):
        return self._op("accept", job_id)

    async def delete(self, job_id):
        self._op("delete", job_id)
        del self.jobs[job_id]


@pytest.fixture
async def env(tmp_path):
    cfg = make_cfg(tmp_path, cluster_token="ctok", admin_key="adm")
    store = Store(":memory:")
    mgr = FakeManager()
    inspected: list[SourceSpec] = []

    async def fake_inspect(spec):
        inspected.append(spec)
        if spec.hf_repo == "bad/repo":
            raise ConvertError("repo not found", 404)
        return InspectResult(source=spec, architecture="LlamaForCausalLM")

    app = create_app(cfg, store=store, client=FakeClient(), start_background=False,
                     library=Library(":memory:", cfg.models_dir),
                     convert_manager=mgr, convert_inspect=fake_inspect)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c, app, store, mgr, inspected


async def test_every_route_needs_the_admin_key(env):
    c, *_ = env
    for method, url in [("GET", "/api/convert/options"), ("POST", "/api/convert/inspect"),
                        ("POST", "/api/convert"), ("GET", "/api/convert"), ("GET", "/api/convert/j1"),
                        ("POST", "/api/convert/j1/cancel"), ("POST", "/api/convert/j1/retry"),
                        ("POST", "/api/convert/j1/accept"), ("DELETE", "/api/convert/j1")]:
        r = await c.request(method, url, json=REQ if method == "POST" else None)
        assert r.status_code == 401, (method, url)
        r = await c.request(method, url, headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401, (method, url)


async def test_options_content(env):
    c, _, _, mgr, _ = env
    body = (await c.get("/api/convert/options", headers=AD)).json()
    assert body["available"] is True and body["problem"] is None
    assert body["cluster"] == {"largest_gpu_mb": 0, "pool_mb": 0}
    from gpupool.converter import quant
    assert [o["type"] for o in body["quant_options"]] == [o.type for o in quant.QUANT_OPTIONS]
    assert all(o["est_bytes"] is None for o in body["quant_options"])
    assert body["imatrix_available"] is True
    assert any(o["needs_imatrix"] for o in body["quant_options"])
    mgr.imatrix = False
    assert (await c.get("/api/convert/options", headers=AD)).json()["imatrix_available"] is False
    mgr.problem = "llama-quantize not found"
    body = (await c.get("/api/convert/options", headers=AD)).json()
    assert body["available"] is False and body["problem"] == "llama-quantize not found"


async def test_inspect_is_wired_and_errors_map(env):
    c, _, _, _, inspected = env
    r = await c.post("/api/convert/inspect", json={"hf_repo": "acme/model", "revision": "v1"}, headers=AD)
    assert r.status_code == 200 and r.json()["architecture"] == "LlamaForCausalLM"
    assert inspected[0].hf_repo == "acme/model" and inspected[0].revision == "v1"
    r = await c.post("/api/convert/inspect", json={"hf_repo": "bad/repo"}, headers=AD)
    assert r.status_code == 404 and r.json()["detail"] == "repo not found"
    r = await c.post("/api/convert/inspect", json={}, headers=AD)  # neither repo nor path
    assert r.status_code == 422


async def test_submit_list_get(env):
    c, _, _, mgr, _ = env
    r = await c.post("/api/convert", json=REQ, headers=AD)
    assert r.status_code == 200 and r.json()["id"] == "new"
    assert mgr.calls[0][1].quant == "Q4_K_M" and mgr.calls[0][1].source.hf_repo == "acme/model"
    assert [j["id"] for j in (await c.get("/api/convert", headers=AD)).json()] == ["j2", "j1"]
    assert (await c.get("/api/convert/j1", headers=AD)).json()["state"] == "queued"
    r = await c.get("/api/convert/zzz", headers=AD)
    assert r.status_code == 404
    assert (await c.post("/api/convert", json={"source": {}}, headers=AD)).status_code == 422
    assert (await c.post("/api/convert", json={**REQ, "quant": "Q9"}, headers=AD)).status_code == 422


async def test_submit_error_mapping(env):
    c, _, _, mgr, _ = env
    for status in (400, 409, 503):
        mgr.fail = ConvertError(f"boom {status}", status)
        r = await c.post("/api/convert", json=REQ, headers=AD)
        assert r.status_code == status and r.json()["detail"] == f"boom {status}"


async def test_job_actions_and_delete(env):
    c, _, _, mgr, _ = env
    for action in ("cancel", "retry", "accept"):
        r = await c.post(f"/api/convert/j1/{action}", headers=AD)
        assert r.status_code == 200 and r.json()["id"] == "j1"
        assert (action, "j1") in mgr.calls
        assert (await c.post(f"/api/convert/nope/{action}", headers=AD)).status_code == 404
    mgr.fail = ConvertError("job is running", 409)
    r = await c.delete("/api/convert/j1", headers=AD)
    assert r.status_code == 409 and r.json()["detail"] == "job is running"
    mgr.fail = None
    r = await c.delete("/api/convert/j1", headers=AD)
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert "j1" not in mgr.jobs


async def test_cluster_vram_from_live_enabled_cuda_devices(env):
    c, app, store, _, _ = env
    now = app.state.reconciler.clock()
    for nid in ("a", "b", "dead"):  # "stray" is not registered: ignored although it heartbeats
        store.add_server(ServerRecord(node_id=nid, agent_url=f"http://{nid}:7070", added_at=0.0))
    store.upsert_node(node("a", devices=[dev("CUDA0", usable=7000), dev("CUDA1", usable=3000)]), now)
    store.upsert_node(node("b", devices=[dev("CUDA0", usable=20000, uuid="GPU-off")]), now)
    store.upsert_node(node("dead", devices=[dev("CUDA0", usable=99000)]), now - 10_000)
    store.upsert_node(node("stray", devices=[dev("CUDA0", usable=88000)]), now)
    # Liveness needs poller evidence: only "dead" has failed polls.
    app.state.poller.failed_polls = lambda nid: 99 if nid == "dead" else 0
    store.set_gpu_enabled("b", "GPU-off", False)
    body = (await c.get("/api/convert/options", headers=AD)).json()
    assert body["cluster"] == {"largest_gpu_mb": 7000, "pool_mb": 10000}
    store.set_gpu_enabled("b", "GPU-off", True)
    body = (await c.get("/api/convert/options", headers=AD)).json()
    assert body["cluster"] == ClusterVram(largest_gpu_mb=20000, pool_mb=30000).model_dump()


async def test_default_inspect_uses_real_wiring(tmp_path, monkeypatch):
    """Without an injected inspect, the app calls source.inspect_source with the library
    locate_dir, the live cluster VRAM and the toolchain architectures."""
    pytest.importorskip("gpupool.converter.toolchain")
    source = pytest.importorskip("gpupool.converter.source")
    seen = {}

    async def fake_inspect_source(spec, *, hf, locate_dir, cluster, supported_architectures):
        seen.update(spec=spec, locate_dir=locate_dir, cluster=cluster, archs=supported_architectures)
        return InspectResult(source=spec)

    monkeypatch.setattr(source, "inspect_source", fake_inspect_source)
    cfg = make_cfg(tmp_path, admin_key="adm")
    lib = Library(":memory:", cfg.models_dir)
    app = create_app(cfg, store=Store(":memory:"), client=FakeClient(), start_background=False,
                     library=lib, convert_manager=FakeManager())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/convert/inspect", json={"path": "/x"}, headers=AD)
    assert r.status_code == 200
    assert seen["locate_dir"] == lib.locate_dir and seen["cluster"] == ClusterVram()


async def test_lifespan_starts_and_stops_the_manager(tmp_path):
    cfg = make_cfg(tmp_path, admin_key="adm")
    mgr = FakeManager()
    app = create_app(cfg, store=Store(":memory:"), client=FakeClient(), start_background=True,
                     library=Library(":memory:", cfg.models_dir), convert_manager=mgr)
    assert app.state.convert is mgr
    async with app.router.lifespan_context(app):
        await asyncio.sleep(0.02)
        assert mgr.started == 1 and mgr.stopped == 0
    assert mgr.stopped == 1


async def test_manager_not_started_without_background(tmp_path):
    cfg = make_cfg(tmp_path, admin_key="adm")
    mgr = FakeManager()
    app = create_app(cfg, store=Store(":memory:"), client=FakeClient(), start_background=False,
                     library=Library(":memory:", cfg.models_dir), convert_manager=mgr)
    async with app.router.lifespan_context(app):
        pass
    assert mgr.started == 0 and mgr.stopped == 1
