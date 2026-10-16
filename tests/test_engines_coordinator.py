"""vLLM / SGLang on the coordinator: HF metadata, tensor-parallel placement, launch, API, router."""
from __future__ import annotations

import json

import httpx
import pytest
import respx
from fastapi import Depends, FastAPI

from gpupool.common.auth import require_bearer
from gpupool.common.models import (
    Device, ModelSpec, NodeReport, ReplicaEndpoint, parse_hf_source,
)
from gpupool.coordinator.api import make_api_router
from gpupool.coordinator.autoscaler import parse_metrics
from gpupool.router.balancer import Balancer
from gpupool.router.proxy import RouterMetrics, make_router
from gpupool.scheduler import tp_placement
from gpupool.scheduler.hf_meta import meta_from_config, weight_files
from gpupool.scheduler.placement import NoFit, plan, rank
from tests.test_coordinator_api import AD, FakeLibrary, FakePoller, item
from tests.test_coordinator_helpers import FakeClient, make_cfg, make_reconciler, settle
from tests.test_coordinator_reconciler import beat

GB = 1024 ** 3

QWEN_VL = {  # Qwen2.5-VL-7B-Instruct, trimmed
    "model_type": "qwen2_5_vl", "vision_config": {"depth": 32},
    "text_config": {"num_hidden_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
                    "num_key_value_heads": 4, "vocab_size": 152064},
}
VL_META = meta_from_config(QWEN_VL, int(16 * GB))


def gpu(i, usable, total=24576, name="RTX 4090", uuid=True) -> Device:
    return Device(device_id=f"CUDA{i}", kind="cuda", name=name, total_mb=total, free_mb=usable + 500,
                  usable_mb=usable, bandwidth_gbps=1000.0, uuid=f"GPU-{i}" if uuid else None)


def server(node_id, devices, engines=("vllm",), host=None) -> NodeReport:
    host = host or {"a": "10.0.0.1", "b": "10.0.0.2"}.get(node_id, "10.0.0.9")
    return NodeReport(node_id=node_id, agent_url=f"http://{host}:7070", host=host, devices=devices, engines=[],
                      llama_version="b1", models=[], ts=0.0, features=[f"engine:{e}" for e in engines])


def vspec(**kw) -> ModelSpec:
    return ModelSpec(**{"name": "vl", "source": "hf://Qwen/Qwen2.5-VL-7B-Instruct", "engine": "vllm",
                        "ctx_size": 32768, "parallel": 4, **kw})


# ---------------------------------------------------------------- sources and metadata
def test_parse_hf_source():
    assert parse_hf_source("hf://a/b") == ("a/b", None)
    assert parse_hf_source("hf://a/b@v1.0") == ("a/b", "v1.0")
    for bad in ("hf://a", "hf://a/b/c", "hf://../b", "a/b", "hf://a/b@x..y"):
        with pytest.raises(ValueError):
            parse_hf_source(bad)


def test_meta_from_vlm_config_reads_the_text_model():
    m = VL_META
    assert (m.n_layers, m.n_embd, m.n_head, m.n_head_kv, m.head_dim) == (28, 3584, 28, 4, 128)
    assert m.file_bytes == 16 * GB and sum(m.layer_bytes) + m.other_bytes == 16 * GB
    assert m.kv_k == [512] * 28 and m.vocab_size == 152064 and m.active_bytes is None


def test_meta_mla_moe_and_layer_types():
    mla = meta_from_config({"model_type": "deepseek_v3", "num_hidden_layers": 4, "hidden_size": 7168,
                            "num_attention_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                            "n_routed_experts": 256, "num_experts_per_tok": 8}, 4000)
    assert mla.kv_k == [576] * 4 and mla.kv_v == [0] * 4
    assert mla.active_bytes[0] < mla.layer_bytes[0]  # a token reads 8 of 256 experts
    hybrid = meta_from_config({"num_hidden_layers": 4, "hidden_size": 1024, "num_attention_heads": 8,
                               "num_key_value_heads": 2, "sliding_window": 512,
                               "layer_types": ["linear_attention", "full_attention", "sliding_attention",
                                               "full_attention"]}, 4000)
    assert hybrid.kv_k == [0, 256, 256, 256] and hybrid.swa == [False, False, True, False]
    with pytest.raises(ValueError):
        meta_from_config({"hidden_size": 1}, 10)
    with pytest.raises(ValueError):
        meta_from_config(QWEN_VL, 0)


def test_weight_files_pick_what_the_engines_load():
    files = [("consolidated.safetensors", 10), ("model-00001-of-00002.safetensors", 4),
             ("model-00002-of-00002.safetensors", 3), ("original/x.safetensors", 99), ("README.md", 1)]
    assert sum(b for _, b in weight_files(files)) == 7
    assert weight_files([("pytorch_model.bin", 5), ("tokenizer.json", 1)]) == [("pytorch_model.bin", 5)]


# ---------------------------------------------------------------- placement
def test_smallest_tensor_parallel_that_fits_on_one_server():
    nodes = [server("a", [gpu(0, 20000), gpu(1, 20000), gpu(2, 20000), gpu(3, 20000)])]
    p = plan(VL_META, vspec(), nodes, "r1", lambda n: 9001)
    assert p.engine == "vllm" and p.tier == "single_gpu" and len(p.assignments) == 1
    assert p.assignments[0].layers == 28 and p.assignments[0].device_uuid == "GPU-0"
    assert 0 < p.gpu_mem_fraction < 1 and p.est_total_mb == p.assignments[0].est_mb
    # a GPU that cannot hold it alone: two GPUs of the same server, never two servers
    small = [server("a", [gpu(0, 11000, 12288, "3060"), gpu(1, 11000, 12288, "3060")]),
             server("b", [gpu(0, 11000, 12288, "3060")])]
    p = plan(VL_META, vspec(), small, "r1", lambda n: 9001)
    assert p.head_node == "a" and [a.device_id for a in p.assignments] == ["CUDA0", "CUDA1"]
    assert p.tier == "single_node" and "tensor parallel over 2 GPUs" in p.reasons


def test_vllm_fraction_takes_more_on_a_bigger_gpu():
    devs = [tp_placement._Dev(server("a", []), gpu(0, 30000, 24576)),
            tp_placement._Dev(server("a", []), gpu(1, 30000, 49152))]
    frac, est = tp_placement.sizing(VL_META, vspec(), devs)
    assert est[1] == pytest.approx(2 * est[0], abs=2) and frac == pytest.approx(est[0] / 24576, abs=1e-3)
    sg_frac, sg_est = tp_placement.sizing(VL_META, vspec(engine="sglang"), devs)
    assert sg_frac < frac  # SGLang's fraction leaves the activations out


def test_explicit_tensor_parallel_and_head_divisibility():
    nodes = [server("a", [gpu(i, 20000) for i in range(4)])]
    p = plan(VL_META, vspec(tensor_parallel=4), nodes, "r1", lambda n: 9001)
    assert len(p.assignments) == 4
    with pytest.raises(NoFit, match="does not divide"):
        plan(VL_META, vspec(tensor_parallel=8), nodes, "r1", lambda n: 9001)  # 28 heads


def test_only_servers_that_run_the_engine_and_never_cpu():
    cpu = Device(device_id="CPU", kind="cpu", name="CPU", total_mb=64000, free_mb=60000, usable_mb=60000)
    nodes = [server("a", [gpu(0, 20000)], engines=()), server("b", [cpu])]
    with pytest.raises(NoFit, match="no server runs vllm"):
        plan(VL_META, vspec(), nodes[:1], "r1", lambda n: 9001)
    with pytest.raises(NoFit, match="needs per GPU"):
        plan(VL_META, vspec(), nodes, "r1", lambda n: 9001)
    ok = plan(VL_META, vspec(engine="sglang"), [server("a", [gpu(0, 20000)], engines=("sglang",))],
              "r1", lambda n: 9001)
    assert ok.engine == "sglang"


def test_rank_scores_existing_tp_placement():
    nodes = [server("a", [gpu(0, 20000)]), server("b", [gpu(0, 20000)])]
    cur = plan(VL_META, vspec(), nodes, "r1", lambda n: 9001)
    out = rank(VL_META, vspec(), nodes, extra=[cur], limit=3)
    assert any(o.replica_id == "r1" for o in out) and len(out) == 2
    assert all(o.engine == "vllm" for o in out)


# ---------------------------------------------------------------- launch
@pytest.fixture
def mock_health():
    with respx.mock(assert_all_called=False) as m:
        yield m.get(url__regex=r"http://10\.0\.0\.\d+:\d+/health").mock(return_value=httpx.Response(200))


async def test_launch_starts_one_hf_engine_without_model_download(mock_health):
    client = FakeClient()
    rec, store, clock = make_reconciler(planner=plan, client=client)

    async def meta_for(spec):
        return VL_META

    rec.meta_for = meta_for
    beat(store, clock, server("a", [gpu(0, 11000, 12288), gpu(1, 11000, 12288)]))
    store.put_model(vspec(source="hf://Qwen/Qwen2.5-VL-7B-Instruct@main", replicas=1, trust_remote_code=True))
    await rec.tick()
    await settle(rec)
    r = store.list_replicas()[0]
    assert r.state == "ready", r.error
    assert client.kinds() == ["start"]  # no /models/ensure, no RPC servers
    s = client.calls[0][3]
    assert s.engine == "vllm" and s.kind == "server" and s.model_path is None
    assert s.hf_repo == "Qwen/Qwen2.5-VL-7B-Instruct" and s.hf_revision == "main"
    assert s.devices == ["CUDA0", "CUDA1"] and s.device_uuids == ["GPU-0", "GPU-1"]
    assert s.gpu_mem_fraction == r.placement.gpu_mem_fraction and s.trust_remote_code
    assert client.memory_calls == []  # no llama.cpp memory calibration
    await rec.shutdown()


# ---------------------------------------------------------------- API
@pytest.fixture
async def api():
    cfg = make_cfg(admin_key="adm", cluster_token="ctok", api_keys=["k"])
    rec, store, clock = make_reconciler(cfg=cfg, planner=plan)
    rec.ranker = rank
    seen = []

    async def meta_for(spec):
        seen.append(spec.source)
        if "gated" in spec.source:
            raise RuntimeError("repo is gated or private")
        return VL_META

    rec.meta_for = meta_for
    app = FastAPI()
    app.include_router(make_api_router(store=store, reconciler=rec, poller=FakePoller(), balancer=Balancer(),
                                       library=FakeLibrary(item()), cfg=cfg, admin_dep=Depends(require_bearer("adm"))))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", headers=AD) as c:
        yield c, store, rec, clock
    await rec.shutdown()


async def test_put_hf_model(api):
    c, store, rec, clock = api
    r = await c.put("/api/models/vl", json={"engine": "vllm", "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
                                            "tensor_parallel": 2, "ctx_size": 8192})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["source"] == "hf://Qwen/Qwen2.5-VL-7B-Instruct" and j["engine"] == "vllm" and j["tensor_parallel"] == 2
    # an update without hf_repo keeps the repo; an explicit null tensor_parallel clears it
    r = await c.put("/api/models/vl", json={"ctx_size": 4096, "tensor_parallel": None})
    assert r.status_code == 200 and r.json()["source"] == j["source"] and r.json()["tensor_parallel"] is None


@pytest.mark.parametrize("body, msg", [
    ({"engine": "vllm"}, "needs hf_repo"),
    ({"engine": "vllm", "hf_repo": "not a repo"}, "invalid Hugging Face repo"),
    ({"engine": "vllm", "hf_repo": "a/gated"}, "cannot read"),
    ({"engine": "vllm", "hf_repo": "a/b", "speculative": "ngram"}, "llama.cpp setting"),
    ({"engine": "sglang", "hf_repo": "a/b", "kv_cache_type": "q8_0"}, "--kv-cache-dtype"),
    ({"engine": "vllm", "hf_repo": "a/b", "tensor_parallel": 8}, "must divide"),
    ({"engine": "llama.cpp"}, "file is required"),
])
async def test_put_hf_model_validation(api, body, msg):
    c, *_ = api
    r = await c.put("/api/models/vl", json=body)
    assert r.status_code == 422 and msg in r.text, r.text


async def test_recommend_and_simulate_hf_model(api):
    c, store, rec, clock = api
    beat(store, clock, server("a", [gpu(0, 20000)]))
    r = await c.post("/api/recommend", json={"engine": "vllm", "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct",
                                             "ctx_size": 8192})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["options"] and j["options"][0]["tier"] == "single_gpu" and j["tips"] == []
    assert j["need_mb"] == tp_placement.need_mb(VL_META, vspec(ctx_size=8192, parallel=1), 1)
    r = await c.post("/api/simulate", json={"add": [{"name": "vl", "engine": "vllm",
                                                     "hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct", "replicas": 1}]})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------- autoscaler metrics
def test_parse_vllm_and_sglang_metrics():
    vllm = ('# HELP vllm:num_requests_running x\n'
            'vllm:num_requests_running{engine="0",model_name="my model"} 3.0\n'
            'vllm:num_requests_running{engine="1",model_name="my model"} 1.0\n'
            'vllm:num_requests_waiting{engine="0",model_name="my model"} 2.0\n')
    assert parse_metrics(vllm) == {"llamacpp:requests_processing": 4.0, "llamacpp:requests_deferred": 2.0}
    sg = 'sglang:num_running_reqs{model_name="m"} 5\nsglang:num_queue_reqs{model_name="m"} 0\n'
    assert parse_metrics(sg) == {"llamacpp:requests_processing": 5.0, "llamacpp:requests_deferred": 0.0}
    assert parse_metrics("llamacpp:requests_processing 2\n") == {"llamacpp:requests_processing": 2.0}


# ---------------------------------------------------------------- router
def router_env(engine: str):
    bodies, paths = [], []

    async def upstream(request: httpx.Request):
        bodies.append(json.loads(request.content))
        paths.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    app = FastAPI()
    app.include_router(make_router(
        get_candidates=lambda m: [ReplicaEndpoint(replica_id="r1", model="m", base_url="http://r1:9000",
                                                  engine=engine)],
        list_models=lambda: ["m"], balancer=Balancer(), metrics=RouterMetrics(), api_keys=[],
        on_replica_error=lambda rid: None, client=httpx.AsyncClient(transport=httpx.MockTransport(upstream))))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t"), bodies, paths


async def test_router_leaves_other_engines_bodies_alone():
    c, bodies, _ = router_env("vllm")
    msg = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    async with c:
        assert (await c.post("/v1/chat/completions", json=msg)).status_code == 200
    assert bodies == [msg]  # no cache_prompt, the image part untouched


@pytest.mark.parametrize("path", ["/v1/embeddings", "/v1/rerank", "/v1/score", "/v1/responses"])
async def test_router_forwards_more_openai_endpoints(path):
    c, bodies, paths = router_env("llama.cpp")
    async with c:
        r = await c.post(path, json={"model": "m", "input": "hello"})
    assert r.status_code == 200 and paths == [path]
    assert "cache_prompt" not in bodies[0]  # only generation requests get it


# ---------------------------------------------------------------- engine advice
from gpupool.converter.models import SourceFile  # noqa: E402
from gpupool.coordinator.engine_advice import ModelFacts, facts_from_config  # noqa: E402
from gpupool.coordinator.engine_advice import recommend as advise  # noqa: E402


def test_facts_from_config():
    vl = facts_from_config({**QWEN_VL, "architectures": ["Qwen2_5_VLForConditionalGeneration"]}, ["config.json"])
    assert (vl.task, vl.vision, vl.quant) == ("generate", True, None)
    awq = facts_from_config({"architectures": ["Qwen2ForCausalLM"], "quantization_config": {"quant_method": "AWQ"}}, [])
    assert awq.quant == "awq" and awq.task == "generate"
    emb = facts_from_config({"architectures": ["XLMRobertaModel"]}, ["modules.json"])
    rr = facts_from_config({"architectures": ["XLMRobertaForSequenceClassification"]}, [], "BAAI/bge-reranker-v2-m3")
    assert emb.task == "embedding" and rr.task == "rerank"


def _placement():
    return plan(VL_META, vspec(), [server("a", [gpu(0, 20000)])], "r1", lambda n: 9001)


def test_advice_for_a_gguf_is_llama_cpp():
    r = advise(ModelFacts(kind="gguf"), {"llama.cpp": ["a"], "vllm": ["a"]}, {"llama.cpp": None}, True)
    assert r["recommended"] == "llama.cpp"
    by = {o["engine"]: o for o in r["options"]}
    assert not by["vllm"]["supported"] and not by["sglang"]["supported"]
    assert by["llama.cpp"]["fits_now"] is False and "does not fit" in by["llama.cpp"]["warnings"][0]


def test_advice_prefers_an_installed_engine_that_fits():
    facts = ModelFacts(kind="hf")
    r = advise(facts, {"llama.cpp": ["a"], "vllm": ["a"], "sglang": []}, {"vllm": _placement()}, True)
    assert r["recommended"] == "vllm"
    by = {o["engine"]: o for o in r["options"]}
    assert by["vllm"]["fits_now"] and by["vllm"]["placement"]["devices"] == ["a/CUDA0"]
    assert "-all" in by["sglang"]["warnings"][0] and by["llama.cpp"]["needs_conversion"]
    # SGLang wins when it is the one that is installed and fits
    r = advise(facts, {"llama.cpp": ["a"], "vllm": [], "sglang": ["a"]}, {"sglang": _placement()}, True)
    assert r["recommended"] == "sglang"


def test_advice_falls_back_to_conversion_when_no_server_fits():
    r = advise(ModelFacts(kind="hf"), {"llama.cpp": ["a", "b"], "vllm": ["a"], "sglang": ["a"]},
               {"vllm": None, "sglang": None}, True)
    assert r["recommended"] == "llama.cpp" and r["options"][0]["needs_conversion"]
    # ...but never for what gpupool cannot convert or llama.cpp cannot serve
    for facts in (ModelFacts(kind="hf", vision=True), ModelFacts(kind="hf", quant="awq"),
                  ModelFacts(kind="hf", task="embedding")):
        r = advise(facts, {"llama.cpp": ["a"], "vllm": ["a"], "sglang": []}, {"vllm": None}, True)
        assert r["recommended"] == "vllm", facts
        assert not {o["engine"]: o for o in r["options"]}["llama.cpp"]["supported"]
    r = advise(ModelFacts(kind="hf"), {"llama.cpp": ["a"], "vllm": [], "sglang": []}, {}, False)
    assert {o["engine"]: o for o in r["options"]}["llama.cpp"]["supported"] is False  # no toolchain


class FakeHf:
    def __init__(self):
        self.calls = 0

    async def list_files(self, repo, rev):
        self.calls += 1
        if "missing" in repo:
            raise RuntimeError("not found")
        return [SourceFile(name="config.json", bytes=10), SourceFile(name="model.safetensors", bytes=16 * GB)]

    async def fetch_json(self, repo, rev, file):
        return {**QWEN_VL, "architectures": ["Qwen2_5_VLForConditionalGeneration"]}


async def test_engine_recommend_api():
    cfg = make_cfg(admin_key="adm", cluster_token="ctok")
    rec, store, clock = make_reconciler(cfg=cfg, planner=plan)
    rec.ranker = rank

    async def meta_for(spec):
        return VL_META

    rec.meta_for = meta_for
    hf = FakeHf()
    app = FastAPI()
    app.include_router(make_api_router(store=store, reconciler=rec, poller=FakePoller(), balancer=Balancer(),
                                       library=FakeLibrary(item()), cfg=cfg, admin_dep=Depends(require_bearer("adm")),
                                       hf=hf, convert_ok=lambda: True))
    beat(store, clock, server("a", [gpu(0, 20000)], engines=("vllm",)), server("b", [gpu(0, 20000)], engines=()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", headers=AD) as c:
        r = await c.post("/api/engines/recommend", json={"hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct", "ctx_size": 8192})
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["recommended"] == "vllm" and j["model"]["vision"] is True
        by = {o["engine"]: o for o in j["options"]}
        assert by["vllm"]["servers"] == ["a"] and by["vllm"]["fits_now"] is True
        assert by["sglang"]["servers"] == [] and by["llama.cpp"]["supported"] is False
        await c.post("/api/engines/recommend", json={"hf_repo": "Qwen/Qwen2.5-VL-7B-Instruct"})
        assert hf.calls == 1  # config.json is read once per repo
        r = await c.post("/api/engines/recommend", json={"file": "x.gguf"})
        assert r.status_code == 200 and r.json()["recommended"] == "llama.cpp"
        assert (await c.post("/api/engines/recommend", json={"hf_repo": "a/missing"})).status_code == 422
        assert (await c.post("/api/engines/recommend", json={})).status_code == 422
        assert (await c.post("/api/engines/recommend", json={"file": "nope.gguf"})).status_code == 422
    await rec.shutdown()
