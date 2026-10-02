"""Static checks of the management UI and the contract of its dev mock server."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "src" / "gpupool" / "ui"
HEAD = {"Authorization": "Bearer dev"}


@pytest.fixture(scope="module")
def mock():
    spec = importlib.util.spec_from_file_location("ui_mock_server", ROOT / "scripts" / "ui_mock_server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client(mock):
    mock.reset()
    return TestClient(mock.app)


def test_files_exist():
    for rel in ("index.html", "app.js", "styles.css", "vendor/alpine.min.js"):
        assert (UI / rel).is_file(), rel
    assert (UI / "vendor" / "alpine.min.js").stat().st_size > 10_000


def test_index_only_local_assets():
    html = (UI / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href)\s*=\s*"([^"]*)"', html)
    assert refs
    for ref in refs:
        if ref == "#":
            continue
        assert not re.match(r"^(https?:)?//", ref), f"remote asset: {ref}"
    assert "vendor/alpine.min.js" in refs


def test_no_cdn_urls_in_js_and_css():
    for rel in ("app.js", "styles.css"):
        text = (UI / rel).read_text(encoding="utf-8")
        assert "cdn" not in text.lower()


def test_alpine_has_license_header():
    assert "MIT" in (UI / "vendor" / "alpine.min.js").read_text(encoding="utf-8")[:300]


def test_mock_requires_auth(client):
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_mock_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "gpupool" in r.text


def test_mock_state_matches_contract(client):
    st = client.get("/api/state", headers=HEAD).json()
    assert {"summary", "servers", "models", "library", "settings", "events", "unread_events"} <= st.keys()
    assert {"servers_total", "servers_online", "gpus_total", "gpus_enabled", "pool_total_mb", "pool_usable_mb",
            "models_running"} <= st["summary"].keys()
    assert {"public_url", "cluster_token", "api_keys_set"} <= st["settings"].keys()
    assert len(st["servers"]) == 3 and sum(1 for s in st["servers"] if not s["alive"]) == 1
    for s in st["servers"]:
        assert {"node_id", "agent_url", "added_at", "alive", "last_seen", "report", "gpu_enabled"} <= s.keys()
        assert {"node_id", "host", "devices", "cpu_pct", "ram_used_mb", "ram_total_mb"} <= s["report"].keys()
        for d in s["report"]["devices"]:
            assert {"device_id", "kind", "name", "total_mb", "free_mb", "usable_mb", "util_pct", "temp_c", "power_w",
                    "processes", "driver", "cuda"} <= d.keys()
    assert sum(1 for s in st["servers"] for d in s["report"]["devices"] if d["kind"] == "cuda") == 8
    m = st["models"][0]
    assert {"spec", "file", "state", "error", "replicas"} <= m.keys()
    assert {"name", "source", "ctx_size", "parallel", "replicas", "pin_devices"} <= m["spec"].keys()
    assert {"name", "path", "source", "bytes", "downloaded", "status", "error", "created_at"} <= st["library"][0].keys()


def test_mock_state_has_bandwidth_and_policy(client):
    st = client.get("/api/state", headers=HEAD).json()
    cuda = [d for s in st["servers"] for d in s["report"]["devices"] if d["kind"] == "cuda"]
    assert cuda and all(d["bandwidth_gbps"] for d in cuda)
    assert {"priority", "spread"} <= st["models"][0]["spec"].keys()


def test_mock_model_put_accepts_priority_spread(client):
    body = {"file": "qwen2.5-3b-q4.gguf", "ctx_size": 2048, "parallel": 1, "pin_devices": [], "priority": 80, "spread": "node"}
    spec = client.put("/api/models/m2", json=body, headers=HEAD).json()
    assert (spec["priority"], spec["spread"]) == (80, "node")
    assert client.put("/api/models/m2", json={**body, "priority": 101}, headers=HEAD).status_code == 422
    assert client.put("/api/models/m2", json={**body, "spread": "x"}, headers=HEAD).status_code == 422


def test_mock_recommend_options_and_not_possible(client):
    req = {"file": "llama-8b.gguf", "ctx_size": 4096, "parallel": 1, "priority": 50, "spread": "gpu", "pin_devices": [], "limit": 3}
    r = client.post("/api/recommend", json=req, headers=HEAD).json()
    assert {"need_mb", "options", "max_ctx_single_gpu", "not_possible"} <= r.keys()
    assert r["not_possible"] is None and 1 <= len(r["options"]) <= 3
    o = r["options"][0]
    assert {"rank", "score", "tier", "fits_now", "assignments", "est_decode_tps", "est_total_mb", "reasons"} <= o.keys()
    assert {"node_id", "device_id", "layers", "est_mb"} <= o["assignments"][0].keys()
    r = client.post("/api/recommend", json={**req, "ctx_size": 200_000}, headers=HEAD).json()
    assert r["options"] == []
    assert {"need_mb", "largest_single_gpu_mb", "largest_single_node_mb", "max_ctx_that_fits"} <= r["not_possible"].keys()
    assert client.post("/api/recommend", json={**req, "file": "nope.gguf"}, headers=HEAD).status_code == 404


def test_mock_capacity_contract(client):
    r = client.get("/api/capacity", headers=HEAD).json()
    assert {"gpus", "summary"} <= r.keys() and len(r["gpus"]) == 8
    assert {"node_id", "device_id", "uuid", "name", "kind", "enabled", "alive", "total_mb", "usable_mb", "free_for_new_mb",
            "reserved_mb", "bandwidth_gbps", "busy", "replicas"} <= r["gpus"][0].keys()
    assert {"gpus", "free_for_new_mb", "largest_single_gpu_mb", "largest_single_node_mb"} <= r["summary"].keys()


def test_ui_uses_recommend_and_policy_fields():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    assert "/api/recommend" in js and "priority" in js and "spread" in js
    for needle in ("Recommend placement", "Pin to these GPUs", "Different GPUs", "Different servers", "Bandwidth"):
        assert needle in html, needle


def test_mock_state_has_scaling_and_idle_model(client):
    st = client.get("/api/state", headers=HEAD).json()
    by = {m["spec"]["name"]: m for m in st["models"]}
    for m in st["models"]:
        assert {"min", "max", "desired", "avg_busy", "unloaded"} <= m["scaling"].keys()
    assert by["chat-demand"]["state"] == "idle" and by["chat-demand"]["scaling"]["unloaded"] is True
    assert by["chat-demand"]["spec"]["idle_unload_s"] == 600 and by["chat-demand"]["spec"]["min_replicas"] == 0
    assert by["chat-auto"]["scaling"]["avg_busy"] is not None and by["chat-auto"]["scaling"]["max"] == 4
    kinds = {e["kind"] for e in st["events"]}
    assert {"scaled_up", "cold_start", "unloaded_idle"} <= kinds


def test_mock_model_put_scaling_validation(client):
    body = {"file": "qwen2.5-3b-q4.gguf", "ctx_size": 2048, "parallel": 1, "pin_devices": []}
    auto = {"target_busy": 0.6, "up_after_s": 20, "down_after_s": 120}
    spec = client.put("/api/models/s1", json={**body, "min_replicas": 1, "max_replicas": 3, "autoscale": auto}, headers=HEAD).json()
    assert (spec["min_replicas"], spec["max_replicas"], spec["autoscale"]) == (1, 3, auto)
    spec = client.put("/api/models/s1", json={**body, "min_replicas": 0, "max_replicas": 2, "idle_unload_s": 900, "autoscale": auto},
                      headers=HEAD).json()
    assert spec["idle_unload_s"] == 900
    spec = client.put("/api/models/s1", json={**body, "min_replicas": None, "max_replicas": None, "autoscale": None,
                                              "idle_unload_s": None}, headers=HEAD).json()
    assert spec["min_replicas"] is None and spec["autoscale"] is None and spec["idle_unload_s"] is None
    for bad in ({"min_replicas": 3, "max_replicas": 2}, {"max_replicas": 0}, {"min_replicas": -1},
                {"min_replicas": 1, "max_replicas": 2, "idle_unload_s": 60},  # idle unload only with min 0
                {"autoscale": {"target_busy": 1.5, "up_after_s": 1, "down_after_s": 1}}):
        r = client.put("/api/models/s1", json={**body, **bad}, headers=HEAD)
        assert r.status_code == 422 and isinstance(r.json()["detail"], str), bad


def test_mock_scaling_endpoint(client):
    r = client.get("/api/models/chat-auto/scaling", headers=HEAD).json()
    assert {"model", "min", "max", "desired", "ready", "launching", "avg_busy", "queued", "idle_s", "state", "last_decision",
            "replicas"} <= r.keys()
    assert r["ready"] == len(r["replicas"]) == 2 and {"ts", "action", "reason"} <= r["last_decision"].keys()
    assert {"replica_id", "busy", "requests_processing", "requests_deferred", "measured_decode_tps", "est_decode_tps",
            "metrics_ok"} <= r["replicas"][0].keys()
    idle = client.get("/api/models/chat-demand/scaling", headers=HEAD).json()
    assert idle["state"] == "unloaded" and idle["replicas"] == []
    assert client.get("/api/models/nope/scaling", headers=HEAD).status_code == 404


def test_ui_has_scaling_controls():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    for needle in ("/scaling", "min_replicas", "max_replicas", "autoscale", "idle_unload_s", "target_busy", "scaled_up", "cold_start"):
        assert needle in js, needle
    for needle in ("Scaling", "Fixed", "Autoscale", "On demand", "Min replicas", "Max replicas", "Target busy", "Scale up after",
                   "Scale down after", "Unload after", "Idle &mdash; loads on first request", "Scaling details", "Replicas desired"):
        assert needle in html, needle


def test_html_never_renders_server_data_as_html():
    # x-html is only for the static icon() helper; data fields must use x-text.
    html = (UI / "index.html").read_text(encoding="utf-8")
    for expr in re.findall(r'x-html="([^"]*)"', html):
        assert expr.startswith("icon("), expr


def test_mock_events_contract(client, mock):
    mock.kill("CTG-Server-2")
    evs = client.get("/api/events?limit=10", headers=HEAD).json()
    assert evs and {"id", "ts", "level", "kind", "message", "node_id", "model", "read"} <= evs[0].keys()
    st = client.get("/api/state", headers=HEAD).json()
    assert st["unread_events"] == len([e for e in evs if not e["read"]]) > 0
    top = evs[0]["id"]
    assert client.post("/api/events/read", json={"up_to_id": top}, headers=HEAD).status_code == 200
    assert client.get("/api/state", headers=HEAD).json()["unread_events"] == 0
    assert client.get(f"/api/events?after_id={top}", headers=HEAD).json() == []


def test_mock_model_flow(client):
    assert client.put("/api/models/m1", json={"file": "qwen2.5-3b-q4.gguf", "ctx_size": 2048, "parallel": 1, "pin_devices": []},
                      headers=HEAD).status_code == 200
    assert client.post("/api/models/m1/start", json={"replicas": 1}, headers=HEAD).status_code == 200
    states = {m["spec"]["name"]: m["state"] for m in client.get("/api/state", headers=HEAD).json()["models"]}
    assert states["m1"] == "starting"
    assert client.post("/api/models/m1/plan", headers=HEAD).json()["assignments"]
    client.put("/api/models/m1", json={"file": "qwen2.5-3b-q4.gguf", "ctx_size": 65536, "parallel": 1, "pin_devices": []}, headers=HEAD)
    r = client.post("/api/models/m1/plan", headers=HEAD)
    assert r.status_code == 409 and isinstance(r.json()["detail"], str)
    assert client.delete("/api/models/m1", headers=HEAD).status_code == 200
