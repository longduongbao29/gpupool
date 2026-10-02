"""Static checks of the management UI and the contract of its dev mock server."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
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


def test_mock_preemptible_roundtrip_and_event(client):
    st = client.get("/api/state", headers=HEAD).json()
    assert all(isinstance(m["spec"]["preemptible"], bool) for m in st["models"])
    assert "preempted" in {e["kind"] for e in st["events"]}
    assert next(e for e in st["events"] if e["kind"] == "preempted")["level"] == "warning"
    body = {"file": "qwen2.5-3b-q4.gguf", "ctx_size": 2048, "parallel": 1, "pin_devices": []}
    assert client.put("/api/models/p1", json={**body, "preemptible": False}, headers=HEAD).json()["preemptible"] is False
    assert client.put("/api/models/p1", json=body, headers=HEAD).json()["preemptible"] is True
    assert client.put("/api/models/p1", json={**body, "preemptible": "no"}, headers=HEAD).status_code == 422


def test_mock_recommend_requires_preemption(client):
    req = {"file": "llama-8b.gguf", "ctx_size": 4096, "parallel": 1, "priority": 90, "spread": "gpu", "pin_devices": [], "limit": 3}
    opts = client.post("/api/recommend", json=req, headers=HEAD).json()["options"]
    needy = [o for o in opts if not o["fits_now"]]
    assert needy and {"replica_id", "model", "priority"} <= needy[0]["requires_preemption"][0].keys()
    low = client.post("/api/recommend", json={**req, "priority": 10}, headers=HEAD).json()["options"]
    assert all(o["fits_now"] and not o.get("requires_preemption") for o in low)


def test_mock_simulate_contract_and_purity(client):
    def sim(body):
        return client.post("/api/simulate", json=body, headers=HEAD)

    before = client.get("/api/state", headers=HEAD).json()["models"]
    keys = {"start", "stop", "preempt", "unplaced"}
    r = sim({"add": [{"name": "big", "file": "qwen2.5-3b-q4.gguf", "priority": 90, "replicas": 4}]}).json()
    assert keys <= r.keys() and r["start"] and r["preempt"] and r["stop"] and r["unplaced"]
    assert {"model", "tier", "assignments", "est_decode_tps"} <= r["start"][0].keys()
    assert {"node_id", "device_id", "layers", "est_mb"} <= r["start"][0]["assignments"][0].keys()
    assert {"replica_id", "model", "priority", "for_model"} <= r["preempt"][0].keys()
    assert {"replica_id", "model", "reason"} <= r["stop"][0].keys() and {"model", "missing", "why"} <= r["unplaced"][0].keys()
    r = sim({"add": [{"name": "huge", "file": "llama-8b.gguf", "ctx_size": 200_000}]}).json()
    assert r["unplaced"] and not r["start"]
    r = sim({"changes": [{"model": "chat-auto", "priority": 70}]}).json()
    assert not any(r[k] for k in keys)
    assert sim({"changes": [{"model": "nope", "priority": 70}]}).status_code == 404
    assert sim({}).status_code == 422
    assert client.get("/api/state", headers=HEAD).json()["models"] == before  # pure: nothing changed
    assert "huge" not in {m["spec"]["name"] for m in client.get("/api/state", headers=HEAD).json()["models"]}


def test_ui_has_preemption_and_preview():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    css = (UI / "styles.css").read_text(encoding="utf-8")
    for needle in ("/api/simulate", "preemptible", "preempted", "changes", "add"):
        assert needle in js, needle
    for needle in ("Can be preempted", "a higher-priority model may stop", "Preview impact", "Will start", "Will stop",
                   "Will be preempted", "Cannot be placed", "No change", "Needs room: would stop", "requires_preemption"):
        assert needle in html, needle
    assert ".warn" in css


def test_mock_rebalance_flow(client, mock):
    st = client.get("/api/state", headers=HEAD).json()
    assert st["rebalance"]["in_progress"] is None and st["rebalance"]["next_run_ts"] > st["events"][0]["ts"]
    assert {"rebalance_failed"} <= {e["kind"] for e in st["events"]}
    r = client.post("/api/rebalance", json={"dry_run": True}, headers=HEAD).json()
    assert r["started"] is None and r["in_progress"] is None and len(r["moves"]) == 1
    mv = r["moves"][0]
    assert {"replica_id", "model", "from", "to", "current_score", "new_score", "gain", "reasons"} <= mv.keys()
    assert {"node_id", "device_id"} <= mv["to"][0].keys() and mv["gain"] >= 25
    assert client.get("/api/state", headers=HEAD).json()["rebalance"]["in_progress"] is None  # dry run changes nothing
    assert client.post("/api/rebalance", json={"dry_run": "yes"}, headers=HEAD).status_code == 422
    r = client.post("/api/rebalance", json={"dry_run": False}, headers=HEAD).json()
    assert r["started"] == {"replica_id": mv["replica_id"], "model": mv["model"]}
    ip = r["in_progress"]
    assert {"model", "old", "new", "since"} <= ip.keys() and ip["old"] == mv["replica_id"]
    st = client.get("/api/state", headers=HEAD).json()
    assert st["rebalance"]["in_progress"] == ip
    reps = {x["replica_id"]: x["state"] for m in st["models"] if m["spec"]["name"] == ip["model"] for x in m["replicas"]}
    assert reps[ip["old"]] == "ready" and reps[ip["new"]] == "starting"
    again = client.post("/api/rebalance", json={"dry_run": False}, headers=HEAD).json()
    assert again["started"] is None and again["moves"] == [] and again["in_progress"] == ip  # one move at a time
    mock.SCHEDULED[-1] = (0.0, mock.SCHEDULED[-1][1])  # the completion is queued last; let the pending completion fire on the next tick
    st = client.get("/api/state", headers=HEAD).json()
    assert st["rebalance"]["in_progress"] is None
    reps = {x["replica_id"]: x["state"] for m in st["models"] if m["spec"]["name"] == ip["model"] for x in m["replicas"]}
    assert ip["old"] not in reps and reps[ip["new"]] == "ready"
    kinds = [e["kind"] for e in st["events"]]
    assert "rebalance_started" in kinds and "rebalanced" in kinds
    assert client.post("/api/rebalance", json={"dry_run": True}, headers=HEAD).json()["moves"] == []


def test_ui_has_rebalance_panel():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    for needle in ("/api/rebalance", "dry_run", "rebalance_started", "rebalanced", "rebalance_failed", "next_run_ts",
                   "being replaced", "replacement", "starts a new replica on the better GPUs, then stops the old one"):
        assert needle in js, needle
    for needle in ("Placement health", "Check placement", "Rebalance now", "All replicas are well placed", "rbMove()"):
        assert needle in html, needle


def test_app_js_parses():
    # The text checks above pass on a script that does not parse (it happened: a broken string
    # rendered a blank page). Node is only a dev tool here, so skip when it is absent.
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    r = subprocess.run([node, "--check", str(UI / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_favicon_declared_and_valid_svg(client):
    import xml.etree.ElementTree as ET

    html = (UI / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<link[^>]+rel="icon"[^>]+type="image/svg\+xml"[^>]+href="favicon\.svg"', html)
    assert 'name="theme-color"' in html
    root = ET.fromstring((UI / "favicon.svg").read_text(encoding="utf-8"))
    assert root.tag == "{http://www.w3.org/2000/svg}svg" and root.get("viewBox")
    r = client.get("/favicon.svg")
    assert r.status_code == 200 and "svg" in r.headers["content-type"]


def test_theme_setting_and_reduced_motion():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    css = (UI / "styles.css").read_text(encoding="utf-8")
    assert "gpupool.theme" in js and "data-theme" in js and "setTheme" in html
    assert "prefers-color-scheme: light" in css and "prefers-reduced-motion: reduce" in css


def test_mock_library_browse_contract(client):
    r = client.get("/api/library/browse", headers=HEAD).json()
    assert {"roots", "files", "truncated"} <= r.keys() and r["truncated"] is False
    assert {"path", "exists", "host_path"} <= r["roots"][0].keys() and r["roots"][0]["host_path"]
    for f in r["files"]:
        assert {"path", "name", "bytes", "in_library", "split_part", "broken_link", "host_path"} <= f.keys()
    assert any(f["in_library"] for f in r["files"]) and any(f["split_part"] for f in r["files"]) and any(f["broken_link"] for f in r["files"])
    assert client.get("/api/library/browse").status_code == 401


def test_mock_library_add_path_translates_and_explains(client):
    files = {f["name"]: f for f in client.get("/api/library/browse", headers=HEAD).json()["files"]}
    ok = files["llama-3.1-8b-instruct-q4_k_m.gguf"]
    r = client.post("/api/library", json={"path": ok["host_path"]}, headers=HEAD)  # host path is translated
    assert r.status_code == 200 and r.json()["path"] == ok["path"]
    assert client.get("/api/library/browse", headers=HEAD).json()["files"][1]["in_library"] is True
    r = client.post("/api/library", json={"path": "/data/missing.gguf"}, headers=HEAD)
    assert r.status_code == 400 and "No such file inside the coordinator" in r.json()["detail"] and "/models" in r.json()["detail"]
    assert client.post("/api/library", json={"path": files["deepseek-r1-70b-q4_k_m-00001-of-00003.gguf"]["path"]}, headers=HEAD).status_code == 400


def test_ui_has_file_picker_and_modal_errors():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    assert "/api/library/browse" in js and "host_path" in html and "addMdl.err" in html and "a.err" in js
    for needle in ("Browse server folders", "in library", "split part &ndash; not supported", "broken link",
                   "When the coordinator runs in Docker it only sees mounted folders", "on the host",
                   "it is translated if the folder is mounted", "No .gguf files found", "nothing to browse"):
        assert needle in html, needle


def test_ui_polish_markup():
    html = (UI / "index.html").read_text(encoding="utf-8")
    css = (UI / "styles.css").read_text(encoding="utf-8")
    assert "Replicas desired <b" not in html  # the cryptic duplicate of the Replicas tile is gone
    assert 'class="modal-foot split"' in html and ".foot-group" in css
    assert re.search(r"\.model-grid \{[^}]*align-items: start", css)
    assert re.search(r"\.sc-table \{[^}]*table-layout: fixed", css)


def test_mock_perf_fields_in_state_and_put(client):
    st = client.get("/api/state", headers=HEAD).json()
    for m in st["models"]:
        assert {"kv_cache_type", "speculative", "draft", "draft_n_max"} <= m["spec"].keys()
    auto = next(m for m in st["models"] if m["spec"]["name"] == "chat-auto")["spec"]
    assert (auto["kv_cache_type"], auto["speculative"], auto["draft"]) == ("q8_0", "ngram", None)
    body = {"file": "qwen2.5-3b-q4.gguf", "ctx_size": 2048, "parallel": 1, "pin_devices": []}
    spec = client.put("/api/models/k1", json=body, headers=HEAD).json()
    assert (spec["kv_cache_type"], spec["speculative"], spec["draft"], spec["draft_n_max"]) == ("f16", "none", None, 4)
    spec = client.put("/api/models/k1", json={**body, "kv_cache_type": "q4_0", "speculative": "draft",
                                              "draft_file": "qwen2.5-0.5b-q8.gguf", "draft_n_max": 8}, headers=HEAD).json()
    assert (spec["kv_cache_type"], spec["draft"], spec["draft_n_max"]) == ("q4_0", "coordinator://qwen2.5-0.5b-q8.gguf", 8)


def test_mock_perf_validation_422(client):
    body = {"file": "qwen2.5-3b-q4.gguf", "ctx_size": 2048, "parallel": 1, "pin_devices": [], "speculative": "draft"}
    for bad, text in (({}, "draft_file"), ({"draft_file": "nope.gguf"}, "draft_file"),
                      ({"draft_file": "llama-8b.gguf"}, "draft_file"),  # still downloading, not ready
                      ({"draft_file": "qwen2.5-3b-q4.gguf"}, "differ"),
                      ({"draft_file": "llama-3.2-1b-q8.gguf"}, "does not share the tokenizer"),
                      ({"draft_file": "qwen2.5-0.5b-q8.gguf", "draft_n_max": 17}, "draft_n_max"),
                      ({"draft_file": "qwen2.5-0.5b-q8.gguf", "draft_n_max": 0}, "draft_n_max")):
        r = client.put("/api/models/k2", json={**body, **bad}, headers=HEAD)
        assert r.status_code == 422 and text in r.json()["detail"], (bad, r.text)
    assert client.put("/api/models/k2", json={**body, "speculative": "x"}, headers=HEAD).status_code == 422
    assert client.put("/api/models/k2", json={**body, "speculative": "none", "kv_cache_type": "q2"}, headers=HEAD).status_code == 422
    assert "k2" not in {m["spec"]["name"] for m in client.get("/api/state", headers=HEAD).json()["models"]}  # nothing saved


def test_mock_recommend_accepts_perf_fields(client):
    req = {"file": "qwen2.5-3b-q4.gguf", "ctx_size": 4096, "parallel": 2, "limit": 3}
    full = client.post("/api/recommend", json=req, headers=HEAD).json()
    q4 = client.post("/api/recommend", json={**req, "kv_cache_type": "q4_0"}, headers=HEAD).json()
    assert q4["need_mb"] < full["need_mb"]
    d = client.post("/api/recommend", json={**req, "speculative": "draft", "draft_file": "qwen2.5-0.5b-q8.gguf"}, headers=HEAD).json()
    assert d["options"][0]["draft_est_mb"] > 0 and d["need_mb"] > full["need_mb"]
    assert client.post("/api/recommend", json={**req, "speculative": "draft"}, headers=HEAD).status_code == 422


def test_ui_has_kv_cache_and_speculative_controls():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    for needle in ("kv_cache_type", "speculative", "draft_file", "draft_n_max", "draft_est_mb", "perfBody", "specChips"):
        assert needle in js, needle
    for needle in ("Performance", "KV cache", "Full precision (f16)", "8-bit (q8_0)", "4-bit (q4_0)", "about half the KV memory",
                   "Speculative decoding", "N-gram (no extra memory)", "Draft model", "Draft tokens",
                   "quantizing it can let a model fit on fewer GPUs", "Measured on one small GPU", "specChips(m)", "p.draft"):
        assert needle in html, needle
    # the request bodies of save, recommend and preview impact all carry the fields
    assert js.count("pb.body") >= 3
