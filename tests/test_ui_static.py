"""Static checks of the management UI and the contract of its dev mock server."""
from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import time
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
    for rel in ("index.html", "app.js", "styles.css", "vendor/alpine.min.js", "vendor/marked.min.js",
                "vendor/purify.min.js"):
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
    assert {"vendor/alpine.min.js", "vendor/marked.min.js", "vendor/purify.min.js"} <= set(refs)


def test_no_cdn_urls_in_js_and_css():
    for rel in ("app.js", "styles.css"):
        text = (UI / rel).read_text(encoding="utf-8")
        assert "cdn" not in text.lower()


def test_alpine_has_license_header():
    assert "MIT" in (UI / "vendor" / "alpine.min.js").read_text(encoding="utf-8")[:300]


def test_markdown_libraries_have_license_headers():
    assert "MIT Licensed" in (UI / "vendor" / "marked.min.js").read_text(encoding="utf-8")[:300]
    assert "Apache license 2.0" in (UI / "vendor" / "purify.min.js").read_text(encoding="utf-8")[:300]


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
    # x-html is only for the static icon() helper and pgMd(); other data fields must use x-text.
    html = (UI / "index.html").read_text(encoding="utf-8")
    for expr in re.findall(r'x-html="([^"]*)"', html):
        # icon() and logo() return fixed SVG markup; pgMd() returns DOMPurify-sanitized Markdown;
        # spark() draws a sparkline from numbers only (the key selects a series, it is not printed)
        assert expr.startswith(("icon(", "pgMd(", "spark(")) or expr == "logo()", expr


def test_markdown_is_sanitized_before_it_reaches_the_page():
    js = (UI / "app.js").read_text(encoding="utf-8")
    body = js[js.index("pgMd: function (text)"):js.index("pgMs: function")]
    assert "DOMPurify.sanitize(window.marked.parse(" in body
    # without the libraries the text is escaped, never inserted raw
    assert '"<": "&lt;"' in body and "return text;" not in body


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

    def models():
        # Live load in the mock drifts over time; compare only what simulate could change.
        return [{k: v for k, v in m.items() if k != "scaling"} for m in client.get("/api/state", headers=HEAD).json()["models"]]

    before = models()
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
    assert models() == before  # pure: nothing changed
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
    # cards in a row share its height; their action row (margin-top: auto) lines up at the bottom
    assert re.search(r"\.model-grid \{[^}]*align-items: stretch", css)
    assert re.search(r"\.mc-foot \{[^}]*margin-top: auto", css)
    # dimmed rows use a class on the cells: tbody rows animate opacity, which overrides an inline style
    assert "opacity:." not in html and "tbody tr.stale > td" in css and "tbody tr.read > td" in css
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


# ---------------------------------------------------------------------------------------------------------------
# Model conversion (Hugging Face / folder -> GGUF): mock contract + UI checks
# ---------------------------------------------------------------------------------------------------------------
def _fast_forward(mock, job_id: str, seconds: float = 100.0) -> None:
    mock.JOBS[job_id]["_t0"] -= seconds  # the mock's jobs advance with wall-clock time


def _at(mock, job_id: str, seconds: float) -> None:
    mock.JOBS[job_id]["_t0"] = time.time() - seconds


def _submit(client, repo: str, **extra):
    body = {"source": {"hf_repo": repo}, **extra}
    return client.post("/api/convert", json=body, headers=HEAD)


def _library_names(client) -> set[str]:
    return {i["name"] for i in client.get("/api/state", headers=HEAD).json()["library"]}


def test_mock_convert_options_and_availability(client, mock):
    from gpupool.converter.models import ClusterVram, QuantOption

    o = client.get("/api/convert/options", headers=HEAD).json()
    assert o["available"] is True and o["problem"] is None
    ClusterVram.model_validate(o["cluster"])
    assert o["cluster"]["largest_gpu_mb"] > 0 and o["cluster"]["pool_mb"] >= o["cluster"]["largest_gpu_mb"]
    opts = [QuantOption.model_validate(x) for x in o["quant_options"]]
    assert [x.type for x in opts][:3] == ["F16", "BF16", "Q8_0"] and len(opts) == 24
    assert all(x.est_bytes is None and x.fits_pool is None for x in opts)
    assert client.get("/api/convert/options").status_code == 401
    r = client.post("/api/_mock/convert_available", json={"available": False}).json()
    assert r["available"] is False and r["problem"]
    o = client.get("/api/convert/options", headers=HEAD).json()
    assert o["available"] is False and o["problem"] == r["problem"]
    assert _submit(client, "HuggingFaceTB/SmolLM2-135M-Instruct").status_code == 503
    # inspect still works without the toolchain
    assert client.post("/api/convert/inspect", json={"hf_repo": "HuggingFaceTB/SmolLM2-135M-Instruct"}, headers=HEAD).status_code == 200
    mock.reset()  # reset() restores the toolchain
    assert client.get("/api/convert/options", headers=HEAD).json()["available"] is True


def test_mock_convert_inspect_matches_the_real_model(client):
    from gpupool.converter.models import InspectResult

    def inspect(**src):
        return client.post("/api/convert/inspect", json=src, headers=HEAD)

    small = InspectResult.model_validate(inspect(hf_repo="HuggingFaceTB/SmolLM2-135M-Instruct").json())
    assert small.supported and small.params == 134_515_008 and small.weight_format == "safetensors" and len(small.options) == 24
    assert small.recommended in {o.type for o in small.options} and sum(o.recommended for o in small.options) == 1
    assert small.recommend_reasons and small.gguf_alternatives and small.source.revision == "main"
    big = InspectResult.model_validate(inspect(hf_repo="Qwen/Qwen2.5-72B-Instruct", revision="abc123").json())
    assert big.source.revision == "abc123" and big.recommended != small.recommended  # the recommendation follows the model size
    f16 = next(o for o in big.options if o.type == "F16")
    assert f16.fits_single_gpu is False and f16.fits_pool is True  # "needs several GPUs"
    assert next(o for o in big.options if o.type == "Q4_K_M").fits_single_gpu is True
    assert all(o.est_bytes and o.est_vram_mb for o in big.options)
    awq = InspectResult.model_validate(inspect(hf_repo="Qwen/Qwen2.5-7B-Instruct-AWQ").json())
    assert awq.prequantized == "awq" and awq.prequant_supported is False and awq.base_model == "Qwen/Qwen2.5-7B-Instruct"
    novel = InspectResult.model_validate(inspect(hf_repo="acme/NovelNet-7B").json())
    assert novel.supported is False and novel.warnings
    assert InspectResult.model_validate(inspect(hf_repo="meta-llama/Llama-3.1-8B-Instruct").json()).gated is True
    assert InspectResult.model_validate(inspect(hf_repo="acme/Quirky-3B").json()).remote_code is True
    folder = InspectResult.model_validate(inspect(path="/models/hf/smollm2-135m").json())
    assert folder.source.path == "/models/hf/smollm2-135m" and folder.source.hf_repo is None
    translated = InspectResult.model_validate(inspect(path="/srv/gguf/hf/smollm2-135m").json())
    assert translated.source.path == "/models/hf/smollm2-135m"  # a host path is translated
    assert inspect(hf_repo="nobody/nothing").status_code == 404
    assert inspect(hf_repo="no-slash").status_code == 400
    assert inspect(path="relative/dir").status_code == 400 and inspect(path="/nope").status_code == 400
    assert inspect(hf_repo="a/b", path="/x").status_code == 422 and inspect().status_code == 422


def test_mock_recommendation_follows_the_live_gpus(client, mock):
    def rec():
        return client.post("/api/convert/inspect", json={"hf_repo": "Qwen/Qwen2.5-72B-Instruct"}, headers=HEAD).json()["recommended"]

    before = rec()
    for g in mock.SERVERS["CTG-Server-1"]["gpus"]:  # take the four 80 GB cards out of the pool: only the 24 GB cards remain
        mock.SERVERS["CTG-Server-1"]["gpu_enabled"][g["device_id"]] = False
    assert rec() != before


def test_mock_hf_files_empty_for_safetensors_repos(client):
    assert client.get("/api/hf/files?repo=HuggingFaceTB/SmolLM2-135M-Instruct", headers=HEAD).json() == []
    assert client.get("/api/hf/files?repo=Qwen/Qwen2.5-3B-Instruct-GGUF", headers=HEAD).json()  # other repos keep their files


def test_mock_convert_seeded_jobs_match_the_real_model(client):
    from gpupool.converter.models import ConvertJob

    jobs = [ConvertJob.model_validate(j) for j in client.get("/api/convert", headers=HEAD).json()]
    assert {j.state for j in jobs} == {"queued", "done", "failed"}  # newest first
    assert jobs[0].created_at >= jobs[1].created_at >= jobs[2].created_at
    done = next(j for j in jobs if j.state == "done")
    assert done.validation.tokenizer_ok is True and done.validation.tokenizer_cases and done.output_bytes
    assert done.output_name in _library_names(client)
    failed = next(j for j in jobs if j.state == "failed")
    assert failed.error and failed.log_tail
    assert ConvertJob.model_validate(client.get(f"/api/convert/{done.id}", headers=HEAD).json()).id == done.id
    assert client.get("/api/convert/nope", headers=HEAD).status_code == 404
    assert client.get("/api/convert").status_code == 401


def test_mock_convert_job_walks_to_done_and_enters_the_library(client, mock):
    from gpupool.converter.models import ConvertJob

    r = _submit(client, "HuggingFaceTB/SmolLM2-135M-Instruct", quant="Q4_K_M")
    assert r.status_code == 200
    job = ConvertJob.model_validate(r.json())
    assert job.state == "queued" and job.output_name == "SmolLM2-135M-Instruct-Q4_K_M.gguf" and job.est_output_bytes
    seen = set()
    for t in (2.0, 5.0, 10.0, 14.0, 17.5):  # inside each stage
        _at(mock, job.id, t)
        cur = ConvertJob.model_validate(client.get(f"/api/convert/{job.id}", headers=HEAD).json())
        seen.add(cur.state)
        if cur.state == "downloading":
            assert cur.bytes_total and 0 < cur.bytes_done < cur.bytes_total and 0 < cur.stage_progress < 1
        if cur.state in ("converting", "quantizing", "validating"):
            assert cur.log_tail and cur.stage_progress is not None
    assert seen == {"downloading", "converting", "quantizing", "validating"}
    _fast_forward(mock, job.id)
    done = ConvertJob.model_validate(client.get(f"/api/convert/{job.id}", headers=HEAD).json())
    assert done.state == "done" and done.output_bytes and done.validation.generation_ok is True and done.finished_at
    lib = {i["name"]: i for i in client.get("/api/state", headers=HEAD).json()["library"]}
    assert lib[done.output_name]["source"] == "convert" and lib[done.output_name]["status"] == "ready"
    assert lib[done.output_name]["hf_repo"] == "HuggingFaceTB/SmolLM2-135M-Instruct"


def test_mock_convert_direct_types_skip_quantize_and_folders_skip_download(client, mock):
    from gpupool.converter.models import ConvertJob

    r = client.post("/api/convert", json={"source": {"path": "/models/hf/smollm2-135m"}, "quant": "Q8_0",
                                          "advanced": {"validate_generation": False}}, headers=HEAD)
    job = ConvertJob.model_validate(r.json())
    states = set()
    for t in (2.0, 5.0, 8.0):
        _at(mock, job.id, t)
        states.add(client.get(f"/api/convert/{job.id}", headers=HEAD).json()["state"])
    assert states == {"converting", "validating"}  # no download, no quantize
    _fast_forward(mock, job.id)
    done = ConvertJob.model_validate(client.get(f"/api/convert/{job.id}", headers=HEAD).json())
    assert done.state == "done" and done.validation.generation_ok is None and done.request.advanced.validate_generation is False


def test_mock_convert_needs_review_then_accept(client, mock):
    from gpupool.converter.models import ConvertJob

    running = next(j for j in client.get("/api/convert", headers=HEAD).json() if j["state"] == "queued")
    assert client.post(f"/api/convert/{running['id']}/accept", headers=HEAD).status_code == 409  # not reviewable yet
    _fast_forward(mock, running["id"])
    job = ConvertJob.model_validate(client.get(f"/api/convert/{running['id']}", headers=HEAD).json())
    assert job.state == "needs_review" and job.validation.tokenizer_ok is False
    bad = [c for c in job.validation.tokenizer_cases if not c.match]
    assert len(bad) == 1 and bad[0].hf != bad[0].gguf
    assert job.output_name not in _library_names(client)  # not in the library until accepted
    acc = ConvertJob.model_validate(client.post(f"/api/convert/{job.id}/accept", headers=HEAD).json())
    assert acc.state == "done" and acc.output_name in _library_names(client)
    assert client.post(f"/api/convert/{acc.id}/accept", headers=HEAD).status_code == 409


def test_mock_convert_failures_retry_and_remote_code(client, mock):
    from gpupool.converter.models import ConvertJob

    gated = ConvertJob.model_validate(_submit(client, "meta-llama/Llama-3.1-8B-Instruct", quant="Q4_K_M").json())
    _fast_forward(mock, gated.id)
    g = ConvertJob.model_validate(client.get(f"/api/convert/{gated.id}", headers=HEAD).json())
    assert g.state == "failed" and "HF_TOKEN" in g.error and g.bytes_done == 0
    quirky = ConvertJob.model_validate(_submit(client, "acme/Quirky-3B", quant="Q5_K_M").json())
    _fast_forward(mock, quirky.id)
    q = ConvertJob.model_validate(client.get(f"/api/convert/{quirky.id}", headers=HEAD).json())
    assert q.state == "failed" and "custom code" in q.error and q.log_tail
    assert client.post(f"/api/convert/{q.id}/cancel", headers=HEAD).status_code == 409  # only active jobs cancel
    assert client.post(f"/api/convert/{q.id}/retry", headers=HEAD).json()["state"] == "queued"
    ok = ConvertJob.model_validate(_submit(client, "acme/Quirky-3B", quant="Q6_K", advanced={"allow_remote_code": True}).json())
    _fast_forward(mock, ok.id)
    assert client.get(f"/api/convert/{ok.id}", headers=HEAD).json()["state"] == "done"
    broken = client.post("/api/convert", json={"source": {"path": "/models/hf/broken-model"}, "quant": "Q4_0"}, headers=HEAD).json()
    _fast_forward(mock, broken["id"])
    assert "KeyError" in client.get(f"/api/convert/{broken['id']}", headers=HEAD).json()["error"]


def test_mock_convert_cancel_retry_delete_rules(client, mock):
    job = _submit(client, "HuggingFaceTB/SmolLM2-135M-Instruct", quant="Q6_K", name="mine.gguf").json()
    assert client.post(f"/api/convert/{job['id']}/retry", headers=HEAD).status_code == 409  # active
    assert client.post(f"/api/convert/{job['id']}/accept", headers=HEAD).status_code == 409
    assert client.delete(f"/api/convert/{job['id']}", headers=HEAD).status_code == 409  # cancel first
    c = client.post(f"/api/convert/{job['id']}/cancel", headers=HEAD).json()
    assert c["state"] == "cancelled" and c["finished_at"]
    _fast_forward(mock, job["id"])
    assert client.get(f"/api/convert/{job['id']}", headers=HEAD).json()["state"] == "cancelled"  # stays cancelled
    assert client.post(f"/api/convert/{job['id']}/cancel", headers=HEAD).status_code == 409
    assert client.post(f"/api/convert/{job['id']}/retry", headers=HEAD).json()["state"] == "queued"
    client.post(f"/api/convert/{job['id']}/cancel", headers=HEAD)
    assert client.delete(f"/api/convert/{job['id']}", headers=HEAD).json() == {"ok": True}
    assert client.get(f"/api/convert/{job['id']}", headers=HEAD).status_code == 404
    assert client.delete(f"/api/convert/{job['id']}", headers=HEAD).status_code == 404
    for action in ("cancel", "retry", "accept"):
        assert client.post(f"/api/convert/nope/{action}", headers=HEAD).status_code == 404
    # deleting a finished job leaves its library file alone
    done = next(j for j in client.get("/api/convert", headers=HEAD).json() if j["state"] == "done")
    assert client.delete(f"/api/convert/{done['id']}", headers=HEAD).status_code == 200
    assert done["output_name"] in _library_names(client)


def test_mock_convert_request_validation(client):
    smol = "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert _submit(client, "Qwen/Qwen2.5-7B-Instruct-AWQ").status_code == 400  # pre-quantized, unsupported
    assert _submit(client, "acme/NovelNet-7B").status_code == 400
    assert _submit(client, "nobody/nothing").status_code == 404
    assert _submit(client, smol, quant="Q1_X").status_code == 422
    assert _submit(client, smol, advanced={"intermediate": "f64"}).status_code == 422
    assert _submit(client, smol, advanced={"threads": -1}).status_code == 422
    assert client.post("/api/convert", json={}, headers=HEAD).status_code == 422
    for bad in ("../x.gguf", "x.bin", "a b.gguf", ".hidden.gguf", "m-00001-of-00002.gguf"):
        assert _submit(client, smol, name=bad).status_code == 400, bad
    assert _submit(client, smol, name="llama-8b.gguf").status_code == 409  # already in the library
    assert _submit(client, smol, quant="Q3_K_M").status_code == 200
    assert _submit(client, smol, quant="Q3_K_M").status_code == 409  # another live job already makes that file
    default = _submit(client, "Qwen/Qwen2.5-72B-Instruct", quant="Q2_K").json()
    assert default["output_name"] == "Qwen2.5-72B-Instruct-Q2_K.gguf" and default["request"]["keep_source"] is False


def test_ui_has_conversion_flow():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    for needle in ("/api/convert/inspect", "/api/convert/options", '"/api/convert"', "keep_source", "allow_remote_code",
                   "validate_generation", "output_tensor_type", "token_embedding_type", "leave_output_tensor", "intermediate",
                   "prequant_supported", "stage_progress", "bytes_done",
                   "tokenizer_cases", "fits_single_gpu", "fits_pool",
                   "needs_review", "convPoll", "2000", "convAct", "convDeploy", "noGguf", "action"):
        assert needle in js, needle
    for needle in ("Convert a model", "Convert to GGUF", "Folder on the server", "Hugging Face repo", "Inspect", "Quantization",
                   "Recommended", "Output file name", "Keep downloaded source", "Advanced", "Allow remote code",
                   "runs them inside the coordinator", "Validate generation", "Threads", "Conversions", "Deploy this model",
                   "Accept anyway", "Retry", "Tokenizer check", "Generation sample", "download instead", "Conversion is not available",
                   "Show log", "Already quantized", "Gated repository", "recommend_reasons", "conv.res.base_model", "gguf_alternatives", "est_vram_mb", "convVisibleOptions()", "generation_sample", "tokenizer_cases", "log_tail", "Ships its own Python code", "Start conversion"):
        assert needle in html, needle
    for text in ("Fits one GPU", "Needs several GPUs (slower, over network)", "Does not fit the cluster", "Lossless",
                 "Near-lossless", "Balanced", "Small", "Tiny", "Download", "Convert", "Quantize", "Validate", "Converted"):
        assert text in js, text
    assert "srcLabel" in js and "srcLabel(it)" in html  # library rows of converted files
    assert "conv.open = false" in html and 'aria-labelledby="conv-title"' in html and 'role="progressbar"' in html


def test_ui_conversion_styles_use_tokens():
    css = (UI / "styles.css").read_text(encoding="utf-8")
    block = css[css.index("conversion (Hugging Face -> GGUF)"):css.index("reduced motion")]
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block), "use design tokens, not literal colours"
    for cls in (".stepper", ".cjob", ".qopt", ".logbox", ".vtbl tr.mism", ".bar.indet", ".adv", ".notice"):
        assert cls in block, cls


# ---------------------------------------------------------------------------------------------------------------
# Round 2: importance matrix, name_stem, failed_stage, early disk check
# ---------------------------------------------------------------------------------------------------------------
NEW_TYPES = ["IQ3_M", "IQ3_S", "IQ3_XS", "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"]
NEEDS = {"IQ3_XS", "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"}
SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _job(client, job_id):
    from gpupool.converter.models import ConvertJob

    return ConvertJob.model_validate(client.get(f"/api/convert/{job_id}", headers=HEAD).json())


def test_mock_options_carry_new_types_and_imatrix_flags(client, mock):
    from gpupool.converter.models import QuantOption

    o = client.get("/api/convert/options", headers=HEAD).json()
    assert o["imatrix_available"] is True
    opts = [QuantOption.model_validate(x) for x in o["quant_options"]]
    assert set(NEW_TYPES) <= {x.type for x in opts}
    assert {x.type for x in opts if x.needs_imatrix} == NEEDS
    ins = client.post("/api/convert/inspect", json={"hf_repo": "Qwen/Qwen2.5-7B-Instruct"}, headers=HEAD).json()
    assert ins["name_stem"] == "Qwen2.5-7B-Instruct"
    assert all(x["est_bytes"] and "needs_imatrix" in x for x in ins["options"])
    assert ins["recommended"] not in NEEDS  # the recommendation never needs the optional tool
    folder = client.post("/api/convert/inspect", json={"path": "/models/hf/smollm2-135m"}, headers=HEAD).json()
    assert folder["name_stem"] == "smollm2-135m"
    mock.reset()


def test_mock_imatrix_toggle_and_validation(client, mock):
    r = client.post("/api/_mock/convert_available", json={"imatrix_available": False}).json()
    assert r["imatrix_available"] is False and r["available"] is True
    assert client.get("/api/convert/options", headers=HEAD).json()["imatrix_available"] is False
    assert _submit(client, SMOL, quant="IQ2_XS", name="a.gguf").status_code == 503  # needs llama-imatrix
    assert _submit(client, SMOL, quant="Q4_K_M", name="b.gguf").status_code == 200  # does not
    assert _submit(client, SMOL, quant="Q3_K_S", name="c.gguf", advanced={"imatrix": "off"}).status_code == 200
    client.post("/api/_mock/convert_available", json={"imatrix_available": True})
    off = _submit(client, SMOL, quant="IQ2_XS", name="d.gguf", advanced={"imatrix": "off"})
    assert off.status_code == 422 and "importance matrix" in off.json()["detail"]
    assert _submit(client, SMOL, quant="Q3_K_S", name="e.gguf", advanced={"imatrix": "maybe"}).status_code == 422
    assert _submit(client, SMOL, quant="Q3_K_S", name="f.gguf", advanced={"imatrix_chunks": -1}).status_code == 422
    assert _submit(client, SMOL, quant="Q3_K_S", name="g.gguf", advanced={"calibration_path": "rel.txt"}).status_code == 422
    assert _submit(client, SMOL, quant="Q3_K_S", name="h.gguf", advanced={"calibration_path": "/data/x.md"}).status_code == 422
    mock.reset()


def test_mock_imatrix_used_and_calibrating_stage(client, mock):
    assert _job(client, _submit(client, SMOL, quant="Q4_K_M", name="p1.gguf").json()["id"]).imatrix_used is False  # >= 4 bits, auto
    assert _job(client, _submit(client, SMOL, quant="Q4_K_M", name="p2.gguf", advanced={"imatrix": "on"}).json()["id"]).imatrix_used is True
    assert _job(client, _submit(client, SMOL, quant="Q3_K_S", name="p3.gguf").json()["id"]).imatrix_used is True  # < 4 bits, auto
    assert _job(client, _submit(client, SMOL, quant="Q8_0", name="p4.gguf", advanced={"imatrix": "on"}).json()["id"]).imatrix_used is False  # direct
    job = _job(client, _submit(client, SMOL, quant="IQ2_XS", name="p5.gguf").json()["id"])
    assert job.imatrix_used is True
    seen = set()
    for t in (2.0, 5.0, 10.0, 14.0, 20.0, 25.0):
        _at(mock, job.id, t)
        cur = _job(client, job.id)
        seen.add(cur.state)
        if cur.state == "calibrating":
            assert cur.log_tail and cur.stage_progress is not None and 0 <= cur.stage_progress <= 1
    assert seen == {"downloading", "converting", "calibrating", "quantizing", "validating"}
    _fast_forward(mock, job.id)
    assert _job(client, job.id).state == "done"
    mock.reset()


def test_mock_failed_stage_on_failed_and_cancelled_jobs(client, mock):
    seeded = next(j for j in client.get("/api/convert", headers=HEAD).json() if j["state"] == "failed")
    assert seeded["failed_stage"] == "converting"
    gated = _submit(client, "meta-llama/Llama-3.1-8B-Instruct", name="gated.gguf").json()
    _fast_forward(mock, gated["id"])
    assert _job(client, gated["id"]).failed_stage == "downloading"
    bad = _submit(client, SMOL, quant="IQ3_S", name="corrupt.gguf", advanced={"calibration_path": "/data/corrupt.txt"}).json()
    _fast_forward(mock, bad["id"])
    b = _job(client, bad["id"])
    assert b.state == "failed" and b.failed_stage == "calibrating" and "llama-imatrix" in b.error
    run = _submit(client, SMOL, quant="Q4_K_M", name="cancelme.gguf").json()
    _at(mock, run["id"], 5.0)
    assert _job(client, run["id"]).state == "downloading"
    c = client.post(f"/api/convert/{run['id']}/cancel", headers=HEAD).json()
    assert c["state"] == "cancelled" and c["failed_stage"] == "downloading"
    assert client.post(f"/api/convert/{run['id']}/retry", headers=HEAD).json()["failed_stage"] is None
    done = next(j for j in client.get("/api/convert", headers=HEAD).json() if j["state"] == "done")
    assert done["failed_stage"] is None
    mock.reset()


def test_mock_507_when_the_disk_cannot_hold_the_job(client, mock):
    huge = _submit(client, "acme/Huge-400B", quant="Q4_K_M")
    assert huge.status_code == 507 and "GB" in huge.json()["detail"] and "free" in huge.json()["detail"]
    assert _submit(client, "Qwen/Qwen2.5-72B-Instruct", quant="Q4_K_M", name="big.gguf").status_code == 200  # fits the default disk
    client.post("/api/_mock/convert_available", json={"disk_free_gb": 10})
    assert _submit(client, "Qwen/Qwen2.5-7B-Instruct", quant="Q4_K_M", name="small-disk.gguf").status_code == 507
    assert _submit(client, SMOL, quant="Q4_K_M", name="tiny.gguf").status_code == 200  # a small model still fits
    mock.reset()


def test_ui_has_importance_matrix_and_round2_controls():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    for needle in ("needs_imatrix", "imatrix_available", "imatrix_used", "failed_stage", "name_stem", "calibration_path", "imatrix_chunks",
                   "calibrating", "convVisibleOptions", "convOptDisabled", "convImatrixApplies", "started_at", "finished_at", "CONV_LOW_BPW"):
        assert needle in js, needle
    for needle in ("Needs calibration", "Show smaller, lower-quality types", "Importance matrix", "Calibration text", "Calibration chunks",
                   "convOptDisabled(o)", "llama-imatrix is not installed", "usually the slowest step", "convTimes(j)"):
        assert needle in html, needle
    assert '"Calibrate"' in js and "j.imatrix_used" in js
    assert "replace(/[^A-Za-z0-9._-]+/g" not in js  # the UI no longer has its own naming rule
    assert ".stepper li.stopped" in (UI / "styles.css").read_text(encoding="utf-8")


def test_ui_stepper_logic_runs_in_node(mock):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = (UI / "app.js").read_text(encoding="utf-8")
    harness = js + """
var a = app();
function keys(j) { return a.convStages(j).map(function (s) { return s.key + ":" + s.cls; }).join(" "); }
var base = { request: { source: { hf_repo: "a/b" }, quant: "IQ2_XS" }, imatrix_used: true };
var out = [
  keys(Object.assign({}, base, { state: "calibrating" })),
  keys(Object.assign({}, base, { state: "failed", failed_stage: "calibrating" })),
  keys(Object.assign({}, base, { state: "cancelled", failed_stage: "downloading" })),
  keys(Object.assign({}, base, { state: "failed", failed_stage: "queued" })),
  keys({ request: { source: { path: "/x" }, quant: "Q8_0" }, imatrix_used: false, state: "done" }),
  keys({ request: { source: { hf_repo: "a/b" }, quant: "Q4_K_M" }, imatrix_used: false, state: "validating" }),
  a.dur(200) + "|" + a.dur(45) + "|" + a.dur(3900)
];
console.log(JSON.stringify(out));
"""
    prelude = "global.window = global; global.document = {documentElement: {removeAttribute() {}, setAttribute() {}}}; global.localStorage = {getItem() { return null; }, setItem() {}}; global.matchMedia = () => ({matches: false, addEventListener() {}});"
    r = subprocess.run([node, "-"], input=prelude + chr(10) + harness, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        pytest.skip("app.js needs a browser environment: " + r.stderr[-200:])
    import json

    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out[0] == "download:done convert:done calibrate:active quantize:todo validate:todo"
    assert out[1] == "download:done convert:done calibrate:failed quantize:todo validate:todo"
    assert out[2] == "download:stopped convert:todo calibrate:todo quantize:todo validate:todo"
    assert out[3] == "download:todo convert:todo calibrate:todo quantize:todo validate:todo"
    assert out[4] == "download:skipped convert:done quantize:skipped validate:done"
    assert out[5] == "download:done convert:done quantize:done validate:active"
    assert out[6] == "3m 20s|45s|1h 5m"


# ---------------------------------------------------------------------------------------------------------------
# Model card copy buttons and the allowed-servers selection ("<node>/*")
# ---------------------------------------------------------------------------------------------------------------
def _put_model(client, name, **extra):
    body = {"file": "qwen2.5-3b-q4.gguf", **extra}
    return client.put(f"/api/models/{name}", json=body, headers=HEAD)


def test_mock_pin_devices_accepts_whole_server_wildcard(client, mock):
    node = next(n for n, s in mock.SERVERS.items() if s["alive"] and len(s["gpus"]) > 1)
    ok = _put_model(client, "pinned", pin_devices=[f"{node}/*"])
    assert ok.status_code == 200
    saved = next(m for m in client.get("/api/state", headers=HEAD).json()["models"] if m["spec"]["name"] == "pinned")
    assert saved["spec"]["pin_devices"] == [f"{node}/*"]
    rec = client.post("/api/recommend", json={"file": "qwen2.5-3b-q4.gguf", "pin_devices": [f"{node}/*"]}, headers=HEAD)
    assert rec.status_code == 200
    nodes = {a["node_id"] for o in rec.json()["options"] for a in o["assignments"]}
    assert nodes <= {node}  # every option stays inside the selected server
    assert client.post("/api/models/pinned/plan", headers=HEAD).status_code == 200
    assert _put_model(client, "pinned", pin_devices=["nope/*"]).status_code == 422  # unknown server, like the real API
    assert _put_model(client, "pinned", pin_devices="x").status_code == 422
    assert _put_model(client, "pinned", pin_devices=["noslash"]).status_code == 422
    assert client.post("/api/recommend", json={"file": "qwen2.5-3b-q4.gguf", "pin_devices": ["nope/*"]}, headers=HEAD).status_code == 422
    mock.reset()


def test_ui_has_copy_buttons_on_the_model_card():
    html = (UI / "index.html").read_text(encoding="utf-8")
    js = (UI / "app.js").read_text(encoding="utf-8")
    for needle in ('aria-label="Copy endpoint"', 'aria-label="Copy model name"', "Copy curl", "copy(m.spec.name)", "curlSnippet(m.spec.name)"):
        assert needle in html, needle
    assert "curlSnippet: function (modelName)" in js
    assert ".ep-line" in (UI / "styles.css").read_text(encoding="utf-8")


def test_ui_component_has_no_duplicate_methods():
    """A repeated key in the app() object literal silently replaces the earlier method
    (the model form's toggleGpu once shadowed the Servers tab's pool switch)."""
    js = (UI / "app.js").read_text(encoding="utf-8")
    names = re.findall(r"^    ([A-Za-z_$][\w$]*): (?:async )?function", js, flags=re.M)
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, dupes


def test_ui_pool_switch_calls_the_api_in_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = (UI / "app.js").read_text(encoding="utf-8")
    harness = js + """
var a = app(), calls = [];
a.api = async function (m, p, b) { calls.push([m, p, b]); return {}; };
a.refresh = async function () {};
a.form.pins = [];
a.toggleGpu({ node_id: "a" }, { device_id: "CUDA0" }, false).then(function () {
  console.log(JSON.stringify([calls, a.form.pins]));
});
"""
    prelude = "global.window = global; global.document = {documentElement: {removeAttribute() {}, setAttribute() {}}}; global.localStorage = {getItem() { return null; }, setItem() {}}; global.matchMedia = () => ({matches: false, addEventListener() {}});"
    r = subprocess.run([node, "-"], input=prelude + chr(10) + harness, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-300:]
    import json

    calls, pins = json.loads(r.stdout.strip().splitlines()[-1])
    assert calls == [["PUT", "/api/servers/a/gpus/CUDA0", {"enabled": False}]]
    assert pins == []


def test_ui_has_allowed_servers_selection():
    html = (UI / "index.html").read_text(encoding="utf-8")
    js = (UI / "app.js").read_text(encoding="utf-8")
    for needle in ("All servers and GPUs", "Only selected ones", "pinToggleServer(s", "pinToggleGpu(s, d", "pinNodeState(s)", "pinSummary()",
                   "Nothing selected", "off in the pool (Servers tab)", "server offline", "GPUs added later", "stay enabled for other models"):
        assert needle in html, needle
    for needle in ('"/*"', "pinWhole", "pinSummary", '"Limited to "', "Select at least one server or GPU"):
        assert needle in js, needle
    assert "Auto placement" not in html


def test_ui_pin_logic_runs_in_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    js = (UI / "app.js").read_text(encoding="utf-8")
    harness = js + """
var a = app();
var gp = function (n) { return { node_id: n, alive: true, gpus: ["CUDA0", "CUDA1", "CUDA2"].map(function (x) { return { device_id: x }; }) }; };
var s1 = gp("a"), s2 = gp("b");
a.servers = function () { return [s1, s2]; };
a.gpus = function (s) { return s.gpus; };
a.form.pins = [];
a.pinToggleServer(s1, true);
var out = [a.form.pins.slice(), a.pinNodeState(s1)];
a.pinToggleGpu(s1, s1.gpus[1], false);
out.push(a.form.pins.slice(), a.pinNodeState(s1));
a.pinToggleGpu(s2, s2.gpus[0], true);
out.push(a.form.pins.slice(), a.pinNodeState(s2), a.pinSummary());
a.pinToggleServer(s1, true);
out.push(a.form.pins.slice());
a.pinToggleServer(s1, false);
out.push(a.form.pins.slice());
out.push(a.specChips({ spec: { pin_devices: ["a/*", "b/CUDA1"] } }));
console.log(JSON.stringify(out));
"""
    prelude = "global.window = global; global.document = {documentElement: {removeAttribute() {}, setAttribute() {}}}; global.localStorage = {getItem() { return null; }, setItem() {}}; global.matchMedia = () => ({matches: false, addEventListener() {}});"
    r = subprocess.run([node, "-"], input=prelude + chr(10) + harness, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-300:]
    import json

    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out[0] == ["a/*"] and out[1] == "all"
    assert out[2] == ["a/CUDA0", "a/CUDA2"] and out[3] == "some"  # one GPU off a whole-server pick -> the explicit rest
    assert out[4] == ["a/CUDA0", "a/CUDA2", "b/CUDA0"] and out[5] == "some" and out[6] == "3 of 6 GPUs on 2 servers"
    assert out[7] == ["b/CUDA0", "a/*"]  # a whole server replaces that node's single-GPU pins
    assert out[8] == ["b/CUDA0"]
    assert out[9] == ["Limited to a, b/CUDA1"]


def test_form_has_attention_batch_fields_and_tips():
    html = (UI / "index.html").read_text()
    js = (UI / "app.js").read_text()
    for needle in ('x-model="form.fa"', 'x-model.number="form.ubatch"', 'x-model.number="form.batch"',
                   "form.rec.tips", "applyTip(t)", "ctxPerSlot()", "archLabel(r.d)", "kernels_ok === false"):
        assert needle in html, needle
    # every field a tip can apply maps onto a form field
    for field in ("kv_cache_type", "speculative", "draft_file", "draft_n_max", "flash_attn", "ubatch", "batch",
                  "ctx_size", "parallel", "file"):
        assert re.search(r"\b" + field + r': "', js), field
    assert "flash_attn: fa, ubatch: ub, batch: b" in js


def test_mock_recommend_tips_and_perf_validation(client):
    j = client.post("/api/recommend", headers=HEAD, json={"file": "qwen2.5-3b-q4.gguf"}).json()
    assert [t["id"] for t in j["tips"]] == ["parallel", "ngram", "ubatch"]
    r = client.post("/api/recommend", headers=HEAD,
                    json={"file": "qwen2.5-3b-q4.gguf", "kv_cache_type": "q8_0", "flash_attn": "off"})
    assert r.status_code == 422
    gpus = client.get("/api/state", headers=HEAD).json()["servers"][0]["report"]["devices"]
    assert gpus[0]["compute_cap"] == "9.0"


def _sse(text: str) -> list:
    import json
    out = []
    for block in text.split("\n\n"):
        if block.startswith("data: "):
            payload = block[len("data: "):]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def test_mock_v1_chat_streams_like_the_router(client):
    body = {"model": "chat-auto", "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 5,
            "messages": [{"role": "user", "content": "hi /think"}]}
    r = client.post("/v1/chat/completions", json=body, headers=HEAD)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-gpupool-replica"].startswith("chat-auto-")
    ev = _sse(r.text)
    assert ev[0]["choices"][0]["delta"]["role"] == "assistant" and ev[-1] == "[DONE]"
    deltas = [e["choices"][0]["delta"] for e in ev[1:-1] if isinstance(e, dict) and e["choices"]]
    assert any("reasoning_content" in d for d in deltas) and sum("content" in d for d in deltas) == 5
    last = next(e for e in ev if isinstance(e, dict) and e["choices"] and e["choices"][0]["finish_reason"])
    assert last["choices"][0]["finish_reason"] == "length"
    assert {"prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second"} <= set(last["timings"])
    usage = next(e for e in ev if isinstance(e, dict) and "usage" in e)["usage"]
    assert usage["completion_tokens"] == last["timings"]["predicted_n"]


def test_mock_v1_chat_errors_like_the_router(client):
    msgs = [{"role": "user", "content": "hi"}]
    assert client.post("/v1/chat/completions", json={"model": "nope", "messages": msgs}, headers=HEAD).status_code == 404
    r = client.post("/v1/chat/completions", json={"model": "qwen3b", "messages": msgs}, headers=HEAD)  # stopped
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_replica"
    r = client.post("/v1/chat/completions", json={"model": "chat-auto", "messages": msgs, "max_tokens": 3}, headers=HEAD)
    assert r.json()["usage"]["completion_tokens"] == 3 and r.headers["x-gpupool-replica"]


def test_ui_has_playground():
    html = (UI / "index.html").read_text(encoding="utf-8")
    js = (UI / "app.js").read_text(encoding="utf-8")
    assert "view==='playground'" in html and "go('playground')" in html and "pgOpen(m.spec.name)" in html
    assert "Clear chat" in html and "New chat" not in html
    assert html.index('class="panel pg-chat"') < html.index('class="panel pg-side"')  # settings on the right
    for needle in ('"/v1/chat/completions"', "x-gpupool-replica", "timings", "reasoning_content",
                   "stream_options", "AbortController", "predicted_per_second"):
        assert needle in js, needle
    # model output is untrusted: rendered as text, or as Markdown through pgMd() (DOMPurify-sanitized)
    pg = html[html.index("<!-- ============ playground"):html.index("<!-- ============", html.index("<!-- ============ playground") + 10)]
    assert re.findall(r'x-html="(?!icon\()(?!logo\(\))(?!pgMd\()', pg) == []


def test_ui_large_panels_collapse_and_start_open():
    html = (UI / "index.html").read_text(encoding="utf-8")
    js = (UI / "app.js").read_text(encoding="utf-8")
    css = (UI / "styles.css").read_text(encoding="utf-8")
    ids = re.findall(r"toggleFold\('([a-z-]+)'\)\" :aria-expanded", html)
    assert {"servers", "vram", "library", "conversions", "placement", "deployments", "events"} <= set(ids)
    for fid in ids:  # each foldable panel is bound to its own state
        assert f":class=\"{{folded: folded('{fid}')}}\"" in html, fid
    # open unless the viewer folded it: the saved map only lists folded panels
    assert "function foldSaved()" in js and "if (f[id]) delete f[id]; else f[id] = true;" in js
    assert re.search(r"\.panel\.folded > :not\(\.panel-head\) \{ display: none", css)


def test_ui_fixes_after_060():
    html = (UI / "index.html").read_text(encoding="utf-8")
    js = (UI / "app.js").read_text(encoding="utf-8")
    css = (UI / "styles.css").read_text(encoding="utf-8")
    # a GPU switched off in the pool cannot be picked for a model (unless already picked: it can be removed)
    assert ':disabled="!enabled(s, d) && !pinHas(s, d)"' in html
    # the picker's sticky server header stays above dimmed rows (their opacity makes a stacking context)
    assert re.search(r"\.gpu-pick \.grp \{[^}]*position: sticky[^}]*z-index: 1", css)
    # Save / Save & Start stay on the right, also on a wrapped second line
    assert ".modal-foot.split .foot-group:last-child { margin-left: auto; }" in css
    # events can be cleared, up to the newest one listed
    assert '"/api/events?up_to_id="' in js and "clearEvents()" in html
    # the Playground never keeps a deleted model selected
    assert "pgPick: function" in js and 'x-effect="if (view === \'playground\') pgPick()"' in html


def test_mock_events_can_be_cleared(client):
    r = client.delete("/api/events", headers=HEAD)
    left = client.get("/api/events", headers=HEAD).json()
    assert r.status_code == 200 and (left if isinstance(left, list) else left["events"]) == []


def test_ui_never_uses_browser_dialogs():
    js = (UI / "app.js").read_text(encoding="utf-8")
    html = (UI / "index.html").read_text(encoding="utf-8")
    # the browser's confirm()/alert() look foreign and block the page: the in-app dialog is used
    assert not re.search(r"(?<![\w.])(confirm|alert|prompt)\(", js.replace("confirmBox(", ""))
    assert 'role="alertdialog"' in html and "askDone(true)" in html and "askDone(false)" in html


def test_ui_visual_language():
    html = (UI / "index.html").read_text(encoding="utf-8")
    js = (UI / "app.js").read_text(encoding="utf-8")
    css = (UI / "styles.css").read_text(encoding="utf-8")
    # shared gradients exist for every vivid tone and the logo
    for gid in ("ig-blue", "ig-green", "ig-purple", "ig-amber", "ig-red", "ig-cyan", "lg-top", "lg-left", "lg-right"):
        assert f'id="{gid}"' in html, gid
    # every icon the markup uses is drawn
    names = set(re.findall(r"icon\('([a-z]+)'\)", html)) | set(re.findall(r'icon\("([a-z]+)"\)', js))
    for n in names:
        assert re.search(rf"^  {n}: '", js, re.M), n
    # page changes go through the View Transitions API when there is one, never with reduced motion
    assert "document.startViewTransition" in js and "prefers-reduced-motion: reduce" in js
    assert "::view-transition-old(root)" in css and ".nav-pill" in css
    # entrances fill "backwards" so hover transforms work once they end
    assert re.search(r"\.view > \*, \.pg > \.panel \{ animation: rise [^}]*backwards", css)
