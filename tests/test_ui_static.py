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
