"""One-command deployment: generated secrets, join strings, identity defaults, auto-join."""
from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from gpupool.agent.app import join_coordinator
from gpupool.common import config as config_mod
from gpupool.common.config import (
    AgentConfig, CoordinatorConfig, build_agent_config, env_overrides, load_or_create_secrets,
    parse_join, sanitize_node_id,
)
from gpupool.coordinator.agent_client import AgentError
from gpupool.coordinator.app import create_app, startup_banner
from gpupool.coordinator.store import ServerRecord, Store
from tests.test_coordinator_helpers import FakeClient, make_cfg, node

CT = {"Authorization": "Bearer ctok"}
AD = {"Authorization": "Bearer adm"}


# ---------------------------------------------------------------- secrets

def test_secrets_generated_reused_and_env_wins(tmp_path):
    db = tmp_path / "data" / "coordinator.db"
    a1, c1 = load_or_create_secrets(db, "", "")
    assert len(a1) >= 20 and len(c1) >= 20 and a1 != c1
    assert json.loads((tmp_path / "data" / "secrets.json").read_text()) == {
        "admin_key": a1, "cluster_token": c1}
    assert not list((tmp_path / "data").glob("*.tmp"))
    assert load_or_create_secrets(db, "", "") == (a1, c1)  # reused, not regenerated
    a2, c2 = load_or_create_secrets(db, "mine", "")  # explicit value wins, other still from file
    assert (a2, c2) == ("mine", c1)
    assert load_or_create_secrets(db, "x", "y") == ("x", "y")


def test_secrets_corrupt_file_is_not_overwritten(tmp_path):
    p = tmp_path / "secrets.json"
    p.write_text("{not json")
    with pytest.raises(RuntimeError):
        load_or_create_secrets(tmp_path / "c.db", "", "")
    assert p.read_text() == "{not json"


def test_banner_contains_join_and_key():
    cfg = CoordinatorConfig(admin_key="AK", cluster_token="CT", port=8098)
    text = "\n".join(startup_banner(cfg, lan_ip="10.0.0.1"))
    assert "http://10.0.0.1:8098" in text and "AK" in text
    assert "GPUPOOL_JOIN='http://10.0.0.1:8098#CT'" in text
    assert "--join 'http://10.0.0.1:8098#CT'" in text
    assert "not reachable" in text
    pub = "\n".join(startup_banner(CoordinatorConfig(admin_key="a", cluster_token="t",
                                                      public_url="http://pub:1/")))
    assert "http://pub:1#t'" in pub and "not reachable" not in pub


# ---------------------------------------------------------------- join string / identity

def test_parse_join():
    assert parse_join("http://10.0.0.1:8080#tok") == ("http://10.0.0.1:8080", "tok")
    assert parse_join(" http://h:1/#t ") == ("http://h:1", "t")
    assert parse_join("http://h:1") == ("http://h:1", "")


def test_join_from_env_and_explicit_wins(tmp_path):
    env = {"GPUPOOL_JOIN": "http://c:8080#tok", "GPUPOOL_LLAMA_DIR": str(tmp_path),
           "GPUPOOL_NODE_ID": "n", "GPUPOOL_HOST": "10.1.1.1"}
    cfg = build_agent_config(env_overrides(AgentConfig, env))
    assert cfg.coordinator_url == "http://c:8080" and cfg.cluster_token == "tok"
    cfg = build_agent_config({**env_overrides(AgentConfig, env), "cluster_token": "explicit",
                              "coordinator_url": "http://other:1"})
    assert cfg.coordinator_url == "http://other:1" and cfg.cluster_token == "explicit"


def test_identity_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod.socket, "gethostname", lambda: "GPU Box_01.Lab")
    seen = {}

    def fake_ip(host, port):
        seen["target"] = (host, port)
        return "10.9.9.9"

    monkeypatch.setattr(config_mod, "detect_local_ip", fake_ip)
    cfg = build_agent_config({"llama_dir": tmp_path, "join": "http://10.0.0.1:8080#t"})
    assert cfg.node_id == "gpu-box_01.lab" and cfg.host == "10.9.9.9"
    assert seen["target"] == ("10.0.0.1", 8080)
    cfg = build_agent_config({"llama_dir": tmp_path, "node_id": "Keep", "host": "1.2.3.4"})
    assert (cfg.node_id, cfg.host) == ("Keep", "1.2.3.4")  # explicit values untouched
    assert sanitize_node_id("  ") == "gpu-node"


def test_detect_local_ip_falls_back(monkeypatch):
    class Boom:
        def __init__(self, *a):
            raise OSError("no network")

    monkeypatch.setattr(config_mod.socket, "socket", Boom)
    assert config_mod.detect_local_ip("10.0.0.1", 80) == "127.0.0.1"


# ---------------------------------------------------------------- agent auto-join

def agent_cfg(tmp_path):
    return AgentConfig(node_id="n1", host="10.0.0.5", port=7099, llama_dir=tmp_path,
                       coordinator_url="http://coord:8080", cluster_token="tok")


class Sleeps:
    def __init__(self):
        self.calls = []

    async def __call__(self, s):
        self.calls.append(s)


@pytest.fixture
def mock():
    with respx.mock(assert_all_called=False) as m:
        m.get("http://127.0.0.1:7099/health").respond(200, json={"ok": True})
        yield m


async def test_join_retries_then_succeeds(tmp_path, mock, caplog):
    route = mock.post("http://coord:8080/internal/join")
    route.side_effect = [httpx.ConnectError("down"), httpx.Response(502, text="x"),
                         httpx.Response(200, json={"node_id": "n1"})]
    sl = Sleeps()
    with caplog.at_level(logging.INFO):
        assert await join_coordinator(agent_cfg(tmp_path), sleep=sl) is True
    assert route.call_count == 3
    assert [c for c in sl.calls if c >= 2] == [2.0, 4.0]  # backoff doubles
    req = route.calls[-1].request
    assert req.headers["authorization"] == "Bearer tok"
    assert json.loads(req.content) == {"agent_url": "http://10.0.0.5:7099"}
    assert "joined as n1" in caplog.text


async def test_join_backoff_caps_at_60(tmp_path, mock):
    route = mock.post("http://coord:8080/internal/join")
    route.side_effect = [httpx.ConnectError("down")] * 8 + [httpx.Response(200, json={})]
    sl = Sleeps()
    assert await join_coordinator(agent_cfg(tmp_path), sleep=sl) is True
    assert max(sl.calls) == 60.0


@pytest.mark.parametrize("status,needle", [(403, "removed in the UI"), (401, "wrong cluster token")])
async def test_join_stops_on_refusal(tmp_path, mock, caplog, status, needle):
    route = mock.post("http://coord:8080/internal/join").respond(status)
    with caplog.at_level(logging.ERROR):
        assert await join_coordinator(agent_cfg(tmp_path), sleep=Sleeps()) is False
    assert route.call_count == 1
    assert needle in caplog.text


# ---------------------------------------------------------------- coordinator /internal/join

@pytest.fixture
async def env(tmp_path):
    cfg = make_cfg(tmp_path, cluster_token="ctok", admin_key="adm")
    cfg.models_dir.mkdir()
    store = Store(":memory:")
    app = create_app(cfg, store=store, client=FakeClient(), start_background=False)
    reports = {}

    async def probe(url):
        r = reports[url]
        if isinstance(r, Exception):
            raise r
        return r

    app.state.poller.probe = probe
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c, store, reports


async def test_join_registers_new_server(env):
    c, store, reports = env
    reports["http://10.0.0.7:7070"] = node("gpu7", host="10.0.0.7")
    r = await c.post("/internal/join", json={"agent_url": "http://10.0.0.7:7070/"}, headers=CT)
    assert r.status_code == 200 and r.json()["node_id"] == "gpu7" and r.json()["new"] is True
    assert store.get_server("gpu7").agent_url == "http://10.0.0.7:7070"
    assert [n.report.node_id for n in store.list_nodes()] == ["gpu7"]
    ev = store.list_events()
    assert ev[0].kind == "server_added" and "joined from http://10.0.0.7:7070" in ev[0].message
    st = (await c.get("/api/state", headers=AD)).json()
    assert st["servers"][0]["node_id"] == "gpu7"


async def test_join_updates_url_of_known_server(env):
    c, store, reports = env
    store.add_server(ServerRecord(node_id="gpu7", agent_url="http://old:7070", added_at=5.0))
    reports["http://new:7070"] = node("gpu7", host="new")
    r = await c.post("/internal/join", json={"agent_url": "http://new:7070"}, headers=CT)
    assert r.status_code == 200 and r.json()["new"] is False
    rec = store.get_server("gpu7")
    assert rec.agent_url == "http://new:7070" and rec.added_at == 5.0
    assert store.list_events() == []  # not a new server


async def test_join_probe_failure_is_502(env):
    c, store, reports = env
    reports["http://x:1"] = httpx.ConnectError("refused")
    reports["http://y:1"] = AgentError(401, "bad token", "http://y:1")
    r = await c.post("/internal/join", json={"agent_url": "http://x:1"}, headers=CT)
    assert r.status_code == 502 and "refused" in r.json()["detail"]
    r = await c.post("/internal/join", json={"agent_url": "http://y:1"}, headers=CT)
    assert r.status_code == 502
    assert store.list_servers() == []


async def test_join_auth_and_bad_url(env):
    c, *_ = env
    body = {"agent_url": "http://x:1"}
    assert (await c.post("/internal/join", json=body)).status_code == 401
    assert (await c.post("/internal/join", json=body, headers=AD)).status_code == 401
    assert (await c.post("/internal/join", json={"agent_url": "nope"}, headers=CT)).status_code == 400


async def test_removal_sticks_until_manual_add(env):
    c, store, reports = env
    reports["http://10.0.0.7:7070"] = node("gpu7", host="10.0.0.7")
    body = {"agent_url": "http://10.0.0.7:7070"}
    assert (await c.post("/internal/join", json=body, headers=CT)).status_code == 200
    assert (await c.delete("/api/servers/gpu7", headers=AD)).status_code == 200
    assert store.is_removed("gpu7") and store.get_server("gpu7") is None
    r = await c.post("/internal/join", json=body, headers=CT)
    assert r.status_code == 403 and store.get_server("gpu7") is None
    # the admin re-adds it by hand: removal is cleared and auto-join works again
    assert (await c.post("/api/servers", json=body, headers=AD)).status_code == 200
    assert not store.is_removed("gpu7")
    assert (await c.post("/internal/join", json=body, headers=CT)).status_code == 200
