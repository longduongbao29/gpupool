"""gpupool command line: config precedence and the admin commands' requests."""
import json

import httpx
import pytest
import respx

from gpupool import cli


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("GPUPOOL_URL", "GPUPOOL_ADMIN_KEY", "GPUPOOL_PORT", "GPUPOOL_NODE_ID", "GPUPOOL_HOST",
              "GPUPOOL_JOIN", "GPUPOOL_LLAMA_DIR", "GPUPOOL_COORDINATOR_URL", "GPUPOOL_CLUSTER_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def test_agent_flags_beat_env_beat_toml(tmp_path, monkeypatch):
    import gpupool.agent.app as agent_app
    seen = {}
    monkeypatch.setattr(agent_app, "run_agent", lambda cfg: seen.setdefault("cfg", cfg))
    toml = tmp_path / "agent.toml"
    toml.write_text('node_id = "from-toml"\nport = 7001\nhost = "10.0.0.7"\nllama_dir = "/toml"\n'
                    'coordinator_url = "http://c:8080"\n')
    monkeypatch.setenv("GPUPOOL_PORT", "7002")
    monkeypatch.setenv("GPUPOOL_NODE_ID", "from-env")
    assert cli.main(["agent", "--config", str(toml), "--node-id", "from-flag", "--no-auto-join"]) == 0
    cfg = seen["cfg"]
    assert (cfg.node_id, cfg.port, cfg.host, str(cfg.llama_dir)) == ("from-flag", 7002, "10.0.0.7", "/toml")
    assert cfg.auto_join is False


def test_coordinator_flags_beat_env(tmp_path, monkeypatch):
    import gpupool.coordinator.app as coord_app
    seen = {}
    monkeypatch.setattr(coord_app, "run_coordinator", lambda cfg: seen.setdefault("cfg", cfg))
    monkeypatch.setenv("GPUPOOL_PORT", "9090")
    assert cli.main(["coordinator", "--host", "127.0.0.2"]) == 0
    assert (seen["cfg"].host, seen["cfg"].port) == ("127.0.0.2", 9090)


@respx.mock
@pytest.mark.parametrize("argv, method, path, params, body", [
    (["register", "m", "coordinator://x.gguf", "--ctx", "8192", "--replicas", "2"], "POST", "/admin/models", {},
     {"name": "m", "source": "coordinator://x.gguf", "ctx_size": 8192, "parallel": 1, "replicas": 2}),
    (["plan", "m"], "POST", "/admin/deploy/m", {"dry_run": "1"}, None),
    (["scale", "m", "3"], "POST", "/admin/models/m/scale", {"replicas": "3"}, None),
    (["undeploy", "m"], "DELETE", "/admin/models/m", {}, None),
    (["status"], "GET", "/admin/status", {}, None),
])
def test_admin_commands(argv, method, path, params, body, monkeypatch, capsys):
    monkeypatch.setenv("GPUPOOL_URL", "http://coord:8080/")
    monkeypatch.setenv("GPUPOOL_ADMIN_KEY", "k")
    route = respx.request(method, f"http://coord:8080{path}").mock(
        return_value=httpx.Response(200, json={"ok": True}))
    assert cli.main(argv) == 0
    req = route.calls.last.request
    assert req.headers["authorization"] == "Bearer k"
    assert dict(req.url.params) == params
    if body is not None:
        assert json.loads(req.content) == body
    assert json.loads(capsys.readouterr().out) == {"ok": True}


@respx.mock
def test_admin_error_status_and_plain_text(monkeypatch, capsys):
    respx.get("http://127.0.0.1:8080/admin/status").mock(return_value=httpx.Response(401, text="nope"))
    assert cli.main(["status"]) == 1
    out = capsys.readouterr().out
    assert out.strip() == "nope"
    assert respx.calls.last.request.headers.get("authorization") is None  # no key, no header


@respx.mock
def test_admin_unreachable(capsys):
    respx.get("http://127.0.0.1:8080/admin/status").mock(side_effect=httpx.ConnectError("refused"))
    assert cli.main(["status"]) == 2
    assert "cannot reach coordinator" in capsys.readouterr().err
