import httpx
import pytest

from gpupool.agent import models_cache
from gpupool.common import net
from gpupool.common.net import (describe_proxy, external_client, internal_client, normalize_proxy_env,
                                proxy_env, proxy_for)

KEYS = ["http_proxy", "https_proxy", "all_proxy", "no_proxy"]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv(k.upper(), raising=False)


def test_precedence_and_empty():
    env = {"http_proxy": "http://lower:1", "HTTP_PROXY": "http://UPPER:1",
           "HTTPS_PROXY": "http://up:2", "no_proxy": "", "NO_PROXY": "a.com", "all_proxy": " "}
    assert proxy_env(env) == {"http_proxy": "http://lower:1", "https_proxy": "http://up:2",
                              "no_proxy": "a.com"}


def test_normalize_sets_both_cases():
    env = {"HTTPS_PROXY": "http://p:3", "http_proxy": "http://q:1"}
    normalize_proxy_env(env)
    assert env["https_proxy"] == env["HTTPS_PROXY"] == "http://p:3"
    assert env["http_proxy"] == env["HTTP_PROXY"] == "http://q:1"
    assert "no_proxy" not in env and "NO_PROXY" not in env


@pytest.mark.parametrize("url,proxied", [
    ("https://huggingface.co/x", True),
    ("http://10.1.2.3:7070/report", False),       # CIDR
    ("http://192.168.0.5/", False),                # exact IP
    ("http://svc.corp.local/", False),             # .suffix
    ("http://corp.local/", False),                 # bare suffix matches the domain itself
    ("http://exact.host/", False),
    ("http://notcorp.local/", True),               # not a label boundary
    ("http://11.0.0.1/", True),
])
def test_no_proxy_matching(url, proxied):
    env = {"http_proxy": "http://p:3128", "https_proxy": "http://p:3128",
           "no_proxy": "10.0.0.0/8, 192.168.0.5,.corp.local,exact.host"}
    assert (proxy_for(url, env) is not None) == proxied


def test_no_proxy_star_and_scheme():
    assert proxy_for("https://a.com", {"https_proxy": "http://p:1", "no_proxy": "*"}) is None
    assert proxy_for("http://a.com", {"https_proxy": "http://p:1"}) is None  # no http proxy set
    assert proxy_for("http://a.com", {"all_proxy": "http://p:1"}) == "http://p:1"


async def test_external_routes_internal_does_not(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://p:3128")
    monkeypatch.setenv("no_proxy", "10.0.0.0/8")
    ext = external_client()
    assert isinstance(ext._transport, net._ProxyRouter)
    assert ext._transport.route("https://huggingface.co/a") == "http://p:3128"
    assert ext._transport.route("https://10.0.0.2/a") is None
    await ext.aclose()
    inn = internal_client()
    assert not isinstance(inn._transport, net._ProxyRouter)
    assert inn._mounts == {}
    await inn.aclose()


async def test_internal_ignores_env_proxy(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://p:3128")
    inn = internal_client()
    assert inn._mounts == {}
    await inn.aclose()


async def test_external_without_proxy_is_plain():
    c = external_client()
    assert not isinstance(c._transport, net._ProxyRouter)
    await c.aclose()


def test_describe_proxy_masks(monkeypatch):
    assert describe_proxy() == "no proxy"
    monkeypatch.setenv("http_proxy", "http://user:secret@proxy:3128")
    monkeypatch.setenv("no_proxy", "10.0.0.0/8")
    s = describe_proxy()
    assert "http://user:***@proxy:3128" in s and "secret" not in s and "no_proxy=10.0.0.0/8" in s


def test_models_cache_client_choice(monkeypatch, tmp_path):
    monkeypatch.setenv("https_proxy", "http://p:3128")
    seen = []
    real = httpx.Client

    def fake(**kw):
        seen.append(kw)
        return real(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=httpx.ByteStream(b"GGUF1234"))),
                    **{k: v for k, v in kw.items() if k not in ("transport", "trust_env")})
    monkeypatch.setattr(models_cache.httpx, "Client", fake)
    models_cache.ensure_model("m", "https://hf.co/a/m.gguf", tmp_path / "c1", "http://coord", "t")
    models_cache.ensure_model("m", "coordinator://m.gguf", tmp_path / "c2", "http://10.0.0.1:8080", "t")
    assert isinstance(seen[0]["transport"], net._SyncProxyRouter)
    assert seen[1] == {**seen[1], "trust_env": False} and "transport" not in seen[1]


async def test_library_uses_external_client(monkeypatch, tmp_path):
    from gpupool.coordinator.library import Library
    monkeypatch.setenv("https_proxy", "http://p:3128")
    lb = Library(":memory:", tmp_path)
    assert isinstance(lb._http._transport, net._ProxyRouter)
    await lb._http.aclose()
