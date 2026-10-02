"""HTTP proxy handling.

Only traffic that leaves the cluster (Hugging Face, model URLs, alert webhook) may use the
corporate proxy. Traffic between coordinator, agents and llama.cpp must never be proxied: a
proxy usually cannot reach private addresses and tends to break streaming.

httpx's default ``trust_env=True`` would send *everything* through the proxy, so internal
clients disable it (``INTERNAL``) and external clients route explicitly (``_ProxyRouter``)
from the environment, honouring ``no_proxy``. Explicit routing also keeps behaviour
independent of the Windows registry proxy and makes it testable.
"""
from __future__ import annotations

import ipaddress
import os
from collections.abc import MutableMapping
from urllib.parse import urlsplit, urlunsplit

import httpx

_NAMES = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")

# Clients for cluster-internal traffic: never read proxy settings from the environment.
INTERNAL: dict = {"trust_env": False}


def proxy_env(environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    """Proxy settings; the lowercase variable wins over the uppercase one, empty ones are ignored."""
    env = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for name in _NAMES:
        val = (env.get(name) or "").strip() or (env.get(name.upper()) or "").strip()
        if val:
            out[name] = val
    return out


def normalize_proxy_env(environ: MutableMapping[str, str] | None = None) -> None:
    """Set both spellings to the same value so child processes (llama.cpp, uv, git, curl) agree.

    curl only reads lowercase ``http_proxy``; other tools only read uppercase.
    """
    env = os.environ if environ is None else environ
    for name, val in proxy_env(env).items():
        env[name] = val
        env[name.upper()] = val


def _bypass(host: str, no_proxy: str) -> bool:
    """True if ``host`` matches a no_proxy entry.

    Entries are comma/space separated: ``*`` (everything), an exact host, a domain suffix
    (``.corp.local`` or ``corp.local``, both match subdomains), an IP, or a CIDR
    (``10.0.0.0/8``). A ``:port`` on an entry is ignored. Matching is case-insensitive.
    """
    host = host.lower().strip("[]").rstrip(".")
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    for raw in no_proxy.replace(" ", ",").split(","):
        entry = raw.strip().lower()
        if not entry:
            continue
        if entry == "*":
            return True
        if "/" in entry:
            if ip is not None:
                try:
                    if ip in ipaddress.ip_network(entry, strict=False):
                        return True
                except ValueError:
                    pass
            continue
        if entry.startswith("[") and "]" in entry:
            entry = entry[1:entry.index("]")]
        elif entry.count(":") == 1:
            entry = entry.split(":", 1)[0]
        if entry.startswith("*."):
            entry = entry[1:]
        if ip is not None:
            try:
                if ip == ipaddress.ip_address(entry):
                    return True
            except ValueError:
                pass
            continue
        bare = entry.lstrip(".")
        if host == bare or host.endswith("." + bare):
            return True
    return False


def proxy_for(url: str | httpx.URL, env: dict[str, str] | None = None) -> str | None:
    """The proxy URL to use for ``url``, or None for a direct connection."""
    env = proxy_env() if env is None else env
    u = httpx.URL(str(url))
    if _bypass(u.host, env.get("no_proxy", "")):
        return None
    if u.scheme == "https":
        return env.get("https_proxy") or env.get("all_proxy")
    if u.scheme == "http":
        return env.get("http_proxy") or env.get("all_proxy")
    return None


class _ProxyRouter(httpx.AsyncBaseTransport):
    """Picks a proxy (or a direct connection) per request from the environment snapshot."""

    def __init__(self, env: dict[str, str]):
        self._env = env
        self._direct = httpx.AsyncHTTPTransport()
        self._proxied: dict[str, httpx.AsyncHTTPTransport] = {}

    def route(self, url: str | httpx.URL) -> str | None:
        return proxy_for(url, self._env)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        proxy = self.route(request.url)
        if proxy is None:
            return await self._direct.handle_async_request(request)
        t = self._proxied.get(proxy)
        if t is None:
            t = self._proxied[proxy] = httpx.AsyncHTTPTransport(proxy=httpx.Proxy(proxy))
        return await t.handle_async_request(request)

    async def aclose(self) -> None:
        await self._direct.aclose()
        for t in self._proxied.values():
            await t.aclose()


class _SyncProxyRouter(httpx.BaseTransport):
    def __init__(self, env: dict[str, str]):
        self._env = env
        self._direct = httpx.HTTPTransport()
        self._proxied: dict[str, httpx.HTTPTransport] = {}

    def route(self, url: str | httpx.URL) -> str | None:
        return proxy_for(url, self._env)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        proxy = self.route(request.url)
        if proxy is None:
            return self._direct.handle_request(request)
        t = self._proxied.get(proxy)
        if t is None:
            t = self._proxied[proxy] = httpx.HTTPTransport(proxy=httpx.Proxy(proxy))
        return t.handle_request(request)

    def close(self) -> None:
        self._direct.close()
        for t in self._proxied.values():
            t.close()


def external_client(**kw) -> httpx.AsyncClient:
    """Client for traffic leaving the cluster: uses the environment proxy, honours no_proxy."""
    env = proxy_env()
    if env.keys() & {"http_proxy", "https_proxy", "all_proxy"} and "transport" not in kw:
        kw["transport"] = _ProxyRouter(env)
    return httpx.AsyncClient(**kw)


def external_sync_kwargs() -> dict:
    """``httpx.Client`` kwargs for traffic leaving the cluster (empty when no proxy is set)."""
    env = proxy_env()
    if env.keys() & {"http_proxy", "https_proxy", "all_proxy"}:
        return {"transport": _SyncProxyRouter(env)}
    return {}


def internal_client(**kw) -> httpx.AsyncClient:
    """Client for cluster-internal traffic: never proxied."""
    return httpx.AsyncClient(**{**INTERNAL, **kw})


def _mask(url: str) -> str:
    try:
        p = urlsplit(url)
        if p.password is None and p.username is None:
            return url
        host = p.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if p.port:
            host += f":{p.port}"
        user = p.username or ""
        auth = f"{user}:***" if p.password is not None else f"{user}"
        return urlunsplit((p.scheme, f"{auth}@{host}", p.path, p.query, p.fragment))
    except ValueError:
        return "<unparseable proxy url>"


def describe_proxy() -> str:
    """One line for startup logs, credentials masked."""
    env = proxy_env()
    parts = [f"{k}={_mask(v)}" for k, v in env.items() if k != "no_proxy"]
    if not parts:
        return "no proxy"
    if "no_proxy" in env:
        parts.append(f"no_proxy={env['no_proxy']}")
    return "proxy: " + " ".join(parts) + " (external traffic only)"
