import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI

from gpupool.common.models import ReplicaEndpoint
from gpupool.router.balancer import Balancer, prefix_key
from gpupool.router.proxy import RouterMetrics, make_router


def ep(i: int) -> ReplicaEndpoint:
    return ReplicaEndpoint(replica_id=f"r{i}", model="m", base_url=f"http://r{i}:9000")


class Env:
    def __init__(self, behaviors, n=3, api_keys=(), max_retries=2, models=("m",), **router_kw):
        self.behaviors = behaviors  # host -> callable(request) -> response | raises
        self.hits: list[str] = []
        self.bodies: list[dict] = []
        self.errors: list[str] = []
        self.balancer = Balancer()
        self.metrics = RouterMetrics()
        self.cands = [ep(i) for i in range(n)]

        async def upstream(request: httpx.Request):
            self.hits.append(request.url.host)
            self.bodies.append(json.loads(request.content))
            b = self.behaviors.get(request.url.host) or self.behaviors["*"]
            r = b(request)
            return await r if asyncio.iscoroutine(r) else r

        client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        self.app = FastAPI()
        self.app.include_router(make_router(
            get_candidates=lambda m: list(self.cands),
            list_models=lambda: list(models),
            balancer=self.balancer, metrics=self.metrics, api_keys=list(api_keys),
            on_replica_error=self.errors.append, client=client, max_retries=max_retries,
            **router_kw,
        ))

    def client(self):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://t")


def ok(request):
    return httpx.Response(200, json={"host": request.url.host})


def chat(content="hi", stream=False, system="sys"):
    return {"model": "m", "stream": stream,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}


async def test_same_prefix_same_replica_and_cache_prompt():
    env = Env({"*": ok})
    async with env.client() as c:
        for i in range(5):
            r = await c.post("/v1/chat/completions", json=chat(content=f"q{i}"))
            assert r.status_code == 200
    assert len(set(env.hits)) == 1
    assert all(b["cache_prompt"] is True for b in env.bodies)
    assert env.balancer.snapshot() == {}


async def test_cache_prompt_preserved_when_set():
    env = Env({"*": ok})
    async with env.client() as c:
        await c.post("/v1/completions", json={"model": "m", "prompt": "x", "cache_prompt": False})
    assert env.bodies[0]["cache_prompt"] is False


async def test_different_prefixes_spread():
    env = Env({"*": ok})
    async with env.client() as c:
        for i in range(30):
            await c.post("/v1/chat/completions", json=chat(system=f"system {i}"))
    assert len(set(env.hits)) == 3


def test_prefix_key_rules():
    assert prefix_key(chat("a")) == prefix_key(chat("b"))
    assert prefix_key(chat(system="x")) != prefix_key(chat(system="y"))
    one = {"messages": [{"role": "user", "content": "z" * 600}]}
    one2 = {"messages": [{"role": "user", "content": "z" * 512 + "tail"}]}
    assert prefix_key(one) == prefix_key(one2)
    parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert len(prefix_key(parts)) == 64
    assert prefix_key({"prompt": "p" * 600}) == prefix_key({"prompt": "p" * 512 + "x"})


def test_prefix_key_keeps_a_conversation_together():
    sys_msg = {"role": "system", "content": "sys"}
    turn = lambda n: [sys_msg] + [  # noqa: E731
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(2 * n - 1)]
    keys = {prefix_key({"messages": turn(n)}) for n in range(2, 8)}
    assert len(keys) == 1  # every turn after the first: same key, same replica (its KV cache)
    other = [sys_msg, {"role": "user", "content": "other"}, *turn(3)[2:]]
    assert prefix_key({"messages": other}) not in keys  # another conversation is free to go elsewhere
    # without a system prompt the first user message anchors the conversation
    assert (prefix_key({"messages": turn(3)[1:]}) == prefix_key({"messages": turn(5)[1:]}))
    # single turn: unchanged, the system prompt is the key
    assert prefix_key(chat("a")) == prefix_key({"messages": [sys_msg, {"role": "user", "content": "b"}]})


def test_router_upstream_pool_is_not_capped(monkeypatch):
    import gpupool.router.proxy as proxy_mod
    seen = {}

    def fake_client(**kw):
        seen.update(kw)
        return httpx.AsyncClient()

    monkeypatch.setattr(proxy_mod, "internal_client", fake_client)
    make_router(get_candidates=lambda m: [], list_models=lambda: [], balancer=Balancer(),
                metrics=RouterMetrics(), api_keys=[], on_replica_error=lambda r: None)
    assert seen["limits"].max_connections is None


def test_balancer_skew_and_release():
    b = Balancer()
    c = [ep(0), ep(1), ep(2)]
    pref = b.pick(c, "k")
    for _ in range(3):
        b.acquire(pref.replica_id)
    other = b.pick(c, "k")
    assert other.replica_id != pref.replica_id
    for _ in range(2):  # outstanding 2 == min(0)+slack -> still preferred
        b.release(pref.replica_id)
    assert b.pick(c, "k").replica_id == pref.replica_id
    for _ in range(5):
        b.release(pref.replica_id)
    assert b.outstanding(pref.replica_id) == 0
    assert b.pick(c, "k", exclude={pref.replica_id}).replica_id != pref.replica_id
    assert b.pick([ep(0)], "k", exclude={"r0"}) is None


async def test_500_then_success_on_other_replica():
    env = Env({"*": ok})
    first = Balancer().pick(env.cands, prefix_key(chat())).replica_id
    env.behaviors[first] = lambda r: httpx.Response(500, json={"x": 1})
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200
    assert env.errors == [first]
    assert env.hits[0] == first and env.hits[1] != first
    assert env.balancer.snapshot() == {}
    text = env.metrics.render()
    assert 'gpupool_retries_total{model="m"} 1' in text
    assert 'gpupool_requests_total{model="m",code="200"} 1' in text
    assert "gpupool_ttft_seconds_count" not in text  # buffered replies are not TTFT samples


async def test_connect_error_retried():
    def boom(request):
        raise httpx.ConnectError("refused", request=request)

    env = Env({"*": ok})
    first = Balancer().pick(env.cands, prefix_key(chat())).replica_id
    env.behaviors[first] = boom
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200
    assert env.errors == [first]
    assert env.balancer.snapshot() == {}


async def test_4xx_not_retried():
    env = Env({"*": lambda r: httpx.Response(400, json={"error": {"message": "bad"}})})
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 400 and r.json()["error"]["message"] == "bad"
    assert len(env.hits) == 1 and env.errors == []


async def test_all_fail_502_with_retry_cap():
    env = Env({"*": lambda r: httpx.Response(503)}, n=5)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 502
    err = r.json()["error"]
    assert {"message", "type", "code"} <= set(err)
    assert len(env.hits) == 3 and len(env.errors) == 3  # 1 + max_retries
    assert env.balancer.snapshot() == {}


async def test_all_replicas_exhausted_before_cap():
    env = Env({"*": lambda r: httpx.Response(500)}, n=2)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 502 and len(env.hits) == 2


async def test_no_candidates_503_and_unknown_model_404_and_bad_body():
    env = Env({"*": ok}, n=0)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
        assert r.status_code == 503 and "error" in r.json()
        r = await c.post("/v1/chat/completions", json={**chat(), "model": "nope"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
        r = await c.post("/v1/chat/completions", content=b"{not json")
        assert r.status_code == 400
        r = await c.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 400


async def test_auth_and_models():
    env = Env({"*": ok}, api_keys=["sekret"])
    async with env.client() as c:
        assert (await c.post("/v1/chat/completions", json=chat())).status_code == 401
        r = await c.get("/v1/models")
        assert r.status_code == 401 and "error" in r.json()
        h = {"Authorization": "Bearer sekret"}
        r = await c.get("/v1/models", headers=h)
        assert r.json() == {"object": "list",
                            "data": [{"id": "m", "object": "model", "owned_by": "gpupool"}]}
        assert (await c.post("/v1/chat/completions", json=chat(), headers=h)).status_code == 200


def sse_response(chunks, delay=0.0, then_raise=False):
    async def body():
        for ch in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield ch
        if then_raise:
            raise httpx.ReadError("boom")

    return httpx.Response(200, headers={"content-type": "text/event-stream"},
                          stream=_Stream(body))


class _Stream(httpx.AsyncByteStream):
    def __init__(self, fn):
        self.fn = fn

    async def __aiter__(self):
        async for c in self.fn():
            yield c


async def test_streaming_in_order_and_outstanding_zero():
    chunks = [b"data: 1\n\n", b"data: 2\n\n", b"data: [DONE]\n\n"]
    env = Env({"*": lambda r: sse_response(chunks)})
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat(stream=True))
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.content == b"".join(chunks)
    assert env.balancer.snapshot() == {}


async def test_stream_break_midway_emits_error_no_retry():
    env = Env({"*": lambda r: sse_response([b"data: 1\n\n"], then_raise=True)})
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat(stream=True))
    assert r.content.startswith(b"data: 1\n\n")
    last = r.content.split(b"\n\n")[-2]
    assert json.loads(last[len(b"data: "):])["error"]["code"] == "upstream_error"
    assert len(env.hits) == 1
    assert env.balancer.snapshot() == {}


async def test_client_disconnect_releases_outstanding():
    env = Env({"*": lambda r: sse_response([b"data: x\n\n"] * 100, delay=0.05)})
    body = json.dumps(chat(stream=True)).encode()
    sent: list[dict] = []
    first_chunk = asyncio.Event()
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        sent.append(msg)
        if msg["type"] == "http.response.body" and msg.get("body"):
            first_chunk.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "http_version": "1.1", "method": "POST", "path": "/v1/chat/completions",
             "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
             "scheme": "http", "server": ("t", 80), "client": ("c", 1),
             "headers": [(b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode())]}
    await asyncio.wait_for(env.app(scope, receive, send), 5)
    assert first_chunk.is_set()
    assert env.balancer.snapshot() == {}
    assert sum(1 for m in sent if m["type"] == "http.response.body") < 50


async def test_metrics_outstanding_gauge():
    env = Env({"*": ok})
    env.balancer.acquire("r1")
    text = env.metrics.render()
    assert 'gpupool_outstanding{replica="r1"} 1' in text


BIG = 1024 * 1024  # max_body_bytes=BIG keeps these tests small


async def test_body_over_limit_with_content_length_is_413():
    env = Env({"*": ok}, max_body_bytes=BIG)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", content=b"x" * (BIG + 1))
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"
    assert env.hits == []


async def test_chunked_body_over_limit_is_413():
    env = Env({"*": ok}, max_body_bytes=BIG)

    async def gen():  # async generator -> no content-length, chunked
        for _ in range(3):
            yield b"x" * (BIG // 2)

    async with env.client() as c:
        r = await c.post("/v1/chat/completions", content=gen())
    assert r.status_code == 413 and env.hits == []


async def test_body_at_limit_passes():
    env = Env({"*": ok}, max_body_bytes=BIG)
    body = json.dumps({"model": "m", "messages": []})
    body = body[:-1] + " " * (BIG - len(body)) + body[-1:]  # pad inside JSON whitespace
    assert len(body) == BIG
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", content=body.encode())
    assert r.status_code == 200


async def test_unauthenticated_big_body_is_401_not_413():
    env = Env({"*": ok}, api_keys=["k"], max_body_bytes=BIG)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", content=b"x" * (BIG + 1))
    assert r.status_code == 401


async def test_cold_start_waits_then_proxies_when_replica_appears():
    seen = []
    env = Env({"*": ok}, n=0, on_request=lambda m: seen.append(m) or True,
              can_cold_start=lambda m: True, cold_start_timeout_s=5, cold_start_poll_s=0.01)

    async def appear():
        await asyncio.sleep(0.05)
        env.cands.append(ep(0))

    task = asyncio.create_task(appear())
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    await task
    assert r.status_code == 200 and env.hits == ["r0"]
    assert seen == ["m"]  # once per request, not per poll


async def test_cold_start_timeout_is_503_model_loading_with_retry_after():
    env = Env({"*": ok}, n=0, can_cold_start=lambda m: True, cold_start_timeout_s=0.05,
              cold_start_poll_s=0.01)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 503 and r.headers["retry-after"] == "10"
    assert r.json()["error"]["code"] == "model_loading" and "'m' is loading" in r.json()["error"]["message"]
    assert env.metrics.requests[("m", 503)] == 1


async def test_cold_start_not_possible_keeps_plain_503():
    env = Env({"*": ok}, n=0, can_cold_start=lambda m: False, cold_start_timeout_s=5)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_replica"
    assert "retry-after" not in r.headers


async def test_no_callbacks_is_todays_503():
    env = Env({"*": ok}, n=0)
    async with env.client() as c:
        r = await c.post("/v1/chat/completions", json=chat())
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_replica"


async def test_on_request_not_called_for_unknown_model_and_called_once_with_replicas():
    seen = []
    env = Env({"*": ok}, on_request=lambda m: seen.append(m) or False)
    async with env.client() as c:
        await c.post("/v1/chat/completions", json={**chat(), "model": "nope"})
        assert seen == []
        await c.post("/v1/chat/completions", json=chat())
    assert seen == ["m"]


async def test_cold_start_stops_waiting_when_client_disconnects():
    env = Env({"*": ok}, n=0, can_cold_start=lambda m: True, cold_start_timeout_s=30,
              cold_start_poll_s=0.01)
    body = json.dumps(chat()).encode()
    sent = {"body": False}
    out = []

    async def receive():
        if not sent["body"]:
            sent["body"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}  # is_disconnected() only sees ready messages

    async def send(msg):
        out.append(msg)

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
             "query_string": b"", "headers": [(b"content-type", b"application/json"),
                                              (b"content-length", str(len(body)).encode())],
             "server": ("t", 80), "client": ("c", 1), "scheme": "http", "http_version": "1.1", "root_path": ""}
    await asyncio.wait_for(env.app(scope, receive, send), timeout=5)  # far below the 30 s timeout
