"""OpenAI-compatible router in front of llama-server replicas."""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from gpupool.common.auth import require_bearer
from gpupool.common.net import internal_client
from gpupool.common.models import ReplicaEndpoint
from gpupool.router.balancer import Balancer, prefix_key


def prom_label_escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class RouterMetrics:
    def __init__(self, balancer: Balancer | None = None) -> None:
        self.balancer = balancer
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.retries: dict[str, int] = defaultdict(int)
        self.ttft_sum: dict[str, float] = defaultdict(float)
        self.ttft_count: dict[str, int] = defaultdict(int)

    def observe_request(self, model: str, code: int) -> None:
        self.requests[(model, code)] += 1

    def observe_retry(self, model: str) -> None:
        self.retries[model] += 1

    def observe_ttft(self, model: str, seconds: float) -> None:
        self.ttft_sum[model] += seconds
        self.ttft_count[model] += 1

    def render(self, balancer: Balancer | None = None) -> str:
        bal = balancer or self.balancer
        out = ["# TYPE gpupool_requests_total counter"]
        for (m, c), n in sorted(self.requests.items()):
            out.append(f'gpupool_requests_total{{model="{prom_label_escape(m)}",code="{c}"}} {n}')
        out.append("# TYPE gpupool_retries_total counter")
        for m, n in sorted(self.retries.items()):
            out.append(f'gpupool_retries_total{{model="{prom_label_escape(m)}"}} {n}')
        out.append("# TYPE gpupool_ttft_seconds summary")
        for m in sorted(self.ttft_count):
            out.append(f'gpupool_ttft_seconds_sum{{model="{prom_label_escape(m)}"}} {self.ttft_sum[m]}')
            out.append(f'gpupool_ttft_seconds_count{{model="{prom_label_escape(m)}"}} {self.ttft_count[m]}')
        out.append("# TYPE gpupool_outstanding gauge")
        if bal is not None:
            for r, n in sorted(bal.snapshot().items()):
                out.append(f'gpupool_outstanding{{replica="{prom_label_escape(r)}"}} {n}')
        return "\n".join(out) + "\n"


def _error(status: int, message: str, etype: str, code: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": etype, "code": code}}, status_code=status
    )


class _Retryable(Exception):
    pass


def make_router(
    *,
    get_candidates: Callable[[str], list[ReplicaEndpoint]],
    list_models: Callable[[], list[str]],
    balancer: Balancer,
    metrics: RouterMetrics,
    api_keys: list[str],
    on_replica_error: Callable[[str], None],
    client: httpx.AsyncClient | None = None,
    max_retries: int = 2,
    max_body_bytes: int = 32 * 1024 * 1024,
    on_request: Callable[[str], bool] | None = None,
    can_cold_start: Callable[[str], bool] | None = None,
    cold_start_timeout_s: float = 120.0,
    cold_start_poll_s: float = 0.5,
) -> APIRouter:
    router = APIRouter()
    auth = require_bearer(*api_keys)
    if metrics.balancer is None:
        metrics.balancer = balancer
    # httpx defaults to 100 connections in total: request 101 would wait for a free one, with
    # no pool timeout (None), invisibly in the coordinator instead of in llama-server's slot
    # queue. llama-server and the balancer are the limit, so the pool is not.
    http = client or internal_client(
        timeout=httpx.Timeout(None, connect=5.0),
        limits=httpx.Limits(max_connections=None, max_keepalive_connections=256))

    def check_auth(request: Request) -> JSONResponse | None:
        try:
            auth(authorization=request.headers.get("authorization"))
        except HTTPException as e:
            return _error(e.status_code, str(e.detail), "authentication_error", "invalid_api_key")
        return None

    @router.get("/v1/models")
    async def models(request: Request):
        if (denied := check_auth(request)) is not None:
            return denied
        return {
            "object": "list",
            "data": [{"id": n, "object": "model", "owned_by": "gpupool"} for n in list_models()],
        }

    async def handle(request: Request, path: str):
        if (denied := check_auth(request)) is not None:
            return denied
        too_big = _error(413, f"request body exceeds {max_body_bytes // (1024 * 1024)} MB",
                         "invalid_request_error", "request_too_large")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_body_bytes:
            return too_big
        # Chunked uploads carry no content-length, so enforce the cap while reading: one
        # client must not be able to buffer an unbounded body in coordinator memory.
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_body_bytes:
                return too_big
            chunks.append(chunk)
        try:
            body = json.loads(b"".join(chunks))
        except ValueError:
            return _error(400, "request body is not valid JSON",
                          "invalid_request_error", "invalid_json")
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            return _error(400, "'model' is required", "invalid_request_error", "missing_model")
        model = body["model"]
        if model not in list_models():
            return _error(404, f"model '{model}' not found",
                          "invalid_request_error", "model_not_found")
        if on_request is not None:
            on_request(model)  # may start loading an unloaded model
        body.setdefault("cache_prompt", True)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        stream = body.get("stream") is True
        key = prefix_key(body)

        exclude: set[str] = set()
        attempts = 0
        last_err = ""
        waited = False
        while True:
            ep = balancer.pick(get_candidates(model), key, exclude)
            if (ep is None and attempts == 0 and not waited and can_cold_start is not None
                    and can_cold_start(model)):
                # The model is unloaded on purpose: hold the request until a replica is ready
                # instead of failing it, but not past the timeout or a gone client.
                waited = True
                deadline = time.monotonic() + cold_start_timeout_s
                while ep is None and time.monotonic() < deadline:
                    await asyncio.sleep(cold_start_poll_s)
                    if await request.is_disconnected():
                        break
                    ep = balancer.pick(get_candidates(model), key, exclude)
                if ep is None:
                    metrics.observe_request(model, 503)
                    resp503 = _error(503, f"model '{model}' is loading, retry shortly",
                                     "server_error", "model_loading")
                    resp503.headers["Retry-After"] = "10"
                    return resp503
            if ep is None:
                if attempts == 0:
                    metrics.observe_request(model, 503)
                    return _error(503, f"no ready replica for model '{model}'",
                                  "server_error", "no_replica")
                metrics.observe_request(model, 502)
                return _error(502, f"all replicas failed: {last_err}",
                              "server_error", "upstream_error")

            rid = ep.replica_id
            balancer.acquire(rid)
            state = {"released": False}

            def release(rid: str = rid, state: dict = state) -> None:
                if not state["released"]:
                    state["released"] = True
                    balancer.release(rid)

            resp: httpx.Response | None = None
            started = time.monotonic()
            try:
                try:
                    req = http.build_request(
                        "POST", ep.base_url.rstrip("/") + path, content=payload,
                        headers={"content-type": "application/json"},
                    )
                    resp = await http.send(req, stream=True)
                    if resp.status_code >= 500:
                        raise _Retryable(f"upstream status {resp.status_code}")
                    ctype = resp.headers.get("content-type", "application/json")
                    if not stream or resp.status_code >= 400:
                        data = await resp.aread()
                        # No TTFT here: a buffered reply's duration is not time-to-first-token.
                        metrics.observe_request(model, resp.status_code)
                        status = resp.status_code
                        await resp.aclose()
                        release()
                        return Response(content=data, status_code=status, media_type=ctype)
                    it = resp.aiter_raw()
                    try:
                        first: bytes | None = await it.__anext__()
                    except StopAsyncIteration:
                        first = None
                    metrics.observe_ttft(model, time.monotonic() - started)
                except (httpx.HTTPError, _Retryable) as e:
                    # nothing has reached the client yet: safe to retry elsewhere
                    if resp is not None:
                        try:
                            await resp.aclose()
                        except Exception:
                            pass
                    release()
                    last_err = str(e) or type(e).__name__
                    on_replica_error(rid)
                    exclude.add(rid)
                    if attempts >= max_retries:
                        metrics.observe_request(model, 502)
                        return _error(502, f"upstream failed: {last_err}",
                                      "server_error", "upstream_error")
                    attempts += 1
                    metrics.observe_retry(model)
                    continue
            except BaseException:
                # cancellation / unexpected error before the stream was handed over
                if resp is not None:
                    try:
                        await resp.aclose()
                    except BaseException:
                        pass
                release()
                raise

            metrics.observe_request(model, resp.status_code)
            is_sse = "text/event-stream" in ctype

            async def cleanup(resp=resp, release=release) -> None:
                release()
                try:
                    await resp.aclose()
                except BaseException:
                    pass

            async def gen(it=it, first=first) -> AsyncIterator[bytes]:
                try:
                    if first:
                        yield first
                    async for chunk in it:
                        yield chunk
                except httpx.HTTPError as e:
                    on_replica_error(rid)
                    if is_sse:
                        err = {"error": {"message": f"upstream stream broke: {e}",
                                         "type": "server_error", "code": "upstream_error"}}
                        yield f"data: {json.dumps(err)}\n\n".encode()
                finally:
                    await cleanup()

            # BackgroundTask is a second safety net for the case where the generator is
            # never started (client gone before the first send); cleanup is idempotent.
            return StreamingResponse(gen(), status_code=resp.status_code, media_type=ctype,
                                     background=BackgroundTask(cleanup))

    @router.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handle(request, "/v1/chat/completions")

    @router.post("/v1/completions")
    async def completions(request: Request):
        return await handle(request, "/v1/completions")

    return router
