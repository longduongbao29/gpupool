"""OpenAI-compatible router in front of llama-server replicas."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Callable

import httpx

from gpupool.common.net import internal_client
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from gpupool.common.auth import require_bearer
from gpupool.common.models import ReplicaEndpoint
from gpupool.router.balancer import Balancer, prefix_key


def _esc(v: str) -> str:
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
            out.append(f'gpupool_requests_total{{model="{_esc(m)}",code="{c}"}} {n}')
        out.append("# TYPE gpupool_retries_total counter")
        for m, n in sorted(self.retries.items()):
            out.append(f'gpupool_retries_total{{model="{_esc(m)}"}} {n}')
        out.append("# TYPE gpupool_ttft_seconds summary")
        for m in sorted(self.ttft_count):
            out.append(f'gpupool_ttft_seconds_sum{{model="{_esc(m)}"}} {self.ttft_sum[m]}')
            out.append(f'gpupool_ttft_seconds_count{{model="{_esc(m)}"}} {self.ttft_count[m]}')
        out.append("# TYPE gpupool_outstanding gauge")
        if bal is not None:
            for r, n in sorted(bal.snapshot().items()):
                out.append(f'gpupool_outstanding{{replica="{_esc(r)}"}} {n}')
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
) -> APIRouter:
    router = APIRouter()
    auth = require_bearer(*api_keys)
    if metrics.balancer is None:
        metrics.balancer = balancer
    http = client or internal_client(timeout=httpx.Timeout(None, connect=5.0))

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
        try:
            body = json.loads(await request.body())
        except ValueError:
            return _error(400, "request body is not valid JSON",
                          "invalid_request_error", "invalid_json")
        if not isinstance(body, dict) or not isinstance(body.get("model"), str):
            return _error(400, "'model' is required", "invalid_request_error", "missing_model")
        model = body["model"]
        if model not in list_models():
            return _error(404, f"model '{model}' not found",
                          "invalid_request_error", "model_not_found")
        body.setdefault("cache_prompt", True)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        stream = body.get("stream") is True
        key = prefix_key(body)

        exclude: set[str] = set()
        attempts = 0
        last_err = ""
        while True:
            ep = balancer.pick(get_candidates(model), key, exclude)
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
                        metrics.observe_ttft(model, time.monotonic() - started)
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
