"""Async HTTP client for the agent API."""
from __future__ import annotations

import httpx

from gpupool.common.auth import bearer_headers
from gpupool.common.models import EngineSpec, EngineStatus, NodeReport


class AgentError(Exception):
    def __init__(self, status: int, body: str, url: str = ""):
        self.status = status
        self.body = body[:500]
        super().__init__(f"agent {url} returned {status}: {self.body}")


class AgentClient:
    def __init__(self, cluster_token: str, http: httpx.AsyncClient | None = None):
        self._headers = bearer_headers(cluster_token)
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(10.0))

    async def _call(self, method: str, url: str, **kw) -> httpx.Response:
        return await self._http.request(method, url, headers=self._headers, **kw)

    @staticmethod
    def _check(r: httpx.Response) -> None:
        if not r.is_success:
            raise AgentError(r.status_code, r.text, str(r.request.url))

    async def start_engine(self, agent_url: str, spec: EngineSpec) -> EngineStatus:
        r = await self._call("POST", f"{agent_url.rstrip('/')}/engines", json=spec.model_dump())
        self._check(r)
        return EngineStatus.model_validate(r.json())

    async def stop_engine(self, agent_url: str, engine_id: str) -> EngineStatus | None:
        r = await self._call("DELETE", f"{agent_url.rstrip('/')}/engines/{engine_id}")
        if r.status_code == 404:
            return None
        self._check(r)
        return EngineStatus.model_validate(r.json())

    async def get_engine(self, agent_url: str, engine_id: str) -> EngineStatus | None:
        r = await self._call("GET", f"{agent_url.rstrip('/')}/engines/{engine_id}")
        if r.status_code == 404:
            return None
        self._check(r)
        return EngineStatus.model_validate(r.json())

    async def ensure_model(self, agent_url: str, name: str, source: str) -> str:
        # downloads can take many minutes: no read timeout
        r = await self._call(
            "POST", f"{agent_url.rstrip('/')}/models/ensure",
            json={"name": name, "source": source},
            timeout=httpx.Timeout(10.0, read=None),
        )
        self._check(r)
        return r.json()["path"]

    async def report(self, agent_url: str) -> NodeReport:
        r = await self._call("GET", f"{agent_url.rstrip('/')}/report", timeout=httpx.Timeout(5.0))
        self._check(r)
        return NodeReport.model_validate(r.json())

    async def aclose(self) -> None:
        await self._http.aclose()
