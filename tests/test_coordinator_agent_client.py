import httpx

from gpupool.coordinator.agent_client import STOP_TIMEOUT_S, AgentClient


async def test_stop_engine_waits_longer_than_the_agents_grace_period():
    # The agent answers DELETE /engines/{id} only after the process exited (10 s graceful, then
    # kill). With the default 10 s read timeout a slow head looked "done" while still cleaning
    # up, and the RPC servers were stopped under it.
    c = AgentClient("tok")
    seen = {}

    async def fake_call(method, url, **kw):
        seen.update(method=method, url=url, **kw)
        return httpx.Response(404, request=httpx.Request(method, url))

    c._call = fake_call
    assert await c.stop_engine("http://a:7070/", "m-1-head") is None
    assert seen["method"] == "DELETE" and seen["url"] == "http://a:7070/engines/m-1-head"
    assert seen["timeout"].read == STOP_TIMEOUT_S and STOP_TIMEOUT_S > 10.0
    await c.aclose()
