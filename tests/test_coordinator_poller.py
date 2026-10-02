import httpx
import pytest

from gpupool.coordinator.agent_client import AgentClient, AgentError
from gpupool.coordinator.poller import Poller
from gpupool.coordinator.store import ServerRecord, Store
from tests.test_coordinator_helpers import Clock, node


class FakeReports:
    def __init__(self):
        self.reports = {}  # url -> NodeReport | Exception
        self.calls = []

    async def report(self, url):
        self.calls.append(url)
        r = self.reports[url]
        if isinstance(r, Exception):
            raise r
        return r


def setup(*ids):
    store, client, clock = Store(":memory:"), FakeReports(), Clock(100.0)
    for i in ids:
        url = f"http://{i}:7070"
        store.add_server(ServerRecord(node_id=i, agent_url=url, added_at=1.0))
        client.reports[url] = node(i)
    return store, client, clock, Poller(store, client, 2.0, clock)


async def test_poll_once_upserts_every_server():
    store, client, clock, poller = setup("a", "b")
    await poller.poll_once()
    assert {n.report.node_id: n.last_seen for n in store.list_nodes()} == {"a": 100.0, "b": 100.0}


async def test_node_id_mismatch_is_skipped_and_logged_once(caplog):
    store, client, clock, poller = setup("a")
    client.reports["http://a:7070"] = node("zzz")
    for _ in range(3):
        await poller.poll_once()
    assert store.list_nodes() == []
    assert len([r for r in caplog.records if "reports node_id" in r.getMessage()]) == 1


async def test_failure_ages_last_seen_and_logs_rate_limited(caplog):
    store, client, clock, poller = setup("a")
    await poller.poll_once()
    client.reports["http://a:7070"] = httpx.ConnectError("down")
    for _ in range(3):
        clock.t += 2
        await poller.poll_once()  # must not raise
    assert store.list_nodes()[0].last_seen == 100.0
    assert len([r for r in caplog.records if "poll of a" in r.getMessage()]) == 1
    clock.t += 61
    await poller.poll_once()
    assert len([r for r in caplog.records if "poll of a" in r.getMessage()]) == 2


async def test_one_failing_server_does_not_block_others():
    store, client, clock, poller = setup("a", "b")
    client.reports["http://a:7070"] = RuntimeError("x")
    await poller.poll_once()
    assert [n.report.node_id for n in store.list_nodes()] == ["b"]


async def test_server_deleted_during_poll_is_not_resurrected():
    store, client, clock, poller = setup("a")
    orig = client.report

    async def slow(url):
        r = await orig(url)
        store.delete_server("a")
        return r

    client.report = slow
    await poller.poll_once()
    assert store.list_nodes() == []


async def test_probe_propagates_errors():
    store, client, clock, poller = setup()
    client.reports["http://x"] = AgentError(401, "no", "http://x")
    with pytest.raises(AgentError):
        await poller.probe("http://x")


async def test_agent_client_report_uses_token_and_parses():
    import respx

    with respx.mock() as m:
        route = m.get("http://h:7070/report").mock(
            return_value=httpx.Response(200, json=node("a").model_dump(mode="json")))
        c = AgentClient("tok")
        rep = await c.report("http://h:7070/")
        assert rep.node_id == "a" and route.calls[0].request.headers["authorization"] == "Bearer tok"
        m.get("http://h:7070/report").mock(return_value=httpx.Response(401, text="nope"))
        with pytest.raises(AgentError) as ei:
            await c.report("http://h:7070")
        assert ei.value.status == 401
        await c.aclose()


async def test_failed_polls_count_reset_and_forget():
    store, client, clock, poller = setup("a", "b")
    await poller.poll_once()
    assert poller.failed_polls("a") == 0
    client.reports["http://a:7070"] = httpx.ConnectError("down")
    for expected in (1, 2, 3):
        await poller.poll_once()
        assert poller.failed_polls("a") == expected
    assert poller.failed_polls("b") == 0
    client.reports["http://a:7070"] = node("a")
    await poller.poll_once()
    assert poller.failed_polls("a") == 0  # success resets
    client.reports["http://a:7070"] = httpx.ConnectError("down")
    await poller.poll_once()
    assert poller.failed_polls("a") == 1
    store.delete_server("a")
    await poller.poll_once()
    assert poller.failed_polls("a") == 0 and "a" not in poller._failed  # forgotten


async def test_cancelled_poll_is_not_a_failure():
    import asyncio
    store, client, clock, poller = setup("a")

    async def cancelled(url):
        raise asyncio.CancelledError()
    client.report = cancelled
    with pytest.raises(asyncio.CancelledError):
        await poller._poll_one(store.list_servers()[0])
    assert poller.failed_polls("a") == 0
