"""Pull model: the coordinator fetches /report from every registered agent.

WHY pull instead of push: removing a server must stick; with push heartbeats a deleted
node would re-appear on its next beat.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from gpupool.common.models import NodeReport
from gpupool.coordinator.agent_client import AgentClient
from gpupool.coordinator.store import ServerRecord, Store

log = logging.getLogger("gpupool.poller")


class Poller:
    FAIL_LOG_EVERY_S = 60.0

    def __init__(self, store: Store, client: AgentClient, interval_s: float,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.client = client
        self.interval_s = interval_s
        self.clock = clock
        self._fail_logged: dict[str, float] = {}  # node_id -> clock() of the last failure log line
        self._mismatch_logged: set[tuple[str, str]] = set()  # (registered id, id the agent reports)

    async def probe(self, agent_url: str) -> NodeReport:
        """Fetch a report for registration. Errors (AgentError, httpx) go to the caller."""
        return await self.client.report(agent_url)

    async def _poll_one(self, server: ServerRecord) -> None:
        try:
            report = await self.client.report(server.agent_url)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # last_seen simply ages: that is how the reconciler marks the node dead
            now = self.clock()
            if now - self._fail_logged.get(server.node_id, float("-inf")) >= self.FAIL_LOG_EVERY_S:
                self._fail_logged[server.node_id] = now
                log.warning("poll of %s (%s) failed: %s: %s", server.node_id, server.agent_url,
                            type(e).__name__, e)
            return
        self._fail_logged.pop(server.node_id, None)
        if report.node_id != server.node_id:
            # The agent at this URL was reconfigured: its data must not be filed under another id.
            key = (server.node_id, report.node_id)
            if key not in self._mismatch_logged:
                self._mismatch_logged.add(key)
                log.warning("agent at %s reports node_id %r but %r is registered; ignoring its reports",
                            server.agent_url, report.node_id, server.node_id)
            return
        self._mismatch_logged = {k for k in self._mismatch_logged if k[0] != server.node_id}
        # The server may have been deleted while the request was in flight: do not resurrect it.
        if self.store.get_server(server.node_id) is None:
            return
        # Engine calls go to the URL the user registered (proven reachable at registration),
        # not to whatever the agent believes its address is (e.g. 0.0.0.0 inside a container).
        report.agent_url = server.agent_url
        self.store.upsert_node(report, self.clock())

    async def poll_once(self) -> None:
        servers = self.store.list_servers()
        if servers:
            await asyncio.gather(*(self._poll_one(s) for s in servers))

    async def run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("poll round failed")
            await asyncio.sleep(self.interval_s)
