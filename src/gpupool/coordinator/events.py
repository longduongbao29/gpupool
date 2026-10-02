"""User-facing event log (stored in SQLite) with optional webhook delivery."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import httpx

from gpupool.common.net import external_client
from pydantic import BaseModel

if TYPE_CHECKING:  # store imports Event from here; avoid the cycle at runtime
    from gpupool.coordinator.store import Store

log = logging.getLogger("gpupool.events")

Level = Literal["info", "warning", "error"]
WEBHOOK_LEVELS = {"warning", "error"}


class Event(BaseModel):
    id: int
    ts: float
    level: Level
    kind: str
    message: str  # human readable
    node_id: str | None = None
    model: str | None = None
    read: bool = False


class Notifier:
    """Records events and pushes warnings/errors to a Slack/Discord-style webhook.

    emit() is synchronous and never raises: the reconciler must keep working when the
    webhook is slow or down, so delivery happens in a background task.
    """

    WEBHOOK_LOG_EVERY_S = 60.0

    def __init__(self, store: Store, webhook_url: str = "", http: httpx.AsyncClient | None = None,
                 clock: Callable[[], float] = time.time):
        self.store = store
        self.webhook_url = webhook_url
        self._http = http
        self._clock = clock
        self._tasks: set[asyncio.Task] = set()
        self._last_fail_log = float("-inf")

    def emit(self, level: Level, kind: str, message: str, node_id: str | None = None,
             model: str | None = None) -> Event:
        ev = self.store.add_event(self._clock(), level, kind, message, node_id, model)
        log.log({"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}[level],
                "[%s] %s", kind, message)
        if self.webhook_url and level in WEBHOOK_LEVELS:
            try:
                task = asyncio.get_running_loop().create_task(self._post(ev))
            except RuntimeError:
                return ev  # no event loop (sync caller): the event is stored, just not pushed
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        return ev

    async def _post(self, ev: Event) -> None:
        try:
            if self._http is None:
                self._http = external_client(timeout=httpx.Timeout(5.0))
            r = await self._http.post(
                self.webhook_url,
                json={"text": ev.message, "content": ev.message, "event": ev.model_dump(mode="json")},
                timeout=5.0,
            )
            if not r.is_success:
                raise RuntimeError(f"webhook returned {r.status_code}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            now = time.monotonic()
            if now - self._last_fail_log >= self.WEBHOOK_LOG_EVERY_S:
                self._last_fail_log = now
                log.warning("webhook delivery failed: %s: %s", type(e).__name__, e)

    async def flush(self) -> None:
        """Wait for in-flight webhook deliveries (tests, shutdown)."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def aclose(self) -> None:
        await self.flush()
        if self._http is not None:
            await self._http.aclose()
            self._http = None
