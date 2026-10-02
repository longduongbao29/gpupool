"""Prefix-aware load balancing: rendezvous hash with an outstanding-request escape hatch."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from gpupool.common.models import ReplicaEndpoint


def _canon(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _text(value: object) -> str:
    return value if isinstance(value, str) else _canon(value)


def prefix_key(body: dict) -> str:
    """Stable key for requests that share a prompt prefix.

    WHY: such requests should land on the replica whose llama.cpp prompt cache
    already holds that prefix.
    """
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        if len(messages) > 1:
            material = _canon(messages[:-1])[:4096]
        else:
            first = messages[0]
            content = first.get("content") if isinstance(first, dict) else first
            material = _text(content)[:512]
    else:
        material = _text(body.get("prompt", ""))[:512]
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _weight(key: str, replica_id: str) -> int:
    digest = hashlib.sha256(f"{key}\x00{replica_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class Balancer:
    def __init__(self, slack: int = 2) -> None:
        self.slack = slack
        self._out: dict[str, int] = {}

    def pick(
        self,
        candidates: list[ReplicaEndpoint],
        key: str,
        exclude: Iterable[str] = frozenset(),
    ) -> ReplicaEndpoint | None:
        excluded = set(exclude)
        pool = [c for c in candidates if c.replica_id not in excluded]
        if not pool:
            return None
        ranked = sorted(pool, key=lambda c: (-_weight(key, c.replica_id), c.replica_id))
        preferred = ranked[0]
        min_out = min(self.outstanding(c.replica_id) for c in pool)
        if self.outstanding(preferred.replica_id) > min_out + self.slack:
            # ranked is in rendezvous order and min() is stable -> ties go to rendezvous order
            return min(ranked, key=lambda c: self.outstanding(c.replica_id))
        return preferred

    def acquire(self, replica_id: str) -> None:
        self._out[replica_id] = self._out.get(replica_id, 0) + 1

    def release(self, replica_id: str) -> None:
        n = self._out.get(replica_id, 0) - 1
        if n > 0:
            self._out[replica_id] = n
        else:
            self._out.pop(replica_id, None)

    def outstanding(self, replica_id: str) -> int:
        return self._out.get(replica_id, 0)

    def snapshot(self) -> dict[str, int]:
        return dict(self._out)
