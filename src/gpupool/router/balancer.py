"""Prefix-aware load balancing: rendezvous hash with an outstanding-request escape hatch."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

from gpupool.common.models import ReplicaEndpoint


def _canon(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _text(value: object) -> str:
    return value if isinstance(value, str) else _canon(value)


_PREAMBLE_ROLES = ("system", "developer")


def _anchor(messages: list) -> list:
    """The messages a key is made of.

    Single turn ([system..., user]): everything but the last message, so requests sharing a
    system prompt meet on one replica. Multi-turn: the system messages and the first
    conversational message, which every later turn of the same conversation repeats verbatim,
    so from its second turn on a conversation stays on the replica whose KV cache holds it
    (without a system prompt, from the first). Keying on messages[:-1] instead would change the
    key on every turn (until the canonical prefix passed the 4096-character cap). The switch
    between turns 1 and 2 of a chat with a system prompt is the price of grouping single-turn
    requests by their shared system prompt.
    """
    first = next((i for i, m in enumerate(messages)
                  if not (isinstance(m, dict) and m.get("role") in _PREAMBLE_ROLES)), len(messages))
    if first + 1 < len(messages):
        return messages[:first + 1]
    return messages[:-1]


def prefix_key(body: dict) -> str:
    """Stable key for requests that share a prompt prefix.

    WHY: such requests should land on the replica whose llama.cpp prompt cache
    already holds that prefix.
    """
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        anchor = _anchor(messages) if len(messages) > 1 else messages
        if len(anchor) > 1:
            material = _canon(anchor)[:4096]
        else:  # one message: its content, the same rule whether or not later turns follow
            first = anchor[0]
            content = first.get("content") if isinstance(first, dict) else first
            material = _text(content)[:512]
    else:
        material = _text(body.get("prompt", ""))[:512]
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _hash01(key: str, replica_id: str) -> float:
    """Uniform in (0, 1) per (key, replica)."""
    digest = hashlib.sha256(f"{key}\x00{replica_id}".encode()).digest()
    # 52 bits + 0.5: exact in a float (at most 1 - 2^-53), never 0 or 1 (log(1) would divide by 0)
    return ((int.from_bytes(digest[:8], "big") >> 12) + 0.5) / 2.0 ** 52


def _score(key: str, c: ReplicaEndpoint) -> float:
    """Weighted rendezvous hashing: the highest -w / ln(u) wins, so a replica gets a share of the
    keys proportional to its weight and keys move only to or from a replica that joins or leaves.
    With equal weights the order is that of u itself (plain rendezvous hashing)."""
    return -max(c.weight, 1e-9) / math.log(_hash01(key, c.replica_id))


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
        ranked = sorted(pool, key=lambda c: (-_score(key, c), c.replica_id))
        preferred = ranked[0]
        # Load relative to capacity: a replica twice as fast may hold twice the requests.
        top = max(max(c.weight, 1e-9) for c in pool)
        load = {c.replica_id: self.outstanding(c.replica_id) * top / max(c.weight, 1e-9) for c in pool}
        if load[preferred.replica_id] > min(load.values()) + self.slack:
            # ranked is in rendezvous order and min() is stable -> ties go to rendezvous order
            return min(ranked, key=lambda c: load[c.replica_id])
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
