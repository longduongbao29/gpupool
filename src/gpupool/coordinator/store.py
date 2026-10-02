"""SQLite state: node heartbeats, model registry, replica records."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from pydantic import BaseModel

from gpupool.common.models import Device, ModelSpec, NodeReport, ReplicaRecord
from gpupool.coordinator.events import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY, report TEXT NOT NULL, last_seen REAL NOT NULL);
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY, spec TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS replicas (
    replica_id TEXT PRIMARY KEY, model TEXT NOT NULL, state TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS replicas_model ON replicas(model);
CREATE TABLE IF NOT EXISTS servers (
    node_id TEXT PRIMARY KEY, agent_url TEXT NOT NULL, added_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS removed_servers (
    node_id TEXT PRIMARY KEY, removed_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS gpu_flags (
    node_id TEXT NOT NULL, device_id TEXT NOT NULL, enabled INTEGER NOT NULL,
    PRIMARY KEY (node_id, device_id));
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL,
    message TEXT NOT NULL, node_id TEXT, model TEXT, read INTEGER NOT NULL DEFAULT 0);
"""

EVENTS_KEEP = 1000


def gpu_key(device: Device) -> str:
    """Key of a GPU in gpu_flags: the physical card (uuid) when reported, else its position.
    device_id shifts when a GPU drops off the bus; the uuid does not."""
    return device.uuid or device.device_id


class NodeRecord(BaseModel):
    report: NodeReport
    last_seen: float

    def alive(self, now: float, timeout_s: float) -> bool:
        return now - self.last_seen <= timeout_s


class ServerRecord(BaseModel):
    """A registered agent: the coordinator polls it and plans onto it."""

    node_id: str
    agent_url: str
    added_at: float


class Store:
    def __init__(self, db_path: Path | str):
        memory = str(db_path) == ":memory:"
        if not memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._version = 0
        with self._lock:
            if not memory:
                self._conn.execute("PRAGMA journal_mode=WAL")
                # Same as Library (own connection, same file): wait for its writes, don't fail.
                self._conn.execute("PRAGMA busy_timeout=5000")
            with self._conn:
                self._conn.executescript(_SCHEMA)

    @property
    def version(self) -> int:
        """Bumped (under the lock) by every write to nodes, models, replicas or servers, so
        readers can cache derived data and rebuild only when this changes."""
        return self._version

    def _write(self, sql: str, params: tuple = (), bump: bool = True) -> None:
        with self._lock, self._conn:
            self._conn.execute(sql, params)
            if bump:
                self._version += 1

    def _read(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # nodes
    def upsert_node(self, report: NodeReport, now: float) -> None:
        by_id = {d.device_id: d.uuid for d in report.devices if d.uuid}
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO nodes(node_id, report, last_seen) VALUES(?,?,?) "
                "ON CONFLICT(node_id) DO UPDATE SET report=excluded.report, last_seen=excluded.last_seen",
                (report.node_id, report.model_dump_json(), now))
            if by_id:
                self._migrate_flags(report.node_id, by_id)
            self._version += 1

    def _migrate_flags(self, node_id: str, uuid_by_device_id: dict[str, str]) -> None:
        """Rewrite legacy flag rows (keyed by device_id) to the card's uuid.

        Runs on the report of an agent that has just started sending uuids: the numbering is
        then still the one the legacy rows were written under, so device_id -> uuid is safe.
        Later reports find nothing to do (rows are already uuid keys). Flags do not affect
        routing, so no version bump. Caller holds the lock and a transaction."""
        have = {r[0] for r in self._conn.execute("SELECT device_id FROM gpu_flags WHERE node_id=?", (node_id,))}
        for did, uid in uuid_by_device_id.items():
            if did in have and uid not in have:
                self._conn.execute("UPDATE gpu_flags SET device_id=? WHERE node_id=? AND device_id=?",
                                   (uid, node_id, did))
                have.discard(did)
                have.add(uid)

    def list_nodes(self) -> list[NodeRecord]:
        rows = self._read("SELECT report, last_seen FROM nodes ORDER BY node_id")
        return [NodeRecord(report=NodeReport.model_validate_json(r), last_seen=t) for r, t in rows]

    # servers
    def add_server(self, rec: ServerRecord) -> None:
        self._write(
            "INSERT INTO servers(node_id, agent_url, added_at) VALUES(?,?,?) "
            "ON CONFLICT(node_id) DO UPDATE SET agent_url=excluded.agent_url, added_at=excluded.added_at",
            (rec.node_id, rec.agent_url, rec.added_at),
        )

    def get_server(self, node_id: str) -> ServerRecord | None:
        rows = self._read("SELECT node_id, agent_url, added_at FROM servers WHERE node_id=?", (node_id,))
        return ServerRecord(node_id=rows[0][0], agent_url=rows[0][1], added_at=rows[0][2]) if rows else None

    def list_servers(self) -> list[ServerRecord]:
        rows = self._read("SELECT node_id, agent_url, added_at FROM servers ORDER BY added_at, node_id")
        return [ServerRecord(node_id=a, agent_url=b, added_at=c) for a, b, c in rows]

    def delete_server(self, node_id: str) -> None:
        """Remove the server with its cached report and GPU flags in one transaction,
        so a removed node can never linger as a half-registered ghost."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM servers WHERE node_id=?", (node_id,))
            self._conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
            self._conn.execute("DELETE FROM gpu_flags WHERE node_id=?", (node_id,))
            self._version += 1

    # servers removed by the operator: auto-join must not bring them back
    def mark_removed(self, node_id: str, now: float | None = None) -> None:
        self._write(
            "INSERT INTO removed_servers(node_id, removed_at) VALUES(?,?) "
            "ON CONFLICT(node_id) DO UPDATE SET removed_at=excluded.removed_at",
            (node_id, time.time() if now is None else now), bump=False)

    def clear_removed(self, node_id: str) -> None:
        self._write("DELETE FROM removed_servers WHERE node_id=?", (node_id,), bump=False)

    def is_removed(self, node_id: str) -> bool:
        return bool(self._read("SELECT 1 FROM removed_servers WHERE node_id=?", (node_id,)))

    # gpu flags
    def set_gpu_enabled(self, node_id: str, key: str, enabled: bool) -> None:
        """`key` is gpu_key(device): the uuid when the agent reports one, else the device_id."""
        self._write(
            "INSERT INTO gpu_flags(node_id, device_id, enabled) VALUES(?,?,?) "
            "ON CONFLICT(node_id, device_id) DO UPDATE SET enabled=excluded.enabled",
            (node_id, key, int(enabled)), bump=False,
        )

    def gpu_flags(self) -> dict[tuple[str, str], bool]:
        """Explicit flags only, keyed (node_id, gpu_key); a device without a row is enabled."""
        return {(n, d): bool(e) for n, d, e in self._read("SELECT node_id, device_id, enabled FROM gpu_flags")}

    # events
    def add_event(self, ts: float, level: str, kind: str, message: str,
                  node_id: str | None = None, model: str | None = None) -> Event:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO events(ts, level, kind, message, node_id, model, read) VALUES(?,?,?,?,?,?,0)",
                (ts, level, kind, message, node_id, model))
            eid = cur.lastrowid
            self._conn.execute("DELETE FROM events WHERE id <= ?", (eid - EVENTS_KEEP,))
        return Event(id=eid, ts=ts, level=level, kind=kind, message=message,  # type: ignore[arg-type]
                     node_id=node_id, model=model, read=False)

    def list_events(self, limit: int = 50, after_id: int | None = None) -> list[Event]:
        """Newest first; `after_id` keeps only events with a larger id."""
        sql = "SELECT id, ts, level, kind, message, node_id, model, read FROM events"
        params: list = []
        if after_id is not None:
            sql += " WHERE id > ?"
            params.append(after_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [Event(id=i, ts=t, level=lv, kind=k, message=m, node_id=n, model=md, read=bool(r))
                for i, t, lv, k, m, n, md, r in self._read(sql, tuple(params))]

    def unread_count(self) -> int:
        return self._read("SELECT COUNT(*) FROM events WHERE read=0")[0][0]

    def mark_read(self, up_to_id: int) -> None:
        self._write("UPDATE events SET read=1 WHERE id<=?", (up_to_id,), bump=False)

    # models
    def put_model(self, spec: ModelSpec) -> None:
        self._write(
            "INSERT INTO models(name, spec) VALUES(?,?) "
            "ON CONFLICT(name) DO UPDATE SET spec=excluded.spec",
            (spec.name, spec.model_dump_json()),
        )

    def get_model(self, name: str) -> ModelSpec | None:
        rows = self._read("SELECT spec FROM models WHERE name=?", (name,))
        return ModelSpec.model_validate_json(rows[0][0]) if rows else None

    def list_models(self) -> list[ModelSpec]:
        return [ModelSpec.model_validate_json(r[0]) for r in self._read("SELECT spec FROM models ORDER BY name")]

    def delete_model(self, name: str) -> None:
        self._write("DELETE FROM models WHERE name=?", (name,))

    # replicas
    def put_replica(self, rec: ReplicaRecord) -> None:
        self._write(
            "INSERT INTO replicas(replica_id, model, state, created_at, updated_at, data) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(replica_id) DO UPDATE SET model=excluded.model, state=excluded.state, "
            "created_at=excluded.created_at, updated_at=excluded.updated_at, data=excluded.data",
            (rec.replica_id, rec.model, rec.state, rec.created_at, rec.updated_at, rec.model_dump_json()),
        )

    def get_replica(self, replica_id: str) -> ReplicaRecord | None:
        rows = self._read("SELECT data FROM replicas WHERE replica_id=?", (replica_id,))
        return ReplicaRecord.model_validate_json(rows[0][0]) if rows else None

    def list_replicas(self, model: str | None = None, states: set[str] | None = None) -> list[ReplicaRecord]:
        sql, params, where = "SELECT data FROM replicas", [], []
        if model is not None:
            where.append("model=?")
            params.append(model)
        if states is not None:
            if not states:
                return []
            where.append(f"state IN ({','.join('?' * len(states))})")
            params.extend(sorted(states))
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at, replica_id"
        return [ReplicaRecord.model_validate_json(r[0]) for r in self._read(sql, tuple(params))]

    def set_replica_state(self, replica_id: str, state: str, error: str | None = None,
                          now: float | None = None) -> None:
        """Read-modify-write in one lock+transaction so the JSON blob and columns never diverge."""
        with self._lock, self._conn:
            row = self._conn.execute("SELECT data FROM replicas WHERE replica_id=?", (replica_id,)).fetchone()
            if row is None:
                return
            rec = ReplicaRecord.model_validate_json(row[0])
            rec.state = state  # type: ignore[assignment]
            rec.error = error
            rec.updated_at = time.time() if now is None else now
            self._conn.execute(
                "UPDATE replicas SET state=?, updated_at=?, data=? WHERE replica_id=?",
                (rec.state, rec.updated_at, rec.model_dump_json(), replica_id),
            )
            self._version += 1

    def prune_replicas(self, keep_per_model: int) -> int:
        """Delete terminal replicas (state 'stopped' or 'failed') of each model beyond the
        newest `keep_per_model` (order: created_at, replica_id). Never deletes any other state."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM replicas WHERE replica_id IN ("
                " SELECT replica_id FROM ("
                "  SELECT replica_id, ROW_NUMBER() OVER ("
                "   PARTITION BY model ORDER BY created_at DESC, replica_id DESC) AS rn"
                "  FROM replicas WHERE state IN ('stopped', 'failed')) WHERE rn > ?)",
                (max(0, keep_per_model),))
            if cur.rowcount:
                self._version += 1
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
