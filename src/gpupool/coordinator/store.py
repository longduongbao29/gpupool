"""SQLite state: node heartbeats, model registry, replica records."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from pydantic import BaseModel

from gpupool.common.models import ModelSpec, NodeReport, ReplicaRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY, report TEXT NOT NULL, last_seen REAL NOT NULL);
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY, spec TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS replicas (
    replica_id TEXT PRIMARY KEY, model TEXT NOT NULL, state TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS replicas_model ON replicas(model);
"""


class NodeRecord(BaseModel):
    report: NodeReport
    last_seen: float


class Store:
    def __init__(self, db_path: Path | str):
        memory = str(db_path) == ":memory:"
        if not memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            if not memory:
                self._conn.execute("PRAGMA journal_mode=WAL")
            with self._conn:
                self._conn.executescript(_SCHEMA)

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._lock, self._conn:
            self._conn.execute(sql, params)

    def _read(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # nodes
    def upsert_node(self, report: NodeReport, now: float) -> None:
        self._write(
            "INSERT INTO nodes(node_id, report, last_seen) VALUES(?,?,?) "
            "ON CONFLICT(node_id) DO UPDATE SET report=excluded.report, last_seen=excluded.last_seen",
            (report.node_id, report.model_dump_json(), now),
        )

    def list_nodes(self) -> list[NodeRecord]:
        rows = self._read("SELECT report, last_seen FROM nodes ORDER BY node_id")
        return [NodeRecord(report=NodeReport.model_validate_json(r), last_seen=t) for r, t in rows]

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

    def close(self) -> None:
        with self._lock:
            self._conn.close()
