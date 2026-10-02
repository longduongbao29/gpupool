"""Model library: GGUF files the coordinator serves to heads as coordinator://<name>.

Two kinds of items: "hf" (downloaded from Hugging Face into models_dir, owned by us) and
"path" (an existing file registered in place, never copied or deleted by us).
Files are only ever looked up by item name, never by joining user input to a directory.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import httpx

from gpupool.common.net import external_client

from gpupool.common.models import LibraryItem

HF_BASE = "https://huggingface.co"
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_SPLIT_RE = re.compile(r"^(?P<stem>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")
_PROGRESS_INTERVAL_S = 1.0


class LibraryError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


def _safe_basename(file: str) -> str:
    base = os.path.basename(file.replace("\\", "/"))
    if not base or base in (".", ".."):
        raise LibraryError(f"invalid file name {file!r}")
    return base


def _split_info(file: str) -> tuple[str, int, int] | None:
    m = _SPLIT_RE.match(os.path.basename(file))
    return (m["stem"], int(m["idx"]), int(m["total"])) if m else None


class Library:
    def __init__(self, db_path: Path | str, models_dir: Path, hf_token: str = "",
                 http: httpx.AsyncClient | None = None, clock: Callable[[], float] = time.time):
        self.models_dir = Path(models_dir)
        self.hf_token = hf_token
        self._clock = clock
        self._own_http = http is None
        self._http = http or external_client(
            timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True)
        self._lock = threading.RLock()
        self._tasks: dict[str, asyncio.Task] = {}
        self._db = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if str(db_path) != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS library ("
                " name TEXT PRIMARY KEY, path TEXT NOT NULL, source TEXT NOT NULL,"
                " hf_repo TEXT, hf_file TEXT, bytes INTEGER, downloaded INTEGER NOT NULL DEFAULT 0,"
                " status TEXT NOT NULL, error TEXT, created_at REAL NOT NULL,"
                " parts TEXT)")  # parts: JSON [{"file","bytes"}] for HF downloads

    # ---- db helpers -------------------------------------------------------------------

    def _item(self, row: sqlite3.Row) -> LibraryItem:
        return LibraryItem(**{k: row[k] for k in LibraryItem.model_fields})

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    def list(self) -> list[LibraryItem]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM library ORDER BY created_at, name").fetchall()
        return [self._item(r) for r in rows]

    def get(self, name: str) -> LibraryItem | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM library WHERE name=?", (name,)).fetchone()
        return self._item(row) if row else None

    def part_paths(self, name: str) -> list[Path] | None:
        """Ordered paths of every file of a READY item: [path] for a single file, all parts for
        a split HF item. None if the item is missing, not ready, or any file is missing."""
        item = self.get(name)
        if item is None or item.status != "ready":
            return None
        if item.source == "hf":
            paths = [self.models_dir / n for n in self._part_names(item)]
        else:
            paths = [Path(item.path)]
        return paths if all(p.is_file() for p in paths) else None

    def resolve(self, name: str) -> Path | None:
        """Path of a ready item, or of a part file (idx >= 2) of a ready split item.

        Never joins `name` to a directory: a part is found through its item's own part list."""
        paths = self.part_paths(name)
        if paths is not None:
            return paths[0]
        m = _SPLIT_RE.match(name)
        if not m or int(m["idx"]) < 2 or "/" in name or "\\" in name:
            return None
        first = self.get(f"{m['stem']}-00001-of-{m['total']}.gguf")
        if first is None or first.source != "hf":
            return None
        paths = self.part_paths(first.name)
        idx = int(m["idx"])
        if paths is None or idx > len(paths) or paths[idx - 1].name != name:
            return None
        return paths[idx - 1]

    def _parts(self, name: str) -> list[dict]:
        with self._lock:
            row = self._db.execute("SELECT parts FROM library WHERE name=?", (name,)).fetchone()
        return json.loads(row["parts"]) if row and row["parts"] else []

    # ---- Hugging Face -----------------------------------------------------------------

    def _hf_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}

    @staticmethod
    def _check_repo(repo: str) -> None:
        if not _REPO_RE.match(repo or "") or any(s in (".", "..") for s in repo.split("/")):
            raise LibraryError("repo must look like owner/name")

    async def hf_files(self, repo: str) -> list[dict]:
        self._check_repo(repo)
        try:
            r = await self._http.get(f"{HF_BASE}/api/models/{repo}/tree/main",
                                     params={"recursive": "true"}, headers=self._hf_headers())
        except httpx.HTTPError as e:
            raise LibraryError(f"cannot reach Hugging Face: {e}", 502) from e
        if r.status_code in (401, 403):
            raise LibraryError("repo is gated or private: set HF_TOKEN", 403)
        if r.status_code == 404:
            raise LibraryError("repo not found", 404)
        if r.status_code != 200:
            raise LibraryError(f"Hugging Face returned HTTP {r.status_code}", 502)
        try:
            out = [{"file": e["path"], "bytes": int(e.get("size") or 0)}
                   for e in r.json()
                   if e.get("type") == "file" and str(e.get("path", "")).endswith(".gguf")]
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            raise LibraryError("unexpected response from Hugging Face", 502) from e
        return sorted(out, key=lambda x: x["file"])

    async def add_hf(self, repo: str, file: str) -> LibraryItem:
        self._check_repo(repo)
        if (not file.endswith(".gguf") or file.startswith("/")
                or ".." in file.replace("\\", "/").split("/")):
            raise LibraryError("file must be a relative path to a .gguf file in the repo")
        name = _safe_basename(file)
        if self.get(name) is not None:
            raise LibraryError(f"library already has a model named {name}", 409)
        split = _split_info(file)
        total_bytes: int | None = None
        if split:
            stem, idx, total = split
            if idx != 1:
                raise LibraryError(
                    "this is a part of a split GGUF; add the first part (-00001-of-NNNNN) instead")
            listing = {f["file"]: f["bytes"] for f in await self.hf_files(repo)}
            prefix = file[: len(file) - len(os.path.basename(file))]
            parts: list[dict] = []
            for i in range(1, total + 1):
                pf = f"{prefix}{stem}-{i:05d}-of-{total:05d}.gguf"
                if pf not in listing:
                    raise LibraryError(f"split GGUF is incomplete: {pf} not found in repo")
                parts.append({"file": pf, "bytes": listing[pf]})
            total_bytes = sum(p["bytes"] for p in parts) or None
        else:
            parts = [{"file": file, "bytes": None}]
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._exec(
                "INSERT INTO library(name,path,source,hf_repo,hf_file,bytes,downloaded,status,error,"
                "created_at,parts) VALUES(?,?,?,?,?,?,0,'downloading',NULL,?,?)",
                (name, str((self.models_dir / name).resolve()), "hf", repo, file, total_bytes,
                 self._clock(), json.dumps(parts)))
        except sqlite3.IntegrityError as e:
            raise LibraryError(f"library already has a model named {name}", 409) from e
        self._start(name, repo, parts)
        return self.get(name)  # type: ignore[return-value]

    def add_path(self, path: str) -> LibraryItem:
        if not path or not os.path.isabs(path):
            raise LibraryError("path must be absolute")
        p = Path(path)
        if p.suffix.lower() != ".gguf":
            raise LibraryError("path must point to a .gguf file")
        if not p.exists():
            raise LibraryError(f"no such file: {path}")
        if not p.is_file():
            raise LibraryError(f"not a regular file: {path}")
        name = p.name
        split = _split_info(name)
        if split and split[2] > 1:
            # Parts are only served and measured together; add split models from Hugging Face.
            raise LibraryError(
                "split GGUF files cannot be registered by path; add the model from "
                "Hugging Face (first part) instead, or merge it with llama-gguf-split --merge")
        try:
            self._exec(
                "INSERT INTO library(name,path,source,bytes,downloaded,status,created_at)"
                " VALUES(?,?,?,?,0,'ready',?)",
                (name, str(p), "path", p.stat().st_size, self._clock()))
        except sqlite3.IntegrityError as e:
            raise LibraryError(f"library already has a model named {name}", 409) from e
        return self.get(name)  # type: ignore[return-value]

    def delete(self, name: str, in_use: Callable[[str], bool]) -> None:
        item = self.get(name)
        if item is None:
            raise LibraryError(f"no library item {name}", 404)
        if in_use(name):
            raise LibraryError(f"model {name} uses this file; stop and delete it first", 409)
        task = self._tasks.pop(name, None)
        if task:
            task.cancel()
        # Row first: a cancelled download must not resurrect it as "failed".
        self._exec("DELETE FROM library WHERE name=?", (name,))
        if item.source == "hf":
            for fname in self._part_names(item):
                for p in (self.models_dir / fname, self.models_dir / (fname + ".part")):
                    try:
                        p.unlink()
                    except OSError:
                        pass  # missing, or still open by the cancelled task (it cleans up)

    @staticmethod
    def _part_names(item: LibraryItem) -> list[str]:
        split = _split_info(item.name)
        if split and split[1] == 1:
            stem, _, total = split
            return [f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
        return [item.name]

    # ---- downloads --------------------------------------------------------------------

    def resume(self) -> None:
        """Restart downloads a previous process left unfinished (partial files are discarded)."""
        for item in self.list():
            if item.status == "downloading" and item.source == "hf" and item.name not in self._tasks:
                parts = self._parts(item.name) or [{"file": item.hf_file, "bytes": None}]
                self._exec("UPDATE library SET downloaded=0 WHERE name=?", (item.name,))
                self._start(item.name, item.hf_repo or "", parts)

    def _start(self, name: str, repo: str, parts: list[dict]) -> None:
        task = asyncio.get_running_loop().create_task(self._download(name, repo, parts))
        self._tasks[name] = task

        def _done(t: asyncio.Task, n: str = name) -> None:
            if self._tasks.get(n) is t:
                self._tasks.pop(n, None)

        task.add_done_callback(_done)

    async def _download(self, name: str, repo: str, parts: list[dict]) -> None:
        tmp_files: list[Path] = []
        done_bytes = 0
        last_flush = time.monotonic()
        try:
            for part in parts:
                fname = _safe_basename(part["file"])
                final = self.models_dir / fname
                tmp = self.models_dir / (fname + ".part")
                tmp_files.append(tmp)
                url = f"{HF_BASE}/{repo}/resolve/main/{quote(part['file'])}"
                async with self._http.stream("GET", url, headers=self._hf_headers(),
                                             follow_redirects=True) as r:
                    if r.status_code in (401, 403):
                        raise LibraryError("repo is gated or private: set HF_TOKEN", 403)
                    if r.status_code == 404:
                        raise LibraryError(f"{part['file']} not found in {repo}", 404)
                    r.raise_for_status()
                    # With a Content-Encoding the raw bytes are compressed and the length
                    # header does not match the decoded file: decode and skip that check.
                    encoded = r.headers.get("content-encoding", "identity") != "identity"
                    expected = None if encoded else r.headers.get("content-length")
                    # Both the transfer length and the tree listing size must match what we got.
                    expected_ns = {n for n in (int(expected) if expected is not None else None,
                                               None if encoded else part.get("bytes")) if n}
                    chunks = r.aiter_bytes(1024 * 1024) if encoded else r.aiter_raw(1024 * 1024)
                    if expected is not None and len(parts) == 1:
                        # Single-file items have no size until now; the UI needs it for a %.
                        self._exec("UPDATE library SET bytes=? WHERE name=? AND bytes IS NULL",
                                   (int(expected), name))
                    got = 0
                    # Disk writes run in a thread: this event loop also serves the router, and
                    # a blocking write/fsync of a multi-GB file would stall inference traffic.
                    with open(tmp, "wb") as f:
                        async for chunk in chunks:
                            await asyncio.to_thread(f.write, chunk)
                            got += len(chunk)
                            now = time.monotonic()
                            if now - last_flush >= _PROGRESS_INTERVAL_S:
                                last_flush = now
                                self._exec("UPDATE library SET downloaded=? WHERE name=?",
                                           (done_bytes + got, name))
                        await asyncio.to_thread(f.flush)
                        await asyncio.to_thread(os.fsync, f.fileno())
                if got == 0:
                    raise LibraryError(f"empty download of {fname}")
                if any(got != n for n in expected_ns):
                    raise LibraryError(
                        f"truncated download of {fname}: got {got} of {sorted(expected_ns)} bytes")
                if self.get(name) is None:  # deleted while downloading
                    self._cleanup(tmp_files)
                    return
                os.replace(tmp, final)
                done_bytes += got
                self._exec("UPDATE library SET downloaded=? WHERE name=?", (done_bytes, name))
            self._exec("UPDATE library SET status='ready', downloaded=?, bytes=?, error=NULL "
                       "WHERE name=?", (done_bytes, done_bytes, name))
        except asyncio.CancelledError:
            self._cleanup(tmp_files)
            raise
        except Exception as e:  # any failure must end in a visible "failed" state
            self._cleanup(tmp_files)
            # Parts already renamed into place belong to a failed item: drop them too.
            self._cleanup([self.models_dir / _safe_basename(p["file"]) for p in parts])
            msg = e.message if isinstance(e, LibraryError) else f"{type(e).__name__}: {e}"
            self._exec("UPDATE library SET status='failed', error=? WHERE name=?", (msg, name))

    @staticmethod
    def _cleanup(files: list[Path]) -> None:
        for p in files:
            try:
                p.unlink()
            except OSError:
                pass

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self._own_http:
            await self._http.aclose()
        with self._lock:
            self._db.close()
