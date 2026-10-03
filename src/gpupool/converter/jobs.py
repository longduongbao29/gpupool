"""Conversion job manager: Hugging Face / local weights -> GGUF in the model library.

One worker runs one job at a time (FIFO): download/stage -> convert -> [calibrate] -> quantize ->
validate -> library. "calibrate" (llama-imatrix on the 16-bit intermediate) only runs for jobs whose
quant type needs an importance matrix or benefits from one (see quant.imatrix_wanted). Job rows live in sqlite (own connection, like Library); the heavy work is subprocesses
(output streamed line by line) and threads, so the event loop that also serves inference is
never blocked.

On-disk layout, everything under models_dir:
  .hf/<owner>__<name>@<revision>/   shared download cache (HF sources only)
  .convert/<job id>/                scratch space of one job: src/ (staged links), intermediate
                                    and output files; for needs_review only the output remains
Nothing outside these two folders is ever deleted, and path sources are never modified: the
converter reads a staging folder of links to the selected files, never the source itself.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from gpupool.common.models import LibraryItem
from gpupool.converter.models import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    ClusterVram,
    ConvertError,
    ConvertJob,
    ConvertRequest,
    SourceFile,
    Validation,
)
from gpupool.converter.quant import (
    check_output_name,
    default_output_name,
    estimate_bytes,
    imatrix_wanted,
    option,
    plan_steps,
)
from gpupool.converter.source import HfClient, inspect_source, local_files, select_files
from gpupool.converter.toolchain import IMATRIX_NOT_INSTALLED, ImatrixProgress, Toolchain, run_tool
from gpupool.converter.validate import validate

log = logging.getLogger(__name__)

_PROGRESS_INTERVAL_S = 1.0
_LOG_MEM = 200
_LOG_DB = 50
_DISK_MARGIN = 512 * 1024 * 1024
_CALIBRATION_MAX = 20 * 1024 * 1024  # largest calibration text accepted
_DEFAULT_CHUNKS = 100  # llama-imatrix chunks when the request says 0
# The multilingual text shipped with gpupool (tests point this at their own file).
BUILTIN_CALIBRATION = Path(__file__).parent / "data" / "calibration.txt"
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_TQDM_RE = re.compile(r"(\d{1,3})%\|")
_QUANT_RE = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]")
_NOTABLE_RE = re.compile(r"error|not supported|not implemented|exception|failed|cannot|invalid",
                         re.IGNORECASE)


class LibraryPort(Protocol):
    def name_taken(self, name: str) -> bool: ...
    def add_converted(self, name: str, path: Path, hf_repo: str | None) -> LibraryItem: ...
    def locate_dir(self, path: str) -> Path: ...
    def locate_file(self, path: str) -> Path: ...


@dataclass
class _Ctx:
    """Runtime state of the job being worked on (not persisted)."""

    task: asyncio.Task | None = None
    user_cancel: bool = False  # cancel() asked, as opposed to the process shutting down
    stop: threading.Event = field(default_factory=threading.Event)  # polled by thread work
    pending: set = field(default_factory=set)  # futures of thread work still running
    tail: deque = field(default_factory=lambda: deque(maxlen=60))  # non-progress lines, this tool


def _is_progress(line: str) -> bool:
    return bool(_TQDM_RE.search(line) or _QUANT_RE.match(line))


def _gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB"


def _replace(src: Path, dst: Path) -> None:
    """os.replace with a few retries: on Windows a just-closed memory map or antivirus scan can
    hold the file for a moment. Falls back to copy+replace across filesystems."""
    last: OSError | None = None
    for _ in range(6):
        try:
            os.replace(src, dst)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.25)
        except OSError as e:
            last = e
            break
    if isinstance(last, PermissionError):
        raise last
    part = Path(str(dst) + ".part")
    try:
        shutil.copyfile(src, part)
        with open(part, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(part, dst)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise
    os.unlink(src)


class ConvertManager:
    def __init__(self, db_path: Path | str, models_dir: Path, toolchain: Toolchain,
                 library: LibraryPort, hf: HfClient, *, threads: int = 0,
                 clock: Callable[[], float] = time.time):
        self.models_dir = Path(models_dir)
        self.convert_root = self.models_dir / ".convert"
        self.hf_root = self.models_dir / ".hf"
        self.toolchain = toolchain
        self.library = library
        self.hf = hf
        self.threads = threads
        self._clock = clock
        self._lock = threading.RLock()
        self._jobs: dict[str, ConvertJob] = {}
        self._plans: dict[str, dict] = {}
        self._logs: dict[str, deque] = {}
        self._ctxs: dict[str, _Ctx] = {}
        self._queue: deque[str] = deque()
        self._last_flush: dict[str, float] = {}
        self._seq = 0
        self._worker_task: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self.models_dir.mkdir(parents=True, exist_ok=True)
        if str(db_path) != ":memory:":
            # Like Store: a fresh install's data folder may not exist yet (sqlite will not create it).
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            if str(db_path) != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
                self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS convert_jobs ("
                " id TEXT PRIMARY KEY, seq INTEGER NOT NULL, state TEXT NOT NULL,"
                " output_name TEXT NOT NULL, request TEXT NOT NULL, validation TEXT, error TEXT,"
                " stage_progress REAL, bytes_done INTEGER NOT NULL DEFAULT 0, bytes_total INTEGER,"
                " output_bytes INTEGER, est_output_bytes INTEGER, log_tail TEXT, plan TEXT,"
                " created_at REAL NOT NULL, started_at REAL, finished_at REAL)")
            cols = {r[1] for r in self._db.execute("PRAGMA table_info(convert_jobs)")}
            if "failed_stage" not in cols:  # databases of the first release
                self._db.execute("ALTER TABLE convert_jobs ADD COLUMN failed_stage TEXT")
            if "imatrix_used" not in cols:
                self._db.execute(
                    "ALTER TABLE convert_jobs ADD COLUMN imatrix_used INTEGER NOT NULL DEFAULT 0")
            rows = self._db.execute("SELECT * FROM convert_jobs ORDER BY seq").fetchall()
        for r in rows:
            job = ConvertJob(
                id=r["id"], request=ConvertRequest.model_validate_json(r["request"]),
                state=r["state"], stage_progress=r["stage_progress"], bytes_done=r["bytes_done"],
                bytes_total=r["bytes_total"], output_name=r["output_name"],
                output_bytes=r["output_bytes"], est_output_bytes=r["est_output_bytes"],
                failed_stage=r["failed_stage"], imatrix_used=bool(r["imatrix_used"]),
                validation=(Validation.model_validate_json(r["validation"])
                            if r["validation"] else None),
                error=r["error"], log_tail=json.loads(r["log_tail"] or "[]"),
                created_at=r["created_at"], started_at=r["started_at"],
                finished_at=r["finished_at"])
            self._jobs[job.id] = job
            self._plans[job.id] = json.loads(r["plan"] or "{}")
            self._seq = max(self._seq, r["seq"])

    # ---- persistence ---------------------------------------------------------------------

    def _persist(self, job: ConvertJob, *, seq: int | None = None) -> None:
        tail = list(self._logs.get(job.id, job.log_tail))[-_LOG_DB:]
        vals = (job.state, job.output_name, job.request.model_dump_json(),
                job.validation.model_dump_json() if job.validation else None, job.error,
                job.stage_progress, job.bytes_done, job.bytes_total, job.output_bytes,
                job.est_output_bytes, json.dumps(tail), json.dumps(self._plans.get(job.id, {})),
                job.created_at, job.started_at, job.finished_at, job.failed_stage,
                int(job.imatrix_used))
        with self._lock:
            cur = self._db.execute(
                "UPDATE convert_jobs SET state=?, output_name=?, request=?, validation=?, error=?,"
                " stage_progress=?, bytes_done=?, bytes_total=?, output_bytes=?, est_output_bytes=?,"
                " log_tail=?, plan=?, created_at=?, started_at=?, finished_at=?, failed_stage=?,"
                " imatrix_used=? WHERE id=?",
                (*vals, job.id))
            if cur.rowcount == 0:
                self._db.execute(
                    "INSERT INTO convert_jobs(state, output_name, request, validation, error,"
                    " stage_progress, bytes_done, bytes_total, output_bytes, est_output_bytes,"
                    " log_tail, plan, created_at, started_at, finished_at, failed_stage,"
                    " imatrix_used, id, seq)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (*vals, job.id, seq if seq is not None else self._seq))
        self._last_flush[job.id] = time.monotonic()

    def _progress(self, job: ConvertJob, *, stage_progress: float | None = None,
                  bytes_done: int | None = None) -> None:
        """Update live progress; written to sqlite at most once per second."""
        if stage_progress is not None:
            job.stage_progress = max(0.0, min(1.0, stage_progress))
        if bytes_done is not None:
            job.bytes_done = bytes_done
        if time.monotonic() - self._last_flush.get(job.id, 0.0) >= _PROGRESS_INTERVAL_S:
            self._persist(job)

    def _set_state(self, job: ConvertJob, state: str, **fields) -> None:
        job.state = state  # type: ignore[assignment]
        job.stage_progress = None
        for k, v in fields.items():
            setattr(job, k, v)
        self._persist(job)

    def _log(self, job: ConvertJob, line: str, ctx: _Ctx | None = None) -> None:
        d = self._logs.setdefault(job.id, deque(maxlen=_LOG_MEM))
        # A progress bar redraws one line many times: keep only its latest state.
        if _is_progress(line) and d and _is_progress(d[-1]):
            d[-1] = line
        else:
            d.append(line)
        if ctx is not None and not _is_progress(line):
            ctx.tail.append(line)

    def _view(self, job: ConvertJob) -> ConvertJob:
        out = job.model_copy(deep=True)
        if job.id in self._logs:
            out.log_tail = list(self._logs[job.id])
        return out

    # ---- public read API -------------------------------------------------------------------

    def available(self) -> str | None:
        return self.toolchain.problem()

    def imatrix_available(self) -> bool:
        return self.toolchain.has_imatrix()

    def list(self) -> list[ConvertJob]:
        return [self._view(j) for j in reversed(self._jobs.values())]

    def get(self, job_id: str) -> ConvertJob | None:
        job = self._jobs.get(job_id)
        return self._view(job) if job else None

    def _need(self, job_id: str) -> ConvertJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise ConvertError(f"no conversion job {job_id}", 404)
        return job

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        """Turn jobs a previous process left running back into queued (their partial outputs are
        discarded by the worker; finished downloads stay in the cache) and start the worker."""
        if self._worker_task is not None:
            return
        for job in self._jobs.values():
            if job.state in ACTIVE_STATES and job.state != "queued":
                job.state = "queued"
                job.stage_progress = None
                job.bytes_done = 0
                self._logs.pop(job.id, None)
                job.log_tail = []
                self._persist(job)
        self._queue = deque(j.id for j in self._jobs.values() if j.state == "queued")
        self._wake = asyncio.Event()
        self._worker_task = asyncio.get_running_loop().create_task(self._worker())

    async def shutdown(self) -> None:
        t, self._worker_task = self._worker_task, None
        if t is not None:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        with self._lock:
            self._db.close()

    # ---- submit / control ------------------------------------------------------------------

    def _name_conflict(self, name: str, exclude: str | None = None) -> str | None:
        if self.library.name_taken(name) or (self.models_dir / name).exists():
            return f"the library already has a model named {name}; choose another output name"
        for j in self._jobs.values():
            if j.id != exclude and j.output_name == name and j.state not in TERMINAL_STATES:
                return f"conversion job {j.id} already produces {name}; choose another output name"
        return None

    async def submit(self, req: ConvertRequest) -> ConvertJob:
        problem = self.toolchain.problem()
        if problem:
            raise ConvertError(problem, 503)
        use_imatrix = self._decide_imatrix(req)
        insp = await inspect_source(
            req.source, hf=self.hf, locate_dir=self.library.locate_dir, cluster=ClusterVram(),
            supported_architectures=await self.toolchain.supported_architectures())
        if insp.supported is False:
            raise ConvertError(
                f"architecture {insp.architecture} is not supported by llama.cpp b11342's converter",
                422)
        if insp.prequantized and insp.prequant_supported is False:
            hint = (f"; convert its base model {insp.base_model} instead" if insp.base_model
                    else "; convert the original unquantized model instead")
            raise ConvertError(
                f"this model is pre-quantized ({insp.prequantized}) and llama.cpp's converter "
                f"cannot read that format{hint}", 422)
        if insp.weight_format == "none":
            raise ConvertError("no safetensors or PyTorch weight files found in the source", 422)
        name = check_output_name(req.name) if req.name else default_output_name(req.source, req.quant)
        est = next((o.est_bytes for o in insp.options if o.type == req.quant), None)
        if est is None and insp.params:
            est = estimate_bytes(insp.params, req.quant)
        await self._early_disk_check(req, insp, name, est)
        # No await between the name check and the insert: two submits cannot both pass.
        conflict = self._name_conflict(name)
        if conflict:
            raise ConvertError(conflict, 409)
        self._seq += 1
        job = ConvertJob(id=uuid.uuid4().hex[:12], request=req, state="queued", output_name=name,
                         est_output_bytes=est, imatrix_used=use_imatrix, created_at=self._clock())
        self._jobs[job.id] = job
        self._plans[job.id] = {"params": insp.params, "source_bytes": insp.source_bytes,
                               "architecture": insp.architecture}
        self._persist(job, seq=self._seq)
        self._enqueue(job.id)
        return self._view(job)

    def _decide_imatrix(self, req: ConvertRequest) -> bool:
        """Does this job compute an importance matrix? Raises ConvertError when the request
        cannot be honoured (type needs one but it is off / llama-imatrix missing / bad text)."""
        adv = req.advanced
        opt = option(req.quant)
        if opt.needs_imatrix and adv.imatrix == "off":
            raise ConvertError(
                f"{req.quant} needs an importance matrix (llama-quantize refuses it without one): "
                "set the importance matrix option to auto or on, or choose a larger type", 422)
        use = imatrix_wanted(req.quant, adv.imatrix)
        if use and not self.toolchain.has_imatrix():
            if opt.needs_imatrix or adv.imatrix == "on":
                raise ConvertError(IMATRIX_NOT_INSTALLED, 503)
            use = False  # auto on a type that only benefits from one: go without
        if use:
            self._check_calibration(adv.calibration_path)
        return use

    def _check_calibration(self, path: str | None) -> None:
        if not path:
            if not BUILTIN_CALIBRATION.is_file():
                raise ConvertError("the built-in calibration text is missing from this install; "
                                   "give a calibration_path or reinstall gpupool", 503)
            return
        try:
            p = Path(self.library.locate_file(path))
        except ConvertError:
            raise
        except Exception as e:
            raise ConvertError(f"calibration text: {getattr(e, 'message', str(e))}", 422) from e
        if p.suffix.lower() != ".txt":
            raise ConvertError(f"calibration text must be a .txt file, got {p.name}", 422)
        try:
            size = p.stat().st_size
        except OSError as e:
            raise ConvertError(f"calibration text cannot be read: {e}", 422) from e
        if size == 0:
            raise ConvertError(f"calibration text {p.name} is empty", 422)
        if size > _CALIBRATION_MAX:
            raise ConvertError(f"calibration text {p.name} is {_gb(size)}; the limit is "
                               f"{_CALIBRATION_MAX // 2**20} MB", 422)

    async def _early_disk_check(self, req: ConvertRequest, insp, name: str, est: int | None) -> None:
        """Refuse a job the disk obviously cannot hold: what is still to download (files already
        in the HF cache do not count) + the 16-bit intermediate + the output + a margin. The
        worker checks again per stage, since space can shrink while a job waits in the queue."""
        spec = req.source
        remaining = 0
        if spec.hf_repo:
            cache = self._cache_dir(spec.hf_repo, spec.revision)
            remaining = await asyncio.to_thread(self._remaining_bytes, cache, insp.files)
        outtype, qtype = plan_steps(req.quant, req.advanced.intermediate, None)
        if qtype is None:
            inter = 0
        elif insp.params:
            inter = insp.params * (4 if outtype == "f32" else 2)
        else:
            inter = insp.source_bytes
        what = (f"downloading {spec.hf_repo} ({_gb(remaining)}) and converting it to {name}"
                if spec.hf_repo else f"converting {spec.path} to {name}")
        await asyncio.to_thread(self._check_disk, remaining + inter + (est or 0) + _DISK_MARGIN, what)

    def _enqueue(self, job_id: str) -> None:
        self._queue.append(job_id)
        if self._wake is not None:
            self._wake.set()

    async def cancel(self, job_id: str) -> ConvertJob:
        job = self._need(job_id)
        if job.state not in ACTIVE_STATES:
            raise ConvertError(f"job is {job.state}; only an active job can be cancelled", 409)
        ctx = self._ctxs.get(job_id)
        if job_id in self._queue:
            self._queue.remove(job_id)
        elif ctx is not None and ctx.task is not None:
            ctx.user_cancel = True
            ctx.stop.set()
            ctx.task.cancel()
            await asyncio.wait({ctx.task})
        if job.state in ACTIVE_STATES:  # task never got to run its own cancel handler
            await self._finalize_cancel(job)
        return self._view(job)

    async def retry(self, job_id: str) -> ConvertJob:
        job = self._need(job_id)
        if job.state not in ("failed", "cancelled"):
            raise ConvertError(f"job is {job.state}; only a failed or cancelled job can be retried", 409)
        conflict = self._name_conflict(job.output_name, exclude=job.id)
        if conflict:
            raise ConvertError(conflict, 409)
        self._logs.pop(job.id, None)
        job.log_tail = []
        job.validation = None
        job.error = None
        job.bytes_done = 0
        job.bytes_total = None
        job.output_bytes = None
        job.failed_stage = None
        job.started_at = None
        job.finished_at = None
        self._set_state(job, "queued")
        self._enqueue(job.id)
        return self._view(job)

    async def accept(self, job_id: str) -> ConvertJob:
        job = self._need(job_id)
        if job.state != "needs_review":
            raise ConvertError(f"job is {job.state}; only a job in needs_review can be accepted", 409)
        src = self._work_dir(job.id) / job.output_name
        if not src.is_file():
            raise ConvertError("the converted file is gone; delete this job and convert again", 409)
        conflict = self._name_conflict(job.output_name, exclude=job.id)
        if conflict:
            raise ConvertError(conflict, 409)
        await self._publish(job, src)
        await self._cleanup_quietly(job, self._work_dir(job.id), release_cache=False)
        self._set_state(job, "done", error=None, finished_at=self._clock())
        return self._view(job)

    async def delete(self, job_id: str) -> None:
        job = self._need(job_id)
        if job.state in ACTIVE_STATES:
            raise ConvertError("cancel the job before deleting it", 409)
        await asyncio.to_thread(self._rmtree, self._work_dir(job.id))
        if job.state in ("failed", "cancelled"):
            await self._release_cache(job)
        with self._lock:
            self._db.execute("DELETE FROM convert_jobs WHERE id=?", (job.id,))
        for d in (self._jobs, self._plans, self._logs, self._ctxs, self._last_flush):
            d.pop(job.id, None)

    # ---- filesystem helpers ----------------------------------------------------------------

    def _work_dir(self, job_id: str) -> Path:
        return self.convert_root / job_id

    def _cache_dir(self, repo: str, revision: str) -> Path:
        if not _REPO_RE.match(repo) or any(p in (".", "..") for p in repo.split("/")):
            raise ConvertError("repo must look like owner/name", 400)
        rev = re.sub(r"[^A-Za-z0-9._-]", "_", revision) or "main"
        return self.hf_root / f"{repo.replace('/', '__')}@{rev}"

    def _rmtree(self, path: Path) -> None:
        """Remove a directory under .convert or .hf, never anything else, never following links."""
        p = Path(os.path.abspath(path))
        ok = any(p != root and root in p.parents
                 for root in (Path(os.path.abspath(self.convert_root)),
                              Path(os.path.abspath(self.hf_root))))
        if not ok:
            log.error("refusing to delete %s: outside the converter folders", p)
            return
        if p.is_symlink():
            p.unlink()
            return
        for attempt in range(4):  # Windows: a just-killed tool may still hold files for a moment
            shutil.rmtree(p, ignore_errors=True)
            if not p.exists():
                return
            time.sleep(0.25 * (attempt + 1))
        log.warning("could not fully remove %s", p)

    def _users_of_cache(self, cache: Path, exclude: str) -> bool:
        for j in self._jobs.values():
            if j.id == exclude or j.state not in ACTIVE_STATES or not j.request.source.hf_repo:
                continue
            try:
                if self._cache_dir(j.request.source.hf_repo, j.request.source.revision) == cache:
                    return True
            except ConvertError:
                continue
        return False

    async def _cleanup_quietly(self, job: ConvertJob, work: Path, *, release_cache: bool) -> None:
        try:
            await asyncio.to_thread(self._rmtree, work)
            if release_cache:
                await self._release_cache(job)
        except Exception:
            log.exception("cleanup after job %s failed", job.id)

    async def _release_cache(self, job: ConvertJob) -> None:
        """Drop the job's HF download cache unless asked to keep it or another job needs it."""
        repo = job.request.source.hf_repo
        if not repo or job.request.keep_source:
            return
        cache = self._cache_dir(repo, job.request.source.revision)
        if not self._users_of_cache(cache, job.id) and cache.exists():
            await asyncio.to_thread(self._rmtree, cache)

    def _check_disk(self, need: int, what: str) -> None:
        free = shutil.disk_usage(self.models_dir).free
        if free < need:
            raise ConvertError(
                f"not enough free disk space in {self.models_dir}: {what} needs about {_gb(need)}, "
                f"only {_gb(free)} is free. Free up space or point the models folder at a "
                "bigger disk.", 507)

    async def _thread(self, ctx: _Ctx, fn: Callable, *args):
        """Run blocking work in a thread; on cancellation the work is still tracked so the
        cancel handler can wait for it before deleting its files."""
        fut = asyncio.get_running_loop().run_in_executor(None, fn, *args)
        ctx.pending.add(fut)
        fut.add_done_callback(ctx.pending.discard)
        return await asyncio.shield(fut)

    # ---- worker ----------------------------------------------------------------------------

    async def _worker(self) -> None:
        assert self._wake is not None
        await asyncio.to_thread(self._sweep)
        while True:
            if not self._queue:
                await self._wake.wait()
                self._wake.clear()
                continue
            job = self._jobs.get(self._queue.popleft())
            if job is None or job.state != "queued":
                continue
            ctx = _Ctx()
            self._ctxs[job.id] = ctx
            self._logs.pop(job.id, None)
            # State changes synchronously with the pop, so cancel() never sees a job that is
            # in neither the queue nor running.
            self._set_state(job, "downloading", started_at=self._clock(), error=None,
                            bytes_done=0)
            ctx.task = asyncio.get_running_loop().create_task(self._run(job.id, ctx))
            try:
                await asyncio.wait({ctx.task})
            except asyncio.CancelledError:
                ctx.task.cancel()
                await asyncio.gather(ctx.task, return_exceptions=True)
                raise
            finally:
                self._ctxs.pop(job.id, None)

    def _sweep(self) -> None:
        """Remove scratch folders nobody owns: leftovers of interrupted/finished jobs."""
        keep = {j.id for j in self._jobs.values() if j.state == "needs_review"}
        try:
            entries = list(self.convert_root.iterdir())
        except OSError:
            return
        for p in entries:
            if p.name not in keep and p.is_dir():
                self._rmtree(p)

    async def _run(self, job_id: str, ctx: _Ctx) -> None:
        job = self._jobs[job_id]
        try:
            await self._pipeline(job, ctx)
        except asyncio.CancelledError:
            await asyncio.gather(*list(ctx.pending), return_exceptions=True)
            if ctx.user_cancel:
                await self._finalize_cancel(job)
            # else: the process is shutting down; the row stays active and start() requeues it
            raise
        except ConvertError as e:
            await self._fail(job, e.message)
        except Exception as e:  # any failure must end in a visible "failed" state
            log.exception("conversion job %s crashed", job.id)
            await self._fail(job, f"{type(e).__name__}: {e}")
        finally:
            await asyncio.gather(*list(ctx.pending), return_exceptions=True)

    async def _fail(self, job: ConvertJob, message: str,
                    validation: Validation | None = None) -> None:
        stage = job.state if job.state in ACTIVE_STATES else None
        await asyncio.to_thread(self._rmtree, self._work_dir(job.id))
        self._set_state(job, "failed", error=message, finished_at=self._clock(),
                        failed_stage=stage, validation=validation or job.validation)

    async def _finalize_cancel(self, job: ConvertJob) -> None:
        if job.state == "cancelled":
            return
        stage = job.state if job.state in ACTIVE_STATES else None
        await asyncio.to_thread(self._rmtree, self._work_dir(job.id))
        self._set_state(job, "cancelled", error=None, finished_at=self._clock(),
                        failed_stage=stage)

    # ---- pipeline --------------------------------------------------------------------------

    async def _pipeline(self, job: ConvertJob, ctx: _Ctx) -> None:
        req = job.request
        adv = req.advanced
        spec = req.source
        plan = self._plans.get(job.id, {})
        work = self._work_dir(job.id)
        src_dir = work / "src"
        await self._thread(ctx, self._fresh_dir, work)

        # --- source files ---
        if spec.hf_repo:
            repo, rev = spec.hf_repo, spec.revision
            cache = self._cache_dir(repo, rev)
            listing = await self.hf.list_files(repo, rev)
            selected, _ = select_files(listing, adv.allow_remote_code)
            origin = cache
        else:
            try:
                origin = Path(self.library.locate_dir(spec.path or ""))
            except ConvertError:
                raise
            except Exception as e:
                raise ConvertError(getattr(e, "message", str(e)), getattr(e, "status", 400)) from e
            selected, _ = select_files(await self._thread(ctx, local_files, origin),
                                       adv.allow_remote_code)
        for f in selected:
            self._check_rel(f.name)
        names = {f.name for f in selected}
        if "config.json" not in names:
            raise ConvertError("the source has no config.json")
        if not any(n.endswith((".safetensors", ".bin")) for n in names):
            raise ConvertError("no safetensors or PyTorch weight files found in the source")
        total = sum(f.bytes for f in selected)
        job.bytes_total = total

        outtype_guess, qtype_guess = plan_steps(req.quant, adv.intermediate, None)
        src_bytes = total or int(plan.get("source_bytes") or 0)
        params = plan.get("params")
        inter_est = 0 if qtype_guess is None else (
            params * (4 if outtype_guess == "f32" else 2) if params else src_bytes)
        out_est = job.est_output_bytes or 0

        # --- download (HF) ---
        if spec.hf_repo:
            remaining = await self._thread(ctx, self._remaining_bytes, cache, selected)
            self._check_disk(remaining + inter_est + out_est + _DISK_MARGIN,
                             f"downloading {spec.hf_repo} ({_gb(remaining)}) and converting it")
            done = 0
            for f in selected:
                dest = cache / f.name

                def on_prog(n: int, base: int = done) -> None:
                    self._progress(job, bytes_done=base + n,
                                   stage_progress=(base + n) / total if total else None)

                got = await self.hf.download(spec.hf_repo, spec.revision, f.name, dest,
                                             f.bytes or None, on_prog)
                done += got if isinstance(got, int) and got > 0 else f.bytes
                self._progress(job, bytes_done=done)
        self._persist(job)
        if adv.imatrix == "on" and not job.imatrix_used and qtype_guess is None:
            self._log(job, f"importance matrix ignored: {req.quant} is written directly by the "
                           "converter, there is no quantize step to apply it to")

        # --- staging: the converter only ever sees links to the selected files ---
        await self._thread(ctx, self._stage, origin, selected, src_dir, ctx.stop)
        self._progress(job, stage_progress=1.0)
        source_dtype = await self._thread(ctx, self._source_dtype, src_dir)
        outtype, qtype = plan_steps(req.quant, adv.intermediate, source_dtype)
        inter_est = 0 if qtype is None else (
            params * (4 if outtype == "f32" else 2) if params else src_bytes)

        # --- convert ---
        self._check_disk(inter_est + out_est + _DISK_MARGIN,
                         f"converting {job.output_name}")
        self._set_state(job, "converting")
        converted = work / f"{outtype}.gguf"
        cmd = self.toolchain.convert_cmd(src_dir, converted, outtype)
        await self._tool(job, ctx, cmd, "convert_hf_to_gguf.py", env=self.toolchain.convert_env(),
                         cwd=self.toolchain.convert_dir)
        if not converted.is_file():
            raise ConvertError("convert_hf_to_gguf.py finished but wrote no output file")
        final = converted

        # --- calibrate: importance matrix of the 16-bit model on the calibration text ---
        imatrix_file: Path | None = None
        if qtype is not None and job.imatrix_used:
            imatrix_file = await self._calibrate(job, ctx, converted, work)

        # --- quantize ---
        if qtype is not None:
            self._check_disk(out_est + _DISK_MARGIN, f"quantizing to {qtype}")
            self._set_state(job, "quantizing")
            final = work / "out.gguf"
            cmd = self.toolchain.quantize_cmd(
                converted, final, qtype, threads=adv.threads or self.threads,
                leave_output_tensor=adv.leave_output_tensor, pure=adv.pure,
                output_tensor_type=adv.output_tensor_type,
                token_embedding_type=adv.token_embedding_type, imatrix=imatrix_file)
            await self._tool(job, ctx, cmd, "llama-quantize")
            if not final.is_file():
                raise ConvertError("llama-quantize finished but wrote no output file")
            await self._thread(ctx, self._unlink, converted)  # free the big intermediate now

        # --- validate ---
        job.output_bytes = final.stat().st_size
        self._set_state(job, "validating")
        self._logs.pop(job.id, None)
        v = await validate(final, src_dir, self.toolchain, generation=adv.validate_generation,
                           allow_remote_code=adv.allow_remote_code)
        job.validation = v
        if v.header_ok is False:
            raise ConvertError("the converted file is not usable: " + "; ".join(v.errors), 422)

        if v.tokenizer_ok is False or v.generation_ok is False:
            kept = work / job.output_name
            await self._thread(ctx, self._keep_only, work, final, kept)
            await self._release_cache(job)
            self._set_state(job, "needs_review", error=self._review_note(v),
                            finished_at=self._clock())
            return

        await self._publish(job, final)
        # Clean up BEFORE reporting done, so an observer of "done" sees a settled disk. The model
        # is already in the library: a cleanup problem must not turn the job into a failure.
        await self._cleanup_quietly(job, work, release_cache=True)
        self._set_state(job, "done", error=None, finished_at=self._clock())

    # ---- pipeline pieces -------------------------------------------------------------------

    async def _calibrate(self, job: ConvertJob, ctx: _Ctx, model: Path, work: Path) -> Path:
        """Run llama-imatrix over `model` (the converted 16-bit file); returns the matrix file."""
        adv = job.request.advanced
        self._check_disk(_DISK_MARGIN, "computing the importance matrix")
        self._set_state(job, "calibrating")
        text = work / "calibration.txt"
        await self._thread(ctx, self._stage_calibration, adv.calibration_path, text)
        out = work / "imatrix.gguf"
        cmd = self.toolchain.imatrix_cmd(model, text, out, chunks=adv.imatrix_chunks or _DEFAULT_CHUNKS,
                                         threads=adv.threads or self.threads)
        await self._tool(job, ctx, cmd, "llama-imatrix", imatrix=ImatrixProgress())
        if not out.is_file() or out.stat().st_size == 0:
            raise ConvertError("llama-imatrix finished but wrote no importance matrix")
        return out

    def _stage_calibration(self, user_path: str | None, dest: Path) -> None:
        """Snapshot the calibration text into the work dir: the file the user pointed at may
        change or vanish while the job runs, and llama-imatrix must see one stable text."""
        if user_path:
            try:
                src = Path(self.library.locate_file(user_path))
            except ConvertError:
                raise
            except Exception as e:
                raise ConvertError(f"calibration text: {getattr(e, 'message', str(e))}", 422) from e
            if src.stat().st_size > _CALIBRATION_MAX:
                raise ConvertError("calibration text is larger than 20 MB", 422)
        else:
            src = BUILTIN_CALIBRATION
            if not src.is_file():
                raise ConvertError("the built-in calibration text is missing from this install")
        shutil.copyfile(src, dest)

    @staticmethod
    def _check_rel(name: str) -> None:
        parts = name.split("/")
        if (not name or name.startswith("/") or "\\" in name or ":" in parts[0]
                or any(p in ("", ".", "..") for p in parts)):
            raise ConvertError(f"refusing unsafe file name in the source: {name!r}")

    def _fresh_dir(self, work: Path) -> None:
        if work.exists():
            self._rmtree(work)
        work.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _remaining_bytes(cache: Path, files: list[SourceFile]) -> int:
        left = 0
        for f in files:
            p = cache / f.name
            try:
                if p.is_file() and f.bytes and p.stat().st_size == f.bytes:
                    continue
            except OSError:
                pass
            left += f.bytes
        return left

    @staticmethod
    def _stage(origin: Path, files: list[SourceFile], dest: Path, stop: threading.Event) -> None:
        """Link (symlink, else hardlink, else copy) exactly the selected files into dest."""
        for f in files:
            if stop.is_set():
                raise RuntimeError("cancelled")
            src = Path(os.path.abspath(origin / f.name))
            dst = dest / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(src, dst)
                continue
            except (OSError, NotImplementedError):
                pass
            try:
                os.link(src, dst)
                continue
            except OSError:
                pass
            shutil.copyfile(src, dst)

    @staticmethod
    def _source_dtype(src_dir: Path) -> str | None:
        try:
            cfg = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        d = cfg.get("torch_dtype") or cfg.get("dtype") if isinstance(cfg, dict) else None
        return d if isinstance(d, str) else None

    @staticmethod
    def _unlink(p: Path) -> None:
        try:
            p.unlink()
        except OSError:
            pass

    def _keep_only(self, work: Path, final: Path, kept: Path) -> None:
        if final != kept:
            _replace(final, kept)
        for p in list(work.iterdir()):
            if p != kept:
                if p.is_dir() and not p.is_symlink():
                    self._rmtree(p)
                else:
                    self._unlink(p)

    @staticmethod
    def _review_note(v: Validation) -> str:
        parts = []
        if v.tokenizer_ok is False:
            bad = sum(1 for c in v.tokenizer_cases if not c.match)
            parts.append(f"the tokenizer differs from Hugging Face on {bad} of "
                         f"{len(v.tokenizer_cases)} test texts")
        if v.generation_ok is False:
            gen = [e for e in v.errors if e.startswith("llama-simple")]
            parts.append(gen[-1] if gen else "the test generation failed")
        return ("Needs review: " + "; ".join(parts) +
                ". Accept to add the model to the library anyway, or delete it.")

    async def _publish(self, job: ConvertJob, src: Path) -> None:
        """Move the finished file to models_dir/<name> and register it."""
        target = self.models_dir / job.output_name
        if target.exists():
            raise ConvertError(f"{target} already exists; refusing to overwrite it", 409)
        await asyncio.to_thread(_replace, src, target)
        try:
            self.library.add_converted(job.output_name, target, job.request.source.hf_repo)
        except BaseException:
            self._unlink(target)  # never leave an unregistered file under a taken name
            raise
        job.output_bytes = target.stat().st_size

    async def _tool(self, job: ConvertJob, ctx: _Ctx, cmd: list[str], label: str, *,
                    env: dict[str, str] | None = None, cwd: Path | None = None,
                    imatrix: ImatrixProgress | None = None) -> None:
        """Run one tool with progress and log tracking; non-zero exit -> ConvertError carrying
        the tool's last meaningful output lines."""
        ctx.tail.clear()
        self._logs.pop(job.id, None)

        def on_line(line: str) -> None:
            self._log(job, line, ctx)
            if imatrix is not None:
                imatrix.feed(line, time.monotonic())
                f = imatrix.fraction(time.monotonic())
                if f is not None:
                    self._progress(job, stage_progress=f)
                return
            m = _TQDM_RE.search(line)
            if m:
                self._progress(job, stage_progress=int(m.group(1)) / 100)
                return
            m = _QUANT_RE.match(line)
            if m and int(m.group(2)) > 0:
                self._progress(job, stage_progress=int(m.group(1)) / int(m.group(2)))

        async def tick() -> None:
            # llama-imatrix is silent between its start-up lines and the end: advance the
            # percentage from its own ETA estimate once a second.
            while True:
                await asyncio.sleep(_PROGRESS_INTERVAL_S)
                f = imatrix.fraction(time.monotonic()) if imatrix is not None else None
                if f is not None:
                    self._progress(job, stage_progress=f)

        ticker = asyncio.ensure_future(tick()) if imatrix is not None else None
        try:
            res = await run_tool(cmd, env=env, cwd=cwd, on_line=on_line)
        finally:
            if ticker is not None:
                ticker.cancel()
        self._persist(job)
        if res.code != 0:
            raise ConvertError(self._tool_error(label, res.code, list(ctx.tail)))

    @staticmethod
    def _tool_error(label: str, code: int | None, tail: list[str]) -> str:
        lines = [ln.strip() for ln in tail if ln.strip()]
        key = next((ln for ln in reversed(lines) if _NOTABLE_RE.search(ln)),
                   lines[-1] if lines else "no output")
        detail = "\n".join(lines[-12:])
        return f"{label} failed (exit code {code}): {key}\n--- last output ---\n{detail}"[:4000]

