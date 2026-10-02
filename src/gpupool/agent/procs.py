"""Spawn and supervise llama-server / ggml-rpc-server processes."""
from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from gpupool.common.models import EngineSpec, EngineStatus

log = logging.getLogger(__name__)

_EXE = ".exe" if os.name == "nt" else ""


class EngineExists(Exception):
    pass


class PortInUse(Exception):
    pass


def find_binaries(llama_dir: Path) -> dict[str, Path]:
    llama_dir = Path(llama_dir)
    server = llama_dir / f"llama-server{_EXE}"
    rpc = llama_dir / f"ggml-rpc-server{_EXE}"
    if not rpc.is_file():
        rpc = llama_dir / f"rpc-server{_EXE}"  # upstream renamed it
    missing = [n for n, p in (("llama-server", server), ("ggml-rpc-server/rpc-server", rpc))
               if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{', '.join(missing)} not found in llama dir {llama_dir}")
    return {"server": server, "rpc": rpc}


def llama_version(llama_dir: Path) -> str:
    """`llama-server --version` prints 'version: 0.5.0-dev (build 11342, commit f1cee99)'
    (older builds: 'version: 11342 (abc)'). Returns 'b11342', or 'unknown' on any failure."""
    try:
        bins = find_binaries(llama_dir)
        r = subprocess.run([str(bins["server"]), "--version"], capture_output=True, text=True,
                           timeout=30, errors="replace")
        text = (r.stdout or "") + "\n" + (r.stderr or "")
        m = re.search(r"build\s+(\d+)", text) or re.search(r"version:\s*(\d+)\b", text)
        return f"b{m.group(1)}" if m else "unknown"
    except Exception:
        return "unknown"


def build_command(spec: EngineSpec, bins: dict[str, Path], bind_host: str,
                  model_path: str | None) -> list[str]:
    if spec.kind == "rpc":
        return [str(bins["rpc"]), "-H", bind_host, "-p", str(spec.port),
                "-d", spec.devices[0], "-c", *spec.extra_args]
    if not model_path:
        raise ValueError("server engine needs a model_path")
    cmd = [str(bins["server"]), "-m", model_path, "--host", bind_host, "--port", str(spec.port),
           "--alias", spec.model or "", "-c", str(spec.ctx_size), "-np", str(spec.parallel),
           "-ngl", "999"]
    # --rpc MUST precede --device: llama.cpp resolves device names while parsing arguments,
    # so "RPC0" in --device only exists once --rpc has registered the servers (b11342 exits
    # with a usage error otherwise).
    if spec.rpc_endpoints:
        cmd += ["--rpc", ",".join(spec.rpc_endpoints)]
    cmd += ["--device", ",".join(spec.devices), "--split-mode", "layer",
            "--cache-reuse", "256", "--metrics",
            # the scheduler computed the layer split; llama.cpp auto-fit must not change it
            "--fit", "off"]
    if len(spec.devices) > 1:
        cmd += ["--tensor-split", ",".join(f"{x:g}" for x in spec.tensor_split)]
    if spec.cache_type != "f16":
        cmd += ["-ctk", spec.cache_type, "-ctv", spec.cache_type]
    # b11342: without --spec-type, -md loads the draft model but never uses it.
    if spec.spec_type == "ngram":
        cmd += ["--spec-type", "ngram-mod"]
    elif spec.spec_type == "draft":
        if not spec.draft_model_path or not spec.draft_device:
            raise ValueError("draft speculative decoding needs draft_model_path and draft_device")
        cmd += ["--spec-type", "draft-simple", "-md", spec.draft_model_path,
                "-devd", spec.draft_device, "-ngld", "999",
                "--spec-draft-n-max", str(spec.draft_n_max)]
        if spec.cache_type != "f16":
            cmd += ["-ctkd", spec.cache_type, "-ctvd", spec.cache_type]
    return cmd + list(spec.extra_args)


def _tail_lines(path: Path, n: int = 50, block: int = 64 * 1024) -> list[str]:
    """Last n lines, reading only the end of the file (logs can be huge)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            data = b""
            while pos > 0 and data.count(b"\n") <= n:
                step = min(block, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-n:]


def _probe_host(bind_host: str) -> str:
    return "127.0.0.1" if bind_host in ("0.0.0.0", "", "::") else bind_host


@dataclass
class _Engine:
    spec: EngineSpec
    proc: subprocess.Popen
    log_path: Path
    running: bool = False
    stopped: bool = False  # we terminated it on purpose
    lock: threading.Lock = field(default_factory=threading.Lock)


class ProcessManager:
    def __init__(self, llama_dir: Path, log_dir: Path, bind_host: str):
        self.llama_dir = Path(llama_dir)
        self.log_dir = Path(log_dir)
        self.bind_host = bind_host
        self._lock = threading.RLock()
        self._engines: dict[str, _Engine] = {}
        self._bins: dict[str, Path] | None = None
        self.reap_orphans()

    # A hard-killed agent (OOM killer, kill -9, server reboot of the agent only) leaves its
    # llama.cpp children running: they keep holding VRAM on a shared GPU and no coordinator
    # knows about them. Each engine therefore gets a pid file; a new ProcessManager on the
    # same log_dir stops whatever is left. create_time guards against PID reuse.
    def _pid_path(self, engine_id: str) -> Path:
        return self.log_dir / f"{engine_id}.pid"

    def _write_pid(self, engine_id: str, proc: subprocess.Popen) -> None:
        try:
            created = psutil.Process(proc.pid).create_time()
        except psutil.Error:
            return  # already gone; nothing to reap later
        tmp = self._pid_path(engine_id).with_suffix(".pid.tmp")
        tmp.write_text(f"{proc.pid} {created}\n")
        os.replace(tmp, self._pid_path(engine_id))

    def _clear_pid(self, engine_id: str) -> None:
        try:
            self._pid_path(engine_id).unlink()
        except FileNotFoundError:
            pass

    def reap_orphans(self) -> list[str]:
        """Stop engines left running by a previous agent on this log_dir. Returns their ids."""
        reaped: list[str] = []
        if not self.log_dir.is_dir():
            return reaped
        for pid_file in self.log_dir.glob("*.pid"):
            engine_id = pid_file.stem
            try:
                pid_s, created_s = pid_file.read_text().split()
                proc = psutil.Process(int(pid_s))
                if abs(proc.create_time() - float(created_s)) < 1.0:
                    log.warning("stopping orphan engine %s (pid %s) from a previous agent",
                                engine_id, pid_s)
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    reaped.append(engine_id)
            except (ValueError, OSError, psutil.Error):
                pass  # malformed file or process already gone
            pid_file.unlink(missing_ok=True)
        return reaped

    def _binaries(self) -> dict[str, Path]:
        if self._bins is None:
            self._bins = find_binaries(self.llama_dir)
        return self._bins

    def _port_free(self, port: int) -> bool:
        try:
            infos = socket.getaddrinfo(self.bind_host, port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        fam, typ, proto, _, addr = infos[0]
        s = socket.socket(fam, typ, proto)
        try:
            if os.name != "nt":
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(addr)
            return True
        except OSError:
            return False
        finally:
            s.close()

    def start(self, spec: EngineSpec, model_path: str | None = None) -> EngineStatus:
        with self._lock:
            old = self._engines.get(spec.engine_id)
            if old is not None and old.proc.poll() is None:
                raise EngineExists(spec.engine_id)
            if not self._port_free(spec.port):
                raise PortInUse(f"port {spec.port} not available on {self.bind_host}")
            cmd = build_command(spec, self._binaries(), self.bind_host, model_path)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / f"{spec.engine_id}.log"
            env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            # File, not PIPE: an unread pipe fills up and blocks llama.cpp mid-load.
            # The child inherits the handle; the parent's copy is closed on leaving `with`.
            with open(log_path, "wb") as logf:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf,
                                        stderr=subprocess.STDOUT, env=env, **kwargs)
            eng = _Engine(spec=spec, proc=proc, log_path=log_path)
            self._engines[spec.engine_id] = eng
            self._write_pid(spec.engine_id, proc)
        return self._status(eng)

    def _status(self, eng: _Engine, tail_when_running: bool = True) -> EngineStatus:
        with eng.lock:
            code = eng.proc.poll()
            if code is not None:
                state = "exited" if (code == 0 or eng.stopped) else "failed"
            elif eng.running:
                state = "running"
            else:
                try:
                    with socket.create_connection(
                            (_probe_host(self.bind_host), eng.spec.port), timeout=0.2):
                        eng.running = True
                except OSError:
                    pass
                state = "running" if eng.running else "starting"
            return EngineStatus(
                engine_id=eng.spec.engine_id, kind=eng.spec.kind, state=state,
                pid=eng.proc.pid, port=eng.spec.port, exit_code=code,
                log_tail=_tail_lines(eng.log_path)
                if state != "running" or tail_when_running else [])

    def get(self, engine_id: str) -> EngineStatus | None:
        with self._lock:
            eng = self._engines.get(engine_id)
        return self._status(eng) if eng else None

    def list(self) -> list[EngineStatus]:
        with self._lock:
            engs = list(self._engines.values())
        # /report polls this every 2 s: a healthy engine's log tail is dead weight (disk read
        # per engine per poll). Starting/exited/failed keep it: the coordinator reads crash
        # and launch diagnostics from there. get() always returns the full tail.
        return [self._status(e, tail_when_running=False) for e in engs]

    def stop(self, engine_id: str, timeout: float = 10.0) -> EngineStatus | None:
        with self._lock:
            eng = self._engines.get(engine_id)
        if eng is None:
            return None
        if eng.proc.poll() is None:
            eng.stopped = True
            eng.proc.terminate()
            try:
                eng.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning("engine %s ignored terminate; killing", engine_id)
                eng.proc.kill()
                eng.proc.wait()  # reap: no zombie
        self._clear_pid(engine_id)
        final = self._status(eng)
        # Forget it: otherwise every /report lists every engine ever stopped and the
        # coordinator keeps treating their ports as taken until the range is exhausted.
        # Only stop() forgets; a crashed engine stays listed so the coordinator sees
        # exited/failed and then calls stop. The identity check protects a concurrent
        # start() that already replaced this entry. The log stays on disk.
        with self._lock:
            if self._engines.get(engine_id) is eng:
                del self._engines[engine_id]
        return final

    def stop_all(self) -> None:
        with self._lock:
            ids = list(self._engines)
        for i in ids:
            try:
                self.stop(i)
            except Exception:
                log.exception("failed stopping %s", i)
