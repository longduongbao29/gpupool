"""Keep ggml-rpc-server's weight cache under a size cap.

`ggml-rpc-server -c` stores every weight tensor above 10 MiB it receives as a file named by its
hash under $LLAMA_CACHE/rpc, so the next load of the same model skips the transfer. llama.cpp never
deletes them: every model ever split onto this server would stay on disk. Least recently used
files go first. Deleting one is always safe: a server that misses a hash answers "not cached" and
the head sends the tensor again (rpc_server::set_tensor_hash).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def rpc_cache_dir(llama_cache: Path) -> Path:
    """Where the engines' rpc-servers cache: a LLAMA_CACHE set by the user wins (procs.py)."""
    return Path(os.environ.get("LLAMA_CACHE") or llama_cache) / "rpc"


def prune(cache_dir: Path, max_bytes: int) -> list[str]:
    """Delete least recently used files until the directory holds at most max_bytes.
    Returns the names deleted. max_bytes <= 0 means no cap."""
    if max_bytes <= 0 or not cache_dir.is_dir():
        return []
    files = []
    for p in cache_dir.iterdir():
        try:
            st = p.stat()
        except OSError:
            continue  # deleted meanwhile
        if p.is_file():
            # atime is coarse under relatime but a cache hit reads the file; mtime covers noatime
            files.append((max(st.st_atime, st.st_mtime), st.st_size, p))
    total = sum(f[1] for f in files)
    removed: list[str] = []
    for _, size, p in sorted(files, key=lambda f: f[0]):
        if total <= max_bytes:
            break
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("rpc cache: cannot delete %s: %s", p, e)
            continue
        total -= size
        removed.append(p.name)
    if removed:
        log.info("rpc cache: deleted %d least recently used tensor files, %.1f GB left",
                 len(removed), total / 1e9)
    return removed
