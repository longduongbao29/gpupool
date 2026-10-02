"""Model file cache: download to <file>.part, verify size, atomic rename."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import httpx

from gpupool.common.net import INTERNAL, external_sync_kwargs

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _safe_name(name: str) -> str:
    base = os.path.basename(name.replace("\\", "/"))
    if not base or base in (".", ".."):
        raise ValueError(f"cannot derive file name from {name!r}")
    return base


_SPLIT_RE = re.compile(r"^(?P<stem>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def _download(url: str, headers: dict[str, str], client_kw: dict, final: Path) -> int:
    """Fetch one file into `final` (skipped when already there); returns its size."""
    fname = final.name
    with _lock_for(final):
        if final.is_file() and final.stat().st_size > 0:
            return final.stat().st_size
        part = final.with_name(fname + ".part")
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0),
                              **client_kw) as client, client.stream("GET", url, headers=headers) as r:
                r.raise_for_status()
                # With a Content-Encoding (gzip...), raw bytes are the compressed stream:
                # saving them would leave a corrupt .gguf whose length still matches
                # Content-Length. Decode instead, and drop the length check (it counts
                # compressed bytes).
                encoded = r.headers.get("content-encoding", "identity") != "identity"
                expected = None if encoded else r.headers.get("content-length")
                chunks = r.iter_bytes(1024 * 1024) if encoded else r.iter_raw(1024 * 1024)
                got = 0
                with open(part, "wb") as f:
                    for chunk in chunks:
                        f.write(chunk)
                        got += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
            if expected is not None and got != int(expected):
                raise IOError(f"truncated download of {fname}: got {got} of {expected} bytes")
            if got == 0:
                raise IOError(f"empty download of {fname}")
            os.replace(part, final)
        except BaseException:
            try:
                part.unlink()
            except FileNotFoundError:
                pass
            raise
        return final.stat().st_size


def ensure_model(name: str, source: str, cache_dir: Path, coordinator_url: str,
                 token: str) -> tuple[Path, int]:
    headers: dict[str, str] = {}
    # coordinator:// is cluster-internal (never proxied); http(s):// leaves the cluster.
    client_kw: dict = external_sync_kwargs()
    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        fname = _safe_name(unquote(parsed.path))

        def url_of(n: str) -> str:
            head = parsed.path.rsplit("/", 1)[0]
            return parsed._replace(path=f"{head}/{quote(n)}").geturl()
    elif source.startswith("coordinator://"):
        fname = _safe_name(source[len("coordinator://"):])
        base = coordinator_url.rstrip("/")

        def url_of(n: str) -> str:
            return f"{base}/files/{n}"
        headers["Authorization"] = f"Bearer {token}"
        client_kw = dict(INTERNAL)
    else:
        p = Path(source)
        if p.is_absolute() and p.is_file():
            return p, p.stat().st_size
        raise FileNotFoundError(f"model {name!r}: source {source!r} is not a URL and no such absolute file")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    m = _SPLIT_RE.match(fname)
    if not (m and int(m["idx"]) == 1):
        return (cache_dir / fname), _download(url_of(fname), headers, client_kw, cache_dir / fname)

    # Split GGUF: llama-server loads part 1 and needs every other part beside it. Part 1 goes
    # LAST so its presence never implies a complete set (a cache check sees part 1 first, and
    # a half-fetched set must resume instead of looking done).
    total = int(m["total"])
    names = [f"{m['stem']}-{i:05d}-of-{m['total']}.gguf" for i in range(2, total + 1)] + [fname]
    size = sum(_download(url_of(n), headers, client_kw, cache_dir / n) for n in names)
    return cache_dir / fname, size


def list_models(cache_dir: Path) -> list[str]:
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return []
    return sorted(p.name for p in cache_dir.iterdir() if p.is_file() and p.suffix == ".gguf")
