"""Bearer-token helpers shared by agent and coordinator."""
from __future__ import annotations

import hmac
from collections.abc import Callable, Iterable

from fastapi import Header, HTTPException


def bearer_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _extract(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer "):].strip()


def token_matches(given: str | None, allowed: Iterable[str]) -> bool:
    if given is None:
        return False
    # compare_digest on every candidate: no early exit that leaks which key matched
    ok = False
    for key in allowed:
        if key and hmac.compare_digest(given.encode(), key.encode()):
            ok = True
    return ok


def require_bearer(*allowed: str) -> Callable[..., None]:
    """FastAPI dependency: 401 unless the Authorization header carries one of `allowed`.

    With no non-empty token configured, auth is disabled (dev mode).
    """
    keys = [k for k in allowed if k]

    def dep(authorization: str | None = Header(default=None)) -> None:
        if not keys:
            return
        if not token_matches(_extract(authorization), keys):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    return dep
