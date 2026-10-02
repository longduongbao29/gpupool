"""Configuration for agent and coordinator. Loaded from TOML, overridable from the CLI."""
from __future__ import annotations

import contextlib
import json
import logging
import re
import secrets
import socket
import os
import tomllib
import typing
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    node_id: str = ""  # empty = this machine's hostname (resolved by resolve_agent_identity)
    host: str = ""  # address other nodes reach us on; engines bind here. Empty = auto-detect
    port: int = 7070
    coordinator_url: str = "http://127.0.0.1:8080"
    cluster_token: str = ""
    # "<coordinator_url>#<cluster_token>": one string that replaces both fields above
    # (explicit coordinator_url / cluster_token win over it).
    join: str = ""
    # Register this server with the coordinator by itself once the agent API is up.
    auto_join: bool = True
    llama_dir: Path  # directory holding llama-server and (ggml-)rpc-server
    cache_dir: Path = Path(".gpupool/cache")  # GGUF files
    log_dir: Path = Path(".gpupool/logs")  # one log file per engine
    margin_pct: float = 0.10  # keep this fraction of total VRAM free for other users
    margin_min_mb: int = 512  # ...but never less than this
    # Cap per device id, e.g. {"CUDA0": 1200, "CPU": 2000}. Also how CPU devices get a size.
    budget_mb: dict[str, int] = Field(default_factory=dict)
    include_cpu: bool = False  # expose a "CPU" device (served through rpc-server -d CPU)
    heartbeat_s: float = 2.0
    # The coordinator pulls /report from registered agents; pushing heartbeats is optional
    # (kept for coordinators that predate pull mode).
    push_heartbeat: bool = False


class CoordinatorConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    db_path: Path = Path(".gpupool/coordinator.db")
    cluster_token: str = ""  # agents <-> coordinator
    admin_key: str = ""  # /admin/*
    api_keys: list[str] = Field(default_factory=list)  # /v1/*; empty = open
    models_dir: Path = Path(".gpupool/models")  # served at /files/<name> for coordinator:// sources
    heartbeat_timeout_s: float = 10.0
    reconcile_s: float = 2.0
    max_request_mb: int = 32  # cap on one /v1 request body; larger -> 413
    # A request for a model scaled to zero waits this long for it to load, then gets 503.
    cold_start_timeout_s: float = 120.0
    # How often the rebalancer looks for replicas with a clearly better placement and moves one
    # (make-before-break). 0 disables the periodic run; POST /api/rebalance still works.
    rebalance_s: float = 600.0
    launch_timeout_s: float = 600.0
    low_free_mb: int = 256  # device free below this while hosting an engine -> move replica
    drain_timeout_s: float = 60.0
    port_range: tuple[int, int] = (9000, 9999)  # ports handed to engines on agents
    poll_s: float = 2.0  # how often registered agents are polled for /report
    hf_token: str = ""  # Hugging Face token for gated/private repos (env HF_TOKEN)
    # URL agents use to reach this coordinator; shown in the UI's agent install command.
    public_url: str = ""
    # Optional webhook (Slack/Discord/generic JSON POST) for warning and error events, so an
    # operator hears about a dead server even with the UI closed.
    webhook_url: str = ""


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def env_overrides(cls: type[BaseModel], environ: dict[str, str] | None = None,
                  prefix: str = "GPUPOOL_") -> dict:
    """Config values from environment variables: field `api_keys` <- GPUPOOL_API_KEYS.

    Lists are comma-separated, dicts are JSON, tuples are "a-b" or comma-separated,
    booleans accept 1/0/true/false. HF_TOKEN is also read for `hf_token`. Used by the
    Docker images, where a config file is awkward.
    """
    env = os.environ if environ is None else environ
    out: dict = {}
    for name, field in cls.model_fields.items():
        raw = env.get(prefix + name.upper())
        if raw is None and name == "hf_token":
            raw = env.get("HF_TOKEN")
        if raw is None:
            continue
        origin = typing.get_origin(field.annotation)
        if origin is list:
            out[name] = [x.strip() for x in raw.split(",") if x.strip()]
        elif origin is dict:
            out[name] = json.loads(raw) if raw.strip() else {}
        elif origin is tuple:
            out[name] = tuple(int(x) for x in raw.replace("-", ",").split(","))
        else:
            out[name] = raw  # pydantic coerces str -> int/float/bool/Path
    return out


# ---------------------------------------------------------------- one-command deployment helpers

def parse_join(join: str) -> tuple[str, str]:
    """Split "<coordinator_url>#<cluster_token>" into (url, token). A bare URL has no token."""
    url, _, token = join.strip().partition("#")
    return url.strip().rstrip("/"), token.strip()


def detect_local_ip(target_host: str = "8.8.8.8", target_port: int = 80) -> str:
    """The local IP the OS would use to reach target_host.

    A UDP socket "connected" to an address sends nothing; it only makes the kernel pick the
    outgoing interface, which getsockname() then reveals. Falls back to 127.0.0.1.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((target_host, target_port))
            ip = s.getsockname()[0]
        return ip if ip and ip != "0.0.0.0" else "127.0.0.1"
    except OSError:
        return "127.0.0.1"


def sanitize_node_id(raw: str) -> str:
    out = re.sub(r"[^a-z0-9._-]", "-", raw.strip().lower()).strip("-.")
    return out[:64] or "gpu-node"


def build_agent_config(data: dict) -> AgentConfig:
    """AgentConfig from merged TOML/env/CLI data, expanding `join` and resolving the defaults.

    Explicit coordinator_url / cluster_token win over `join`. An empty node_id becomes the
    sanitized hostname; an empty host becomes the local IP used to reach the coordinator.
    Resolved values are logged so the operator can see what was chosen.
    """
    data = dict(data)
    join = data.get("join") or ""
    if join:
        url, token = parse_join(join)
        if url and not data.get("coordinator_url"):
            data["coordinator_url"] = url
        if token and not data.get("cluster_token"):
            data["cluster_token"] = token
    cfg = AgentConfig(**data)
    updates: dict = {}
    if not cfg.node_id:
        updates["node_id"] = sanitize_node_id(socket.gethostname())
        log.info("node_id not set: using hostname %r", updates["node_id"])
    if not cfg.host:
        u = urlparse(cfg.coordinator_url)
        updates["host"] = detect_local_ip(u.hostname or "8.8.8.8", u.port or (443 if u.scheme == "https" else 80))
        log.info("host not set: using %s (the local address used to reach the coordinator)", updates["host"])
    return cfg.model_copy(update=updates) if updates else cfg


def load_or_create_secrets(db_path: Path, admin_key: str, cluster_token: str) -> tuple[str, str]:
    """Fill in an empty admin_key / cluster_token from <db dir>/secrets.json, generating
    and persisting new ones when missing. Values passed in (env/TOML) always win.

    The file is written atomically (tmp + replace) so a crash cannot leave a truncated
    secrets file that would silently regenerate keys and lock out every agent.
    """
    if admin_key and cluster_token:
        return admin_key, cluster_token
    path = Path(db_path).parent / "secrets.json"
    stored: dict = {}
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            # Do not overwrite an unreadable file: that would destroy keys the operator may recover.
            raise RuntimeError(f"cannot read {path}: {e}; fix or delete it") from e
    changed = False
    for name in ("admin_key", "cluster_token"):
        if not stored.get(name):
            stored[name] = secrets.token_urlsafe(18)
            changed = True
    if changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(stored, indent=2), encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    return admin_key or stored["admin_key"], cluster_token or stored["cluster_token"]
