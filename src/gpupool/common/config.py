"""Configuration for agent and coordinator. Loaded from TOML, overridable from the CLI."""
from __future__ import annotations

import json
import os
import tomllib
import typing
from pathlib import Path

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    node_id: str
    host: str = "127.0.0.1"  # address other nodes reach us on; engines bind here
    port: int = 7070
    coordinator_url: str = "http://127.0.0.1:8080"
    cluster_token: str = ""
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
