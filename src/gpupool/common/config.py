"""Configuration for agent and coordinator. Loaded from TOML, overridable from the CLI."""
from __future__ import annotations

import tomllib
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


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
