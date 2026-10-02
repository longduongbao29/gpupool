"""gpupool command line.

  gpupool agent --join "http://10.0.0.1:8080#<cluster_token>" --llama-dir /path/to/bin
  gpupool agent --config agent.toml [--node-id ... --port ...]
  gpupool coordinator --config coordinator.toml
  gpupool register NAME SOURCE [--ctx 4096 --parallel 1 --replicas 1]
  gpupool plan NAME            # dry-run placement
  gpupool scale NAME N
  gpupool undeploy NAME
  gpupool status

Admin commands talk to the coordinator at $GPUPOOL_URL (default http://127.0.0.1:8080)
with $GPUPOOL_ADMIN_KEY.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import httpx

from gpupool.common.auth import bearer_headers
from gpupool.common.net import describe_proxy, normalize_proxy_env
from gpupool.common.config import (AgentConfig, CoordinatorConfig, build_agent_config, env_overrides,
                                   load_toml)


log = logging.getLogger("gpupool.cli")


def _merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _cmd_agent(args: argparse.Namespace) -> int:
    from gpupool.agent.app import run_agent

    # precedence: CLI flags > GPUPOOL_* environment > TOML file > defaults
    data = load_toml(Path(args.config)) if args.config else {}
    data = _merge(data, env_overrides(AgentConfig))
    data = _merge(data, {"node_id": args.node_id, "host": args.host, "port": args.port,
                         "coordinator_url": args.coordinator, "llama_dir": args.llama_dir,
                         "join": args.join, "auto_join": False if args.no_auto_join else None})
    cfg = build_agent_config(data)  # expands --join, resolves node_id / host defaults
    log.info("agent %s on %s:%d, coordinator %s", cfg.node_id, cfg.host, cfg.port, cfg.coordinator_url)
    if os.environ.get("http_proxy") or os.environ.get("https_proxy") or os.environ.get("all_proxy"):
        log.info("%s", describe_proxy())
    run_agent(cfg)
    return 0


def _cmd_coordinator(args: argparse.Namespace) -> int:
    from gpupool.coordinator.app import run_coordinator

    data = load_toml(Path(args.config)) if args.config else {}
    data = _merge(data, env_overrides(CoordinatorConfig))
    data = _merge(data, {"host": args.host, "port": args.port})
    if os.environ.get("http_proxy") or os.environ.get("https_proxy") or os.environ.get("all_proxy"):
        log.info("%s", describe_proxy())
    run_coordinator(CoordinatorConfig(**data))
    return 0


def _admin(method: str, path: str, **kw) -> int:
    url = os.environ.get("GPUPOOL_URL", "http://127.0.0.1:8080").rstrip("/")
    key = os.environ.get("GPUPOOL_ADMIN_KEY", "")
    try:
        # the coordinator is cluster-internal: never through the proxy
        with httpx.Client(trust_env=False, timeout=60) as c:
            r = c.request(method, url + path, headers=bearer_headers(key) if key else {}, **kw)
    except httpx.HTTPError as e:
        print(f"cannot reach coordinator at {url}: {e}", file=sys.stderr)
        return 2
    try:
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    except ValueError:
        print(r.text)
    return 0 if r.is_success else 1


def main(argv: list[str] | None = None) -> int:
    normalize_proxy_env()  # lowercase and uppercase spellings agree for us and for child processes
    p = argparse.ArgumentParser(prog="gpupool")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("agent", help="run the node agent")
    a.add_argument("--config")
    a.add_argument("--node-id")
    a.add_argument("--host")
    a.add_argument("--port", type=int)
    a.add_argument("--coordinator")
    a.add_argument("--llama-dir")
    a.add_argument("--join", help="<coordinator_url>#<cluster_token> (env GPUPOOL_JOIN)")
    a.add_argument("--no-auto-join", action="store_true", help="do not register with the coordinator by itself")
    a.set_defaults(fn=_cmd_agent)

    c = sub.add_parser("coordinator", help="run the coordinator (scheduler + router)")
    c.add_argument("--config")
    c.add_argument("--host")
    c.add_argument("--port", type=int)
    c.set_defaults(fn=_cmd_coordinator)

    r = sub.add_parser("register", help="register a model (and desired replicas)")
    r.add_argument("name")
    r.add_argument("source", help="https URL, absolute path on the coordinator, or coordinator://file")
    r.add_argument("--ctx", type=int, default=4096)
    r.add_argument("--parallel", type=int, default=1)
    r.add_argument("--replicas", type=int, default=1)
    r.set_defaults(fn=lambda x: _admin("POST", "/admin/models", json={
        "name": x.name, "source": x.source, "ctx_size": x.ctx,
        "parallel": x.parallel, "replicas": x.replicas}))

    pl = sub.add_parser("plan", help="dry-run placement for a registered model")
    pl.add_argument("name")
    pl.set_defaults(fn=lambda x: _admin("POST", f"/admin/deploy/{x.name}", params={"dry_run": 1}))

    sc = sub.add_parser("scale", help="set desired replicas")
    sc.add_argument("name")
    sc.add_argument("replicas", type=int)
    sc.set_defaults(fn=lambda x: _admin("POST", f"/admin/models/{x.name}/scale",
                                        params={"replicas": x.replicas}))

    u = sub.add_parser("undeploy", help="drain all replicas and remove the model")
    u.add_argument("name")
    u.set_defaults(fn=lambda x: _admin("DELETE", f"/admin/models/{x.name}"))

    st = sub.add_parser("status", help="nodes, devices, replicas")
    st.set_defaults(fn=lambda x: _admin("GET", "/admin/status"))

    args = p.parse_args(argv)
    logging.basicConfig(level=os.environ.get("GPUPOOL_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
