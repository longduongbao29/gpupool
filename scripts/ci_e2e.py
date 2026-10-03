"""CI end-to-end test: a real coordinator + 3 real agents (Docker images, CPU only) serve a tiny
model split over several servers through llama.cpp RPC.

Why it exists: Dockerfile errors, a wrong Python in the image or a bad multi-server split passed
the unit tests and only showed up when real images ran. Stdlib only, so it runs on a bare CI host.

    python scripts/ci_e2e.py [--project gpupool-ci] [--port 8080] [--keep]

Environment: GPUPOOL_AGENT_IMAGE / GPUPOOL_COORDINATOR_IMAGE (default gpupool-*:latest),
GPUPOOL_CI_MODELS_DIR (where the GGUF is cached; default <repo>/.cache/ci-models).
Exit code 0 = all checks passed; on failure `docker compose logs` is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.ci.yml"

# bartowski's SmolLM2-135M-Instruct Q8_0 (public, no login). The hash is verified after every
# download so a truncated or swapped file fails loudly instead of confusing the cluster.
MODEL_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
MODEL_FILE = "SmolLM2-135M-Instruct-Q8_0.gguf"
MODEL_BYTES = 144_811_360
MODEL_SHA256 = "5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}"
MODEL_NAME = "smol"
ADMIN_KEY = "ci-admin"  # matches CI_ADMIN_KEY's default in docker-compose.ci.yml

results: list[tuple[str, bool, str]] = []
t_start = time.monotonic()
base = ""


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(f"check failed: {name}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILE
    if path.exists() and path.stat().st_size == MODEL_BYTES and sha256(path) == MODEL_SHA256:
        return path
    print(f"downloading {MODEL_URL}", flush=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as r, tmp.open("wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    if sha256(tmp) != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{MODEL_FILE}: SHA256 mismatch after download")
    tmp.replace(path)
    return path


def api(method: str, path: str, body: dict | None = None, auth: bool = True, timeout: float = 60):
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {ADMIN_KEY}"
    req = urllib.request.Request(base + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {e.read()[:300]!r}") from None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def wait(pred, timeout: float, what: str, every: float = 2.0):
    end = time.monotonic() + timeout
    err = None
    while time.monotonic() < end:
        try:
            v = pred()
            if v:
                return v
        except Exception as e:  # the coordinator may still be starting
            err = e
        time.sleep(every)
    raise SystemExit(f"timed out after {timeout:.0f}s: {what} (last error: {err})")


def state() -> dict:
    return api("GET", "/api/state")


def model() -> dict | None:
    return next((m for m in state()["models"] if m["spec"]["name"] == MODEL_NAME), None)


def live_engines(st: dict) -> list[str]:
    return [f"{s['node_id']}:{e['engine_id']}" for s in st["servers"]
            for e in (s.get("report") or {}).get("engines", []) if e["state"] in ("starting", "running")]


def compose(project: str, *args: str, check_rc: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", "-p", project, "-f", str(COMPOSE), *args],
                          check=check_rc, text=True)


def run_checks() -> None:
    print("== 1. cluster up: 3 CPU-only agents self-join", flush=True)
    t0 = time.monotonic()
    wait(lambda: api("GET", "/healthz", auth=False), 120, "coordinator /healthz")
    st = wait(lambda: (s := state())["summary"]["servers_online"] == 3 and s, 180, "3 servers online")
    check("3 servers online", True, f"{time.monotonic() - t0:.0f}s")
    devs = [(s["node_id"], d["device_id"]) for s in st["servers"]
            for d in (s["report"] or {}).get("devices", [])]
    check("each server reports exactly one CPU device (no NVML, no GPU)",
          len(devs) == 3 and all(d[1].startswith("CPU") for d in devs), str(devs))

    print("== 2. model library", flush=True)
    lib = api("POST", "/api/library", {"path": f"/models/{MODEL_FILE}"})
    check("model registered", lib.get("status") == "ready", str(lib.get("name", lib)))
    item = lib["name"]
    api("PUT", f"/api/models/{MODEL_NAME}", {"file": item, "ctx_size": 512})

    print("== 3. API surface", flush=True)
    cap = api("GET", "/api/capacity")
    check("/api/capacity", isinstance(cap, dict) and bool(cap), f"keys={sorted(cap)[:6]}")
    rec = api("POST", "/api/recommend", {"file": item, "ctx_size": 512})
    check("/api/recommend", isinstance(rec, dict) and bool(rec), f"keys={sorted(rec)[:6]}")
    plan = api("POST", f"/api/models/{MODEL_NAME}/plan")
    nodes = {a["node_id"] for a in plan["assignments"]}
    check("planner splits the model over >= 2 servers", len(nodes) >= 2,
          f"{plan.get('tier')} {[(a['node_id'], a['device_id'], a['layers']) for a in plan['assignments']]}")

    print("== 4. start -> running", flush=True)
    t0 = time.monotonic()
    api("POST", f"/api/models/{MODEL_NAME}/start", {"replicas": 1})

    def running():
        m = model()
        if m and m["state"] == "failed":
            raise SystemExit(f"model failed: {m.get('error')}")
        return m if m and m["state"] == "running" else None
    m = wait(running, 600, "model running", every=3)
    check("model running", True, f"{time.monotonic() - t0:.0f}s")
    ready = [r for r in m["replicas"] if r["state"] == "ready"]
    check("a replica is ready", len(ready) >= 1, str([r["state"] for r in m["replicas"]]))
    asg = ready[0]["placement"]["assignments"]
    nodes = {a["node_id"] for a in asg}
    rpc = [a["rpc_endpoint"] for a in asg if a.get("rpc_endpoint")]
    check("placement spans >= 2 servers", len(nodes) >= 2, f"{sorted(nodes)}")
    check("placement uses RPC endpoints", len(rpc) >= 1, str(rpc))

    print("== 5. chat completion", flush=True)
    t0 = time.monotonic()
    r = api("POST", "/v1/chat/completions", {
        "model": MODEL_NAME, "max_tokens": 24, "temperature": 0,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}]},
        auth=False, timeout=300)
    ans = r["choices"][0]["message"]["content"]
    check("non-empty answer through the router", bool(ans.strip()),
          f"{time.monotonic() - t0:.1f}s, {ans.strip()[:80]!r}")

    print("== 6. metrics", flush=True)
    text = api("GET", "/metrics", auth=False)
    check("/metrics responds with gpupool series", isinstance(text, str) and "gpupool_" in text,
          f"{len(text)} bytes")

    print("== 7. stop -> engines gone", flush=True)
    api("POST", f"/api/models/{MODEL_NAME}/stop")
    wait(lambda: (model() or {}).get("state") == "stopped", 180, "model stopped")
    wait(lambda: not live_engines(state()), 180, "engines gone from every agent report")
    check("no engine left on any agent", not live_engines(state()))


def main() -> int:
    global base
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="gpupool-ci")
    ap.add_argument("--port", type=int, default=8080, help="coordinator port on the host")
    ap.add_argument("--keep", action="store_true", help="leave the cluster running")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    models_dir = Path(os.environ.get("GPUPOOL_CI_MODELS_DIR", ROOT / ".cache" / "ci-models")).resolve()
    os.environ["GPUPOOL_HOST_MODELS_DIR"] = str(models_dir)
    os.environ["CI_COORDINATOR_PORT"] = str(args.port)
    ok = False
    try:
        path = ensure_model(models_dir)
        check("model file present and verified", True, f"{path.name} {path.stat().st_size} B sha256 ok")
        compose(args.project, "up", "-d")
        run_checks()
        ok = True
    except BaseException as e:
        print(f"\nFAILED: {e}", file=sys.stderr, flush=True)
        compose(args.project, "ps", "-a", check_rc=False)
        compose(args.project, "logs", "--no-color", "--tail", "300", check_rc=False)
    finally:
        if not args.keep:
            compose(args.project, "down", "-v", "--remove-orphans", check_rc=False)
    print(f"\n{sum(o for _, o, _ in results)}/{len(results)} checks passed in "
          f"{time.monotonic() - t_start:.0f}s -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
