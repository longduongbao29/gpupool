"""End-to-end check of the 0.2 features on one machine, through the same API the web UI uses.

Emulates three servers (agents bound to 127.0.0.1/.2/.3) that join by themselves:
  a: real CUDA0 capped at 900 MB (room for exactly one 0.5B replica, so a second one
     must land on b or c and the failover step has something to kill)
  b, c: CPU devices (1000 MB each) behind a real ggml-rpc-server

Steps (each asserts and prints what happened):
  1. coordinator starts with no keys (generated) and agents join with only --join
  2. a GPU switched off in the pool is not used by the planner
  3. a model downloaded from Hugging Face, another registered by path
  4. Start -> running -> chat answers -> Stop -> engines gone
  5. failover: 2 replicas, server b killed mid-use -> requests keep working,
     replica re-allocated, events node_offline / realloc_started / realloc_done
  6. not enough capacity left after c also dies -> realloc_failed event
  7. a server deleted in the UI cannot re-join by itself

Usage: uv run python scripts/e2e_v2.py   (needs ~3 GB free RAM, internet for step 3)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".cache" / "e2e-v2"
LLAMA = ROOT / ".cache" / "llama" / "b11342-cuda12.4"
PORT = 8085
COORD = f"http://127.0.0.1:{PORT}"
HF_REPO, HF_FILE = "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"
LOCAL_FILE = ROOT / ".cache" / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

NODES = {
    "a": {"host": "127.0.0.1", "port": 7171, "env": {"GPUPOOL_BUDGET_MB": '{"CUDA0": 900}'}},
    "b": {"host": "127.0.0.2", "port": 7172, "env": {
        "GPUPOOL_BUDGET_MB": '{"CUDA0": 0, "CPU": 1000}', "GPUPOOL_INCLUDE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "-1"}},
    "c": {"host": "127.0.0.3", "port": 7173, "env": {
        "GPUPOOL_BUDGET_MB": '{"CUDA0": 0, "CPU": 1000}', "GPUPOOL_INCLUDE_CPU": "true",
        "CUDA_VISIBLE_DEVICES": "-1"}},
}

procs: dict[str, subprocess.Popen] = {}
H: dict[str, str] = {}
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(f"step failed: {name}")


def spawn(name: str, args: list[str], env: dict | None = None) -> None:
    log = open(WORK / f"{name}.log", "wb")
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" \
        else {"start_new_session": True}
    procs[name] = subprocess.Popen([sys.executable, "-m", "gpupool.cli", *args], stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   env={**os.environ, **(env or {})}, cwd=WORK, **kw)
    log.close()


def kill_tree(name: str) -> None:
    p = procs[name]
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(p.pid, signal.SIGKILL)
    p.wait()


def stop_all() -> None:
    for name, p in reversed(list(procs.items())):
        if p.poll() is None:
            p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                kill_tree(name)


def wait(pred, timeout: float, what: str, every: float = 1.0):
    end = time.monotonic() + timeout
    err = None
    while time.monotonic() < end:
        try:
            v = pred()
            if v:
                return v
        except Exception as e:
            err = e
        time.sleep(every)
    raise TimeoutError(f"timed out: {what} (last error: {err})")


def api(method: str, path: str, **kw) -> httpx.Response:
    return httpx.request(method, COORD + path, headers=H, timeout=60, **kw)


def state() -> dict:
    return api("GET", "/api/state").json()


def model(name: str) -> dict | None:
    return next((m for m in state()["models"] if m["spec"]["name"] == name), None)


def events(kind: str) -> list[dict]:
    return [e for e in api("GET", "/api/events?limit=500").json()["events"] if e["kind"] == kind]


def chat(name: str, text: str, n: int = 16) -> str:
    r = httpx.post(f"{COORD}/v1/chat/completions", timeout=120, json={
        "model": name, "max_tokens": n, "temperature": 0,
        "messages": [{"role": "user", "content": text}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> int:
    try:
        httpx.get(COORD, timeout=2)
        raise SystemExit(f"something already listens on {COORD}")
    except (httpx.ConnectError, httpx.ConnectTimeout):
        pass
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    print("== 1. one-command start: generated keys, self-joining agents", flush=True)
    spawn("coordinator", ["coordinator", "--port", str(PORT)], {
        "GPUPOOL_DB_PATH": str(WORK / "data" / "coordinator.db"),
        "GPUPOOL_MODELS_DIR": str(WORK / "data" / "models"),
        "GPUPOOL_HEARTBEAT_TIMEOUT_S": "6", "GPUPOOL_POLL_S": "1"})
    banner = wait(lambda: re.search(r"Admin key:\s+(\S+)", (WORK / "coordinator.log").read_text(
        errors="replace")), 60, "banner")
    H["Authorization"] = f"Bearer {banner.group(1)}"
    join = re.search(r"GPUPOOL_JOIN='([^']+)'", (WORK / "coordinator.log").read_text(errors="replace"))
    check("banner prints admin key and join string", bool(join), join.group(1)[:40] + "..." if join else "")
    secrets = json.loads((WORK / "data" / "secrets.json").read_text())
    check("secrets persisted next to the database", set(secrets) >= {"admin_key", "cluster_token"})
    # The banner shows this machine's LAN IP; for the emulation all agents reach 127.0.0.1.
    join_str = f"http://127.0.0.1:{PORT}#{secrets['cluster_token']}"
    for nid, n in NODES.items():
        d = WORK / nid
        spawn(f"agent-{nid}", ["agent", "--join", join_str, "--llama-dir", str(LLAMA),
                               "--port", str(n["port"]), "--node-id", nid, "--host", n["host"]],
              {**n["env"], "GPUPOOL_CACHE_DIR": str(d / "cache"), "GPUPOOL_LOG_DIR": str(d / "logs")})
    t0 = time.monotonic()
    wait(lambda: state()["summary"]["servers_online"] == 3, 60, "3 servers joined")
    check("3 servers joined by themselves", True, f"{time.monotonic() - t0:.1f}s, no UI step")
    check("server_added events", len(events("server_added")) == 3)

    print("== 2. GPU switch", flush=True)
    lib = api("POST", "/api/library", json={"path": str(LOCAL_FILE)}).json()
    check("model registered by path", lib.get("status") == "ready", lib.get("name", lib))
    api("PUT", "/api/models/probe", json={"file": LOCAL_FILE.name, "ctx_size": 2048})
    plan = api("POST", "/api/models/probe/plan").json()
    check("planner uses the GPU when enabled", plan["assignments"][0]["device_id"] == "CUDA0",
          f"{plan['tier']} {[(a['node_id'], a['device_id']) for a in plan['assignments']]}")
    api("PUT", "/api/servers/a/gpus/CUDA0", json={"enabled": False})
    plan = api("POST", "/api/models/probe/plan").json()
    check("disabled GPU is not used", all(a["device_id"] != "CUDA0" for a in plan["assignments"]),
          f"{[(a['node_id'], a['device_id']) for a in plan['assignments']]}")
    api("PUT", "/api/servers/a/gpus/CUDA0", json={"enabled": True})
    api("DELETE", "/api/models/probe")

    print("== 3. Hugging Face download", flush=True)
    files = api("GET", f"/api/hf/files?repo={HF_REPO}").json()
    check("HF repo listing", any(f["file"] == HF_FILE for f in files), f"{len(files)} gguf files")
    # The path item above has the same file name; remove it so the HF item can take the name.
    api("DELETE", f"/api/library/{LOCAL_FILE.name}")
    r = api("POST", "/api/library", json={"hf_repo": HF_REPO, "hf_file": HF_FILE})
    check("HF download started", r.status_code == 200, r.text[:120])
    seen = set()

    def done():
        it = next(i for i in state()["library"] if i["name"] == HF_FILE)
        if it["bytes"]:
            seen.add(round(100 * it["downloaded"] / it["bytes"]))
        return it if it["status"] in ("ready", "failed") else None
    t0 = time.monotonic()
    it = wait(done, 900, "HF download", every=2)
    check("HF download finished", it["status"] == "ready",
          f"{it['bytes'] / 2**20:.0f} MiB in {time.monotonic() - t0:.0f}s, progress seen {sorted(seen)[:6]}...")

    print("== 4. Start / chat / Stop", flush=True)
    api("PUT", "/api/models/qwen05", json={"file": HF_FILE, "ctx_size": 2048})
    t0 = time.monotonic()
    api("POST", "/api/models/qwen05/start", json={"replicas": 1})
    m = wait(lambda: (lambda m: m if m["state"] in ("running", "failed") else None)(model("qwen05")),
             300, "qwen05 running")
    check("Start -> running", m["state"] == "running", f"{time.monotonic() - t0:.1f}s; error={m['error']}")
    ans = chat("qwen05", "What is the capital of France? One word.")
    check("chat through the router", "paris" in ans.lower(), repr(ans))
    api("POST", "/api/models/qwen05/stop")
    wait(lambda: model("qwen05")["state"] == "stopped", 120, "qwen05 stopped")
    engines = [e for s in state()["servers"] for e in (s["report"] or {}).get("engines", [])
               if e["state"] in ("starting", "running")]
    check("Stop -> engines gone", not engines, f"{len(engines)} engines still running")

    print("== 5. failover: server b dies mid-use", flush=True)
    api("POST", "/api/models/qwen05/start", json={"replicas": 2})
    wait(lambda: len([r for r in model("qwen05")["replicas"] if r["state"] == "ready"]) == 2, 600,
         "2 replicas ready", every=2)
    ready = [r for r in model("qwen05")["replicas"] if r["state"] == "ready"]
    used = {r["replica_id"]: sorted({a["node_id"] for a in r["placement"]["assignments"]}) for r in ready}
    print(f"  replicas and the servers they use: {used}", flush=True)
    victims = sorted({n for nodes in used.values() for n in nodes} - {"a"})
    check("a replica uses a server other than a", bool(victims), str(used))
    victim = victims[0]
    t_kill = time.monotonic()
    kill_tree(f"agent-{victim}")
    ok = fail = 0
    while time.monotonic() - t_kill < 25:
        try:
            chat("qwen05", f"Say {ok}.", 4)
            ok += 1
        except Exception:
            fail += 1
    check("requests during the outage", fail == 0, f"ok={ok} failed={fail}")
    wait(lambda: events("node_offline"), 60, "node_offline event")
    wait(lambda: events("realloc_done"), 600, "realloc_done event", every=2)
    t_rec = time.monotonic() - t_kill
    for kind in ("node_offline", "realloc_started", "realloc_done"):
        ev = events(kind)
        check(f"event {kind}", bool(ev), ev[0]["message"][:110] if ev else "")
    nodes_now = sorted({a["node_id"] for r in model("qwen05")["replicas"] if r["state"] == "ready"
                        for a in r["placement"]["assignments"]})
    check("re-allocated off the dead server", victim not in nodes_now, f"now on {nodes_now}, {t_rec:.0f}s after the kill")

    print("== 6. capacity exhausted", flush=True)
    other = next(n for n in ("b", "c") if n != victim)
    kill_tree(f"agent-{other}")
    ev = wait(lambda: events("realloc_failed"), 120, "realloc_failed event", every=2)
    check("not-enough-capacity event", True, ev[0]["message"][:140])
    m = model("qwen05")
    check("model still served by the surviving replica",
          any(r["state"] == "ready" for r in m["replicas"]) and "paris" in chat("qwen05", "Capital of France? One word.").lower())

    print("== 7. a deleted server cannot re-join by itself", flush=True)
    api("DELETE", f"/api/servers/{other}")
    n = NODES[other]
    d = WORK / other
    spawn(f"agent-{other}", ["agent", "--join", join_str, "--llama-dir", str(LLAMA),
                             "--port", str(n["port"]), "--node-id", other, "--host", n["host"]],
          {**n["env"], "GPUPOOL_CACHE_DIR": str(d / "cache"), "GPUPOOL_LOG_DIR": str(d / "logs")})
    time.sleep(10)
    check("removed server stays removed", all(s["node_id"] != other for s in state()["servers"]))
    log = (WORK / f"agent-{other}.log").read_text(errors="replace")
    check("agent explains why", "removed" in log.lower(), next((l for l in log.splitlines() if "removed" in l.lower()), "")[:120])
    unread = state()["unread_events"]
    print(f"\nunread events in the UI bell: {unread}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        stop_all()
        (WORK / "results.json").write_text(json.dumps(results, indent=1))
        print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} checks passed; logs in {WORK}")
    sys.exit(code)
