"""End-to-end run on one machine against real llama.cpp.

Emulates three servers with three agents bound to 127.0.0.1/.2/.3:
  a: real CUDA0, capped by budget_mb so big models must split
  b, c: CPU devices served by a real ggml-rpc-server (real RPC over TCP)

Scenarios:
  1. 0.5B model fits one GPU -> single_gpu; measure TTFT and decode speed via the router.
  2. 3B model does not fit a's budget -> multi-node split over RPC; check the answer.
  3. Failover: 0.5B with 2 replicas, kill node b's whole process tree, keep sending
     requests, expect no client-visible failure and the replica re-placed on c.

Usage:  uv run python scripts/e2e_local.py [--llama-dir DIR] [--models-dir DIR] [--only 1,2,3]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".cache" / "e2e"
COORD = "http://127.0.0.1:8080"
ADMIN = {"Authorization": "Bearer admin-secret"}
API = {"Authorization": "Bearer api-secret"}
SMALL = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
BIG = "qwen2.5-3b-instruct-q4_k_m.gguf"

NODES = {
    "a": {"host": "127.0.0.1", "port": 7071, "budget_mb": {"CUDA0": 1200}, "include_cpu": False},
    "b": {"host": "127.0.0.2", "port": 7072, "budget_mb": {"CUDA0": 0, "CPU": 1000}, "include_cpu": True},
    "c": {"host": "127.0.0.3", "port": 7073, "budget_mb": {"CUDA0": 0, "CPU": 1000}, "include_cpu": True},
}
# Nodes b and c share the one physical GPU with a: budget CUDA0=0 removes it from their
# pool (NVML ignores CUDA_VISIBLE_DEVICES), and the env hides it from their engines.
CPU_ONLY_ENV = {"CUDA_VISIBLE_DEVICES": "-1"}


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, dict):
        return "{ " + ", ".join(f'{k} = {toml_value(x)}' for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def write_toml(path: Path, data: dict) -> None:
    path.write_text("\n".join(f"{k} = {toml_value(v)}" for k, v in data.items()) + "\n")


class Cluster:
    def __init__(self, llama_dir: Path, models_dir: Path):
        self.llama_dir, self.models_dir = llama_dir, models_dir
        self.procs: dict[str, subprocess.Popen] = {}

    def _spawn(self, name: str, args: list[str], env_extra: dict | None = None) -> None:
        log = open(WORK / f"{name}.log", "wb")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True  # own process group, so kill_tree reaches children
        env = {**os.environ, **(env_extra or {})}
        self.procs[name] = subprocess.Popen([sys.executable, "-m", "gpupool.cli", *args],
                                            stdout=log, stderr=subprocess.STDOUT, env=env,
                                            cwd=WORK, **kw)
        log.close()

    def start(self) -> None:
        # A leftover run (driver or cluster) on these ports would silently drive the new
        # cluster too; refuse instead of interleaving two runs.
        try:
            httpx.get(f"{COORD}/v1/models", timeout=2)
            raise RuntimeError(f"something already listens on {COORD}; stop the previous run")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass  # nothing listening (Windows may time out instead of refusing)
        if WORK.exists():
            shutil.rmtree(WORK)
        WORK.mkdir(parents=True)
        write_toml(WORK / "coordinator.toml", {
            "host": "127.0.0.1", "port": 8080, "db_path": str(WORK / "coord.db"),
            "cluster_token": "cluster-secret", "admin_key": "admin-secret",
            "api_keys": ["api-secret"], "models_dir": str(self.models_dir),
            "heartbeat_timeout_s": 6.0, "launch_timeout_s": 300.0,
        })
        self._spawn("coordinator", ["coordinator", "--config", str(WORK / "coordinator.toml")])
        for nid, n in NODES.items():
            d = WORK / nid
            d.mkdir()
            write_toml(WORK / f"agent-{nid}.toml", {
                "node_id": nid, "host": n["host"], "port": n["port"],
                "coordinator_url": COORD, "cluster_token": "cluster-secret",
                "llama_dir": str(self.llama_dir), "cache_dir": str(d / "cache"),
                "log_dir": str(d / "logs"), "budget_mb": n["budget_mb"],
                "include_cpu": n["include_cpu"], "heartbeat_s": 1.0,
            })
            self._spawn(f"agent-{nid}", ["agent", "--config", str(WORK / f"agent-{nid}.toml")],
                        None if nid == "a" else CPU_ONLY_ENV)
        wait(lambda: len([n for n in status()["nodes"] if n["alive"]]) == 3, 60, "3 nodes alive")

    def kill_tree(self, name: str) -> None:
        """Simulate a server dying: the agent and every engine it spawned."""
        p = self.procs[name]
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGKILL)
        p.wait()

    def stop(self) -> None:
        # Agents first, gracefully, so they stop their engines.
        for name, p in reversed(list(self.procs.items())):
            if p.poll() is not None:
                continue
            if os.name == "nt":
                p.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.kill_tree(name)


def status() -> dict:
    return httpx.get(f"{COORD}/admin/status", headers=ADMIN, timeout=10).json()


def wait(pred, timeout: float, what: str, every: float = 1.0):
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            v = pred()
            if v:
                return v
        except Exception as e:  # coordinator still starting
            last_err = e
        time.sleep(every)
    raise TimeoutError(f"timed out waiting for {what} (last error: {last_err})")


def admin(method: str, path: str, **kw) -> httpx.Response:
    r = httpx.request(method, COORD + path, headers=ADMIN, timeout=60, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {r.status_code}: {r.text}")
    return r


def replicas(model: str, states=("ready",)) -> list[dict]:
    return [r for r in status()["replicas"] if r["model"] == model and r["state"] in states]


def free_mb(node: str, device: str) -> int:
    for n in status()["nodes"]:
        if n["node_id"] == node:
            for d in n["devices"]:
                if d["device_id"] == device:
                    return d["free_mb"]
    raise KeyError((node, device))


def chat(model: str, prompt: str, max_tokens: int = 128, stream: bool = False,
         system: str = "You are a concise assistant.") -> dict:
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "stream": stream,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    if not stream:
        r = httpx.post(f"{COORD}/v1/chat/completions", json=body, headers=API, timeout=600)
        r.raise_for_status()
        j = r.json()
        return {"text": j["choices"][0]["message"]["content"], "timings": j.get("timings", {}),
                "wall_s": time.perf_counter() - t0}
    ttft, n_chunks, text = None, 0, []
    with httpx.stream("POST", f"{COORD}/v1/chat/completions", json=body, headers=API,
                      timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if "error" in j:
                raise RuntimeError(j["error"])
            delta = j["choices"][0]["delta"].get("content") or ""
            if delta:
                ttft = ttft if ttft is not None else time.perf_counter() - t0
                n_chunks += 1
                text.append(delta)
    total = time.perf_counter() - t0
    return {"text": "".join(text), "ttft_s": ttft, "tokens": n_chunks,
            "decode_tps": (n_chunks - 1) / (total - ttft) if ttft and n_chunks > 1 else None,
            "wall_s": total}


def deploy(name: str, file: str, replicas_n: int = 1, ctx: int = 4096) -> dict:
    admin("POST", "/admin/models", json={"name": name, "source": f"coordinator://{file}",
                                         "ctx_size": ctx, "replicas": replicas_n})
    plan = admin("POST", f"/admin/deploy/{name}", params={"dry_run": 1}).json()
    return plan


def undeploy(name: str) -> None:
    admin("DELETE", f"/admin/models/{name}")
    wait(lambda: not replicas(name, ("pending", "launching", "ready", "draining")), 120,
         f"{name} drained")


def show_plan(plan: dict) -> None:
    print(f"  plan: tier={plan['tier']} head={plan['head_node']} est_total={plan['est_total_mb']} MB")
    for a in plan["assignments"]:
        print(f"    {a['node_id']}/{a['device_id']:6} as {a['llama_device']:6} "
              f"layers={a['layers']:3} est={a['est_mb']} MB endpoint={a['rpc_endpoint']}")


def scenario_single_gpu(results: dict) -> None:
    print("\n== 1. 0.5B on one GPU")
    before = free_mb("a", "CUDA0")
    plan = deploy("qwen05", SMALL)
    show_plan(plan)
    t0 = time.monotonic()
    wait(lambda: replicas("qwen05"), 300, "qwen05 ready")
    load_s = time.monotonic() - t0
    time.sleep(3)  # next heartbeat reflects the loaded model
    used = before - free_mb("a", "CUDA0")
    est = plan["est_total_mb"]
    chat("qwen05", "Say hi.", 8)  # warm-up
    s = chat("qwen05", "Write a short paragraph about GPUs.", 128, stream=True)
    ns = chat("qwen05", "Write a short paragraph about GPUs.", 128)
    print(f"  ready in {load_s:.1f}s; VRAM used {used} MB vs estimate {est} MB "
          f"({(est - used) / used * 100:+.0f}%)")
    print(f"  via router: TTFT {s['ttft_s'] * 1000:.0f} ms, decode {s['decode_tps']:.1f} tok/s; "
          f"llama-server timings: prefill {ns['timings'].get('prompt_per_second', 0):.0f} t/s, "
          f"decode {ns['timings'].get('predicted_per_second', 0):.1f} t/s")
    results["single_gpu"] = {"plan_tier": plan["tier"], "load_s": load_s, "vram_used_mb": used,
                             "vram_est_mb": est, "ttft_ms": s["ttft_s"] * 1000,
                             "decode_tps_router": s["decode_tps"], "timings": ns["timings"]}
    undeploy("qwen05")


def scenario_split(results: dict) -> None:
    print("\n== 2. 3B split across nodes over RPC")
    plan = deploy("qwen3b", BIG)
    show_plan(plan)
    t0 = time.monotonic()
    try:
        wait(lambda: replicas("qwen3b"), 900, "qwen3b ready", every=2)
    except TimeoutError:
        print(json.dumps(replicas("qwen3b", ("failed", "launching")), indent=1)[:3000])
        raise
    load_s = time.monotonic() - t0
    ans = chat("qwen3b", "What is the capital of France? Answer in one word.", 16)
    s = chat("qwen3b", "Write a short paragraph about GPUs.", 96, stream=True)
    ns = chat("qwen3b", "Write a short paragraph about GPUs.", 96)
    ok = "paris" in ans["text"].lower()
    print(f"  ready in {load_s:.1f}s; answer={ans['text']!r} correct={ok}")
    print(f"  via router: TTFT {s['ttft_s'] * 1000:.0f} ms, decode {s['decode_tps']:.1f} tok/s; "
          f"llama-server timings: prefill {ns['timings'].get('prompt_per_second', 0):.0f} t/s, "
          f"decode {ns['timings'].get('predicted_per_second', 0):.1f} t/s")
    results["split"] = {"plan": plan, "load_s": load_s, "answer": ans["text"], "correct": ok,
                        "ttft_ms": s["ttft_s"] * 1000, "decode_tps_router": s["decode_tps"],
                        "timings": ns["timings"]}
    undeploy("qwen3b")


def scenario_failover(cluster: Cluster, results: dict) -> None:
    print("\n== 3. Failover: 2 replicas of 0.5B, node b dies")
    deploy("qwen05", SMALL, replicas_n=2)
    wait(lambda: len(replicas("qwen05")) == 2, 600, "2 replicas ready", every=2)
    before = {r["replica_id"]: r["placement"]["head_node"] for r in replicas("qwen05")}
    print(f"  replicas: {before}")
    victim = next((n for n in before.values() if n != "a"), None)
    if victim is None:
        raise RuntimeError(f"expected one replica off node a, got {before}")
    # Distinct system prompts spread requests across both replicas.
    ok = fail = 0
    errors: list[str] = []
    t_kill = time.monotonic()
    cluster.kill_tree(f"agent-{victim}")
    print(f"  killed node {victim} (agent + engines)")
    t_end = t_kill + 40
    i = 0
    while time.monotonic() < t_end:
        try:
            chat("qwen05", "Count to three.", 12, system=f"Assistant number {i}.")
            ok += 1
        except Exception as e:
            fail += 1
            errors.append(str(e)[:200])
        i += 1
    replaced = wait(lambda: [r for r in replicas("qwen05")
                             if r["replica_id"] not in before], 600, "replacement ready", every=2)
    t_recover = time.monotonic() - t_kill
    new_nodes = [r["placement"]["head_node"] for r in replicas("qwen05")]
    failed = [r for r in status()["replicas"] if r["state"] == "failed"]
    print(f"  requests during failover: ok={ok} failed={fail}")
    print(f"  replacement {replaced[0]['replica_id']} on {replaced[0]['placement']['head_node']} "
          f"ready {t_recover:.1f}s after the kill; replicas now on {new_nodes}")
    print(f"  failed replica reason: {failed[0]['error'] if failed else None}")
    results["failover"] = {"ok": ok, "failed": fail, "errors": errors[:5],
                           "recover_s": t_recover, "nodes_after": new_nodes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-dir", default=str(ROOT / ".cache/llama/b11342-cuda12.4"))
    ap.add_argument("--models-dir", default=str(ROOT / ".cache/models"))
    ap.add_argument("--only", default="1,2,3")
    args = ap.parse_args()
    only = set(args.only.split(","))
    cluster = Cluster(Path(args.llama_dir).resolve(), Path(args.models_dir).resolve())
    results: dict = {}
    try:
        cluster.start()
        print("cluster up:", [(n["node_id"], [(d["device_id"], d["usable_mb"]) for d in n["devices"]])
                              for n in status()["nodes"]])
        if "1" in only:
            scenario_single_gpu(results)
        if "2" in only:
            scenario_split(results)
        if "3" in only:
            scenario_failover(cluster, results)
    finally:
        cluster.stop()
        (WORK / "results.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"\nresults: {WORK / 'results.json'}; logs: {WORK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
