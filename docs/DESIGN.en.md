# gpupool — Design

> English version. Vietnamese version: [DESIGN.vi.md](DESIGN.vi.md). Keep both in sync.

gpupool pools scattered free VRAM across many servers and serves LLMs through one
OpenAI-compatible API. The engine is llama.cpp (`llama-server` + `ggml-rpc-server`); what we
write is the control plane: agent, scheduler, coordinator, router.

## 1. Decisions that changed from the original plan

| Item | Plan | Decision | Why |
| --- | --- | --- | --- |
| Control-plane language | Go or Rust | Python 3.12, installed with `uv` | `uv` is one user-space binary that fetches its own Python → no dependency on the server's Python or sudo. |
| RPC binary name | `rpc-server` | `ggml-rpc-server` (b11342) | Renamed upstream; the agent looks for both. |
| llama.cpp build | build from source | pin b11342; prebuilt or self-built, passed via `llama_dir` | The dev driver supports CUDA 13.3 → the `cuda-13.4` build does not run; use `cuda-12.4`. |
| Device order | implicit | always pass `--device` + `--tensor-split` in the same order | Verified: llama.cpp lists local devices first, then RPC. |
| Remote devices | 1 rpc-server per node | 1 rpc-server per device | Own port, deterministic `RPCi` name, can be stopped individually. |
| Model source | URL | URL, local path, or `coordinator://<file>` | Servers reach each other but may have no internet. |

## 2. Scope of the first version

In: agent (phase 1), scheduler (phase 2), router + load balancing + prefix-aware routing +
metrics (phase 3), the core of phase 4 (node loss → re-place, retry before the first token,
drain, API keys). Phase 0 (baseline numbers) runs on the dev machine during the test step.

Deferred: autoscaling, web dashboard, rate limiting, phase 5, vLLM backend.

## 3. Architecture

```
client ──HTTP (OpenAI API, API key)──► coordinator
                                       ├ router     (proxy, load balancing, prefix)
                                       ├ scheduler  (estimate + placement, pure functions)
                                       ├ reconciler (2 s loop: health, re-place, drain)
                                       └ store      (SQLite, WAL)
                                            ▲ heartbeat 2 s       │ start/stop engine
                                            │                     ▼
   server A: agent ─ llama-server (head) ──TCP RPC──► server B: agent ─ ggml-rpc-server (CUDA0)
                                         └─TCP RPC──► server C: agent ─ ggml-rpc-server (CUDA0)
```

- One package `gpupool`, one CLI: `gpupool agent | coordinator | register | plan | scale | undeploy | status`.
- Only the head node needs the GGUF file; rpc-servers receive weights over the network (`-c` enables the rpc-side cache).
- Agent ↔ coordinator authenticate with `Authorization: Bearer <cluster_token>`. RPC is
  unencrypted: rpc-servers bind only the node's internal IP.

## 4. Data contracts

Source of truth: `src/gpupool/common/models.py` (pydantic v2). Summary:

| Model | Direction | Key fields |
| --- | --- | --- |
| `Device` | agent → coordinator | `device_id` ("CUDA0", "CPU"), `kind`, `total_mb`, `free_mb`, `usable_mb` = max(0, min(free − margin, budget)) |
| `NodeReport` | heartbeat | `node_id`, `agent_url`, `host` (IP other nodes use for RPC), devices, engines, `llama_version`, models |
| `EngineSpec` | coordinator → agent | `engine_id`, `kind` rpc/server, `port`, `devices` (order = `--device`), `rpc_endpoints`, `tensor_split`, `ctx_size`, `parallel` |
| `EngineStatus` | agent → coordinator | `state` starting/running/exited/failed, `exit_code`, `log_tail` (≤ 50 lines) |
| `ModelSpec` | admin | `name`, `source`, `ctx_size`, `parallel`, `replicas` |
| `ModelMeta` | read from GGUF | `n_layers`, `n_head_kv`, `head_dim`, `layer_bytes[i]`, `output_bytes` |
| `Placement` | scheduler | `tier`, `head_node`, `head_port`, `assignments` (node, device, `llama_device`, `rpc_endpoint`, layers, est_mb), `tensor_split` |
| `ReplicaRecord` | store | placement + state `pending → launching → ready → draining → stopped`, or `failed` |

## 5. API

**Agent** (`:7070`, bearer token except `/health`):

| Method | Path | In → Out |
| --- | --- | --- |
| GET | `/health` | `{"ok": true}` |
| GET | `/report` | `NodeReport` |
| POST | `/engines` | `EngineSpec` → `EngineStatus` (409 already running, 422 port busy / model missing) |
| GET / DELETE | `/engines/{id}` | `EngineStatus` (DELETE: terminate, kill after 10 s) |
| POST | `/models/ensure` | `{"name","source"}` → `{"path","bytes"}` (downloads into `.part`, atomic rename) |

**Coordinator** (`:8080`):

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/internal/heartbeat` | cluster token |
| GET | `/files/{name}` | cluster token; a file in `models_dir`, path traversal rejected |
| POST | `/admin/models` | register a `ModelSpec` (admin key) |
| DELETE | `/admin/models/{name}` | drain every replica, then delete |
| POST | `/admin/models/{name}/scale?replicas=N` | change the desired replica count |
| POST | `/admin/deploy/{model}?dry_run=1` | returns a `Placement`, launches nothing |
| DELETE | `/admin/replicas/{id}` | drain one replica |
| GET | `/admin/status` | nodes, devices, replicas, outstanding |
| GET | `/v1/models` | OpenAI format |
| POST | `/v1/chat/completions`, `/v1/completions` | proxied, `stream: true` supported (API key) |
| GET | `/metrics` | Prometheus text |

## 6. Scheduler

**Memory estimate** (`scheduler/estimate.py`):

- `kv_bytes_per_layer = 2 × ctx_size × n_head_kv × head_dim × 2` (K and V, f16).
- A device holding layer range `L` needs: sum of `layer_bytes` over `L` + `|L| × kv` + overhead
  (300 MB for CUDA, 150 MB for CPU).
- `output_bytes` (output.weight, or token_embd when tied, + output_norm) counts on the **last
  device** in `--device` order. token_embd stays in host RAM and is not counted on any GPU.
- Metadata comes from our own GGUF parser that reads only the header (local file or HTTP
  stream), never the whole file.

**Placement rules** (`scheduler/placement.py`, `plan(...) -> Placement`, raises `NoFit`):

1. **GPUs first.** Run all three tiers below with CUDA devices only. CPU devices join only when
   the GPUs of the whole pool cannot hold the model; then a lone CPU device may not win the
   single_gpu tier (a GPU + CPU split is faster than CPU only).
2. **single_gpu**: best fit — the smallest device that still fits, keeping big devices for big models.
3. **single_node**: on one node, add devices by usable desc until it fits; pick the node needing the fewest devices.
4. **multi_node**: add nodes by total usable desc (fewest nodes), then drop devices that are not needed.
5. Split layers in proportion to capacity, then repair with exact per-layer bytes until every
   device has `est_mb ≤ usable_mb`. Head = the node holding the most layers. `tensor_split` = layer counts.
6. Device order: the head's CUDA devices (`CUDA0`…), then every other device over RPC
   (`RPC0`, `RPC1`…), including the head's own CPU.

No network measurements yet, so no latency-aware ordering (phase 5).

## 7. Router

- Candidates = `ready` replicas of the model whose head node is alive.
- **Prefix key** = sha256 of all messages but the last (canonical JSON, cut at 4 KB); with a
  single message, its first 512 characters.
- Rendezvous hash(prefix, replica) picks the preferred replica; if it is more than 2 requests
  busier than the least-loaded one, the least-loaded one is used.
- Sends `cache_prompt: true`; the head runs with `--cache-reuse 256 --metrics`.
- Retries on connection errors or 5xx **before the first byte**, at most twice. Never after bytes were sent.
- The outstanding counter is released exactly once, including when the client disconnects mid-stream.

## 8. Reconciler (2 s loop)

- Node without heartbeat for > 10 s → replicas using it become `failed`; surviving engines are stopped.
- An engine reported `exited/failed`, or missing from its node's report → replica `failed`.
- Fewer replicas than `replicas` → `plan()` and launch, at most one replica per model per tick;
  a failed launch backs off 5 s, 10 s, 20 s… up to 300 s.
- **VRAM reservation**: when planning, `usable_mb` is reduced by replicas in `launching`, and by
  replicas that just became `ready` until their node sends a report more than 5 s after the
  ready time (an older report may predate the model load → the same VRAM would be handed out twice).
- Launch: start rpc engines → wait running → `ensure` the model on the head → start the head →
  wait for `/health` 200 (`launch_timeout_s`) → `ready`. Any failure → stop every engine created so far.
- A device hosting an engine with `free_mb < 256` → launch a replacement first, drain the old replica once it is ready.
- Drain: wait for outstanding = 0 (at most 60 s) → stop engines → `stopped`.

## 9. Code layout

```
src/gpupool/
  common/      models.py auth.py config.py           shared contracts
  agent/       gpu.py procs.py models_cache.py app.py
  scheduler/   gguf_meta.py estimate.py placement.py
  coordinator/ store.py agent_client.py reconciler.py app.py
  router/      balancer.py proxy.py
  cli.py
tests/         test_agent_* test_scheduler_* test_coordinator_* test_router_*
```

Tests: `uv run pytest` (unit), `uv run pytest -m real` (needs llama.cpp binaries, a GGUF, a GPU).

## 10. Test plan

The dev machine has one GTX 1650 4 GB → emulate 3 servers with 3 agents on localhost:

- Agent A: real `CUDA0`, `budget_mb` capped to force a split.
- Agents B, C: a `CPU` device with a fixed `budget_mb`, running a real `ggml-rpc-server -d CPU` → real RPC over TCP.

| Test | How | Passes when |
| --- | --- | --- |
| Baseline (phase 0) | Qwen2.5-0.5B Q4_K_M on CUDA0 | prefill and decode tokens/s, TTFT recorded |
| Estimate vs reality | compare `est_mb` with measured VRAM | error ≤ 20% |
| Real split | Qwen2.5-3B Q4_K_M, A capped → split over B, C | correct answer via `/v1/chat/completions`, tokens/s recorded |
| Failover | kill the agent of one replica | later requests still succeed, the replica is re-placed |

## 11. Answered open questions (2026-10-02)

- Servers can reach each other; internet is not guaranteed → `coordinator://<file>` source;
  llama.cpp binaries are passed via `llama_dir`.
- No model size limit; the goal is to use the pool's total usable VRAM → multi-node split is the
  main path; margins are configurable per node (`margin_pct`, `margin_min_mb`, `budget_mb`).
