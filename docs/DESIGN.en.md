# gpupool — Design

> English version. Vietnamese version: [DESIGN.vi.md](DESIGN.vi.md). Keep both in sync.

gpupool pools scattered free VRAM across many servers and serves several LLMs through one
OpenAI-compatible API. The engine is llama.cpp (`llama-server` + `ggml-rpc-server`, build b11342);
what we write is the control plane: an **agent** on every GPU server, and a **coordinator** that
holds the scheduler, the reconciler, the autoscaler, the router and the web UI.

This document describes the system as it is in the code. Related documents:

- Full HTTP API (every endpoint, body and error): [API.en.md](API.en.md).
- Why the multi-model platform works the way it does (resource model, policy, algorithm): [PLATFORM_DESIGN.en.md](PLATFORM_DESIGN.en.md).
- Web UI design: [UI_DESIGN.en.md](UI_DESIGN.en.md). Installation: [QUICKSTART.en.md](QUICKSTART.en.md). Test results: [TEST_REPORT.en.md](TEST_REPORT.en.md).

## 1. Decisions that changed from the original plan

| Item | Plan | Decision | Why |
| --- | --- | --- | --- |
| Control-plane language | Go or Rust | Python 3.12, installed with `uv` | `uv` is one user-space binary that fetches its own Python, so there is no dependency on the server's Python or sudo. |
| RPC binary name | `rpc-server` | `ggml-rpc-server` (b11342) | Renamed upstream; the agent looks for both. |
| llama.cpp build | build from source | pin b11342; prebuilt or self-built, passed via `llama_dir`; Docker images carry it | The dev driver supports CUDA 13.3, so the `cuda-13.4` build does not run; use `cuda-12.4`. |
| Device order | implicit | always pass `--device` + `--tensor-split` in the same order, and `--rpc` before `--device` | llama.cpp lists local devices first, then RPC; it resolves device names while parsing arguments, so `RPC0` exists only after `--rpc` registered the servers. |
| Remote devices | 1 rpc-server per node | 1 rpc-server per device | Own port, deterministic `RPCi` name, can be stopped individually. |
| Model source | URL | URL, absolute local path, or `coordinator://<file>` | Servers reach each other but may have no internet. |
| Agent liveness | agent pushes a heartbeat every 2 s | the coordinator **pulls** `GET /report` from every registered agent (`poll_s` = 2 s) | With push, a server deleted in the UI would re-appear on its next beat. Push stays available (`push_heartbeat`) for old setups, but only registered servers are accepted. |
| Server registration | coordinator knows its agents from config | an agent self-registers (`POST /internal/join`, driven by `--join "<url>#<token>"`); the UI can also add by URL | One command per server. The coordinator probes `/report` first, so only a reachable, correctly-tokened agent registers; a server removed in the UI gets 403 so removal sticks. |
| Autoscaling, dashboard | deferred | done (sections 7 to 9, web UI) | |
| GPU identity | `device_id` ("CUDA0") | `uuid` / `pci_bus_id` as the stable key; `device_id` is only a position | `device_id` shifts when a GPU drops off the bus. Placements store `device_uuid`; failure detection and occupancy resolve by uuid. |
| Memory estimate | static formula | formula + per-model self-calibration (section 12) | The formula is exact on small models only; real runs correct it. |

## 2. Scope

In: agent, scheduler, router with load balancing and prefix-aware routing, metrics, node-loss
recovery, drain, API keys, **multi-model policy** (priority, spread, replica bounds, autoscaling,
scale-to-zero, preemption), **rebalancing**, KV-cache quantization and speculative decoding,
RPC firewall, VRAM self-calibration, persisted control state, web UI, a model library (Hugging
Face download, local paths), events and webhook, a simulated 3-server cluster for demos and CI.

Not done: rate limiting, latency-aware ordering from measured network round trips, a vLLM backend,
encryption of the RPC traffic (llama.cpp RPC is plain TCP; see section 11).

## 3. Architecture

```
client ──HTTP (OpenAI API, API key)──► coordinator
                                       ├ router      (proxy, load balancing, prefix, cold start)
                                       ├ scheduler   (estimate + placement + scoring, pure functions)
                                       ├ reconciler  (2 s loop: health, launch, drain, preempt, rebalance)
                                       ├ autoscaler  (2 s loop: scrape /metrics, desired replica count)
                                       ├ poller      (2 s: GET /report from every agent)
                                       ├ library     (GGUF files: HF download / local path)
                                       ├ events      (+ optional webhook)
                                       ├ web UI + /api
                                       └ store       (SQLite, WAL)
                                            ▲ pull /report 2 s    │ start/stop engine
                                            │                     ▼
   server A: agent ─ llama-server (head) ──TCP RPC──► server B: agent ─ ggml-rpc-server (CUDA0)
                                         └─TCP RPC──► server C: agent ─ ggml-rpc-server (CUDA0)
```

- One package `gpupool`, one CLI: `gpupool agent | coordinator | register | plan | scale | undeploy | status`.
- Agent and coordinator authenticate with `Authorization: Bearer <cluster_token>`; `/admin/*` and
  `/api/*` use the admin key; `/v1/*` uses the API keys (open when none are configured, and the
  coordinator prints a warning). Missing `admin_key` / `cluster_token` are generated once and kept
  in `secrets.json` next to the database (written atomically; an unreadable file is never overwritten).
- Traffic between coordinator, agents and llama.cpp never goes through an HTTP proxy
  (`common/net.py`); only traffic that leaves the cluster (Hugging Face, model URLs, webhook) may.

### 3.1 How RPC layer split works

llama.cpp splits a model **by layers** across devices (`--split-mode layer`). A remote GPU is just
another device: the head's `llama-server` is started with `--rpc host:port,...` and sees them as
`RPC0`, `RPC1`, and so on.

- **Only the head node needs the GGUF.** The coordinator never copies the model to RPC nodes; the
  head loads it and sends each remote device its tensors over TCP when the model loads.
- Each `ggml-rpc-server` is started with `-c`, which enables its **local tensor cache**: a later
  load of the same model transfers (almost) nothing.
- Activations cross the network once per token per hop, so a split over RPC trades speed for
  capacity. The scheduler models this as a fixed per-hop cost (section 6.3) and uses multi-node
  placements only when no single GPU or single node fits.
- One `ggml-rpc-server` per device, bound to the agent's `host` on a port from `port_range`
  (9000-9999) that the coordinator allocates, so ports never collide across replicas.
- The head's own CPU device, if exposed (`include_cpu`), is also reached through an rpc-server
  (`-d CPU`), so it is an `RPCi` device too.

### 3.2 What runs on the head

The head command line (`agent/procs.py`, `build_command`) always contains `-ngl 999`,
`--split-mode layer`, `--cache-reuse 256`, `--metrics`, `--fit off` (the scheduler already chose the
split; llama.cpp auto-fit must not change it) and `-lv 4` (verbosity 4 prints per-device model, KV
and compute buffer sizes, which calibration parses; 5 would add a dry-run pass, so it is exactly 4).
`--tensor-split` is the layer counts, passed when there is more than one device.

## 4. Data contracts

Source of truth: `src/gpupool/common/models.py` (pydantic v2). Changing a field is a protocol
change; agent and coordinator of different versions must keep talking, so every field added after
0.1 is optional.

| Model | Direction | Key fields |
| --- | --- | --- |
| `Device` | agent to coordinator | `device_id` ("CUDA0", "CPU"), `kind`, `total_mb`, `free_mb`, `usable_mb` = max(0, min(free - margin, budget)), `budget_mb` (configured cap; the coordinator also subtracts the estimates of its own live replicas from it, since free memory does not show their share), `uuid` / `pci_bus_id` (stable identity), `bandwidth_gbps` (NVML bus width x memory clock; ranks GPUs), telemetry (`util_pct`, `temp_c`, `power_w`, `processes`, `driver`, `cuda`) |
| `NodeReport` | agent to coordinator (`GET /report`) | `node_id`, `agent_url`, `host` (IP other nodes use for RPC), devices, engines, `llama_version`, `models` (GGUF files in the local cache), CPU/RAM telemetry |
| `EngineSpec` | coordinator to agent | `engine_id`, `kind` rpc/server, `port`, `devices` (order = `--device`), `rpc_endpoints`, `tensor_split`, `ctx_size`, `parallel`, `cache_type`, `spec_type`, `draft_model_path`, `draft_device`, `draft_n_max`, `allowed_peers` (rpc: hosts allowed to connect) |
| `EngineStatus` | agent to coordinator | `state` starting/running/exited/failed, `exit_code`, `log_tail` (at most 50 lines) |
| `ModelSpec` | admin | `name`, `source`, `ctx_size`, `parallel`, `replicas` (0 = stopped), `pin_devices`, `priority`, `spread`, `min_replicas`, `max_replicas`, `autoscale`, `idle_unload_s`, `preemptible`, `kv_cache_type`, `speculative`, `draft`, `draft_n_max` |
| `AutoscalePolicy` | inside `ModelSpec` | `target_busy` 0.7, `up_after_s` 30, `down_after_s` 300 |
| `ModelMeta` | read from the GGUF header | `n_layers`, `n_embd`, `n_head_kv`, `head_dim`, `layer_bytes[i]`, `output_bytes`, `vocab_size`, `tokenizer_model` |
| `Placement` | scheduler | `tier`, `head_node`, `head_port`, `assignments` (node, device, `device_uuid`, `llama_device`, `rpc_endpoint`, layers, `est_mb`), `tensor_split`, `est_total_mb`, `score`, `est_decode_tps`, `reasons`, `draft_est_mb`, `mem_factor` (calibration factor the estimates were multiplied by) |
| `Occupant` | reconciler to scheduler | an engine already on a GPU: node, device, model, `est_mb`, `busy` (0..1) |
| `ReplicaRecord` | store | placement + state `pending`, `launching`, `ready`, `draining`, `stopped`, or `failed` |
| `LibraryItem` | library | a GGUF the coordinator can serve as `coordinator://<name>` |

Replica state groups: *active* = pending, launching, ready (count toward the desired number);
*live* = active + draining (still hold ports and engines); *terminal* = stopped, failed.

## 5. API overview

Details, bodies and error codes are in [API.en.md](API.en.md). Groups:

| Group | Auth | Prefix and purpose |
| --- | --- | --- |
| Agent | cluster token (`/health` open) | `/health`, `/report`, `POST /engines`, `GET/DELETE /engines/{id}`, `GET /engines/{id}/memory` (buffer sizes parsed from the engine log), `POST /models/ensure` (download into `.part`, atomic rename; also fetches every part of a split GGUF, part 1 last) |
| Coordinator internal | cluster token | `/internal/join` (self-registration), `/internal/heartbeat` (push, registered servers only), `GET /files/{name}` (library file for heads; name resolved through the library, never joined to a path), `/healthz` open |
| Admin | admin key | `/admin/models`, `/admin/models/{name}/scale`, `/admin/deploy/{model}?dry_run=1`, `/admin/replicas/{id}`, `/admin/status` (the CLI uses these) |
| Web UI API | admin key | `/api/state`, `/api/servers`, `/api/servers/{id}/gpus/{dev}` (enable/disable a GPU in the pool), `/api/models/{name}` (PUT, start, stop, scaling, plan, delete), `/api/capacity`, `/api/simulate`, `/api/rebalance`, `/api/recommend`, `/api/events`, `/api/library`, `/api/hf/files` |
| OpenAI | API keys | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` (`stream: true` supported) |
| Metrics | none | `GET /metrics` (Prometheus text) |

## 6. Scheduler

`src/gpupool/scheduler/`: `gguf_meta.py`, `estimate.py`, `scoring.py`, `placement.py`. All pure
functions of (model metadata, spec, node reports, occupants); ports are allocated only for the
winning placement.

### 6.1 Memory estimate (`estimate.py`)

For a device holding layer range `L` (calibrated against llama.cpp b11342 verbose load logs):

```
need = ceil( sum(layer_bytes[i] for i in L) + |L| x kv_bytes_per_layer [+ output_bytes if last device] )
       + compute_buffer + runtime_context
kv_bytes_per_layer = 2 x ctx_size x n_head_kv x head_dim x bytes_per_element(kv_cache_type)
compute_buffer     = ceil(21 x 512 x n_embd x 4 bytes)      (default ubatch 512)
runtime_context    = 128 MB (CUDA) | 32 MB (CPU)
```

- Bytes per KV element: f16 2, q8_0 34/32, q4_0 18/32 (ggml block layouts).
- `output_bytes` (output.weight, or token_embd when tied, plus output_norm) counts on the **last
  device** in `--device` order. token_embd stays in host RAM and is not counted on any GPU.
- The result is multiplied by the model's calibrated memory factor (section 12), which is 1.0 until
  the model has been measured.
- A draft model (section 8.2) is estimated as a whole model on one CUDA device at the same ctx and
  cache type, with its own compute buffer and context.
- Metadata comes from our own GGUF header parser (local file or HTTP range stream), never the whole
  file; split GGUFs are summed over all parts (reading part 1 alone under-counts VRAM).

### 6.2 Candidate generation (`placement.py`)

`plan(...)` returns the best `Placement` or raises `NoFit`; `rank(...)` returns the top few for
display, simulation and rebalancing.

1. **GPUs first.** All tiers run on CUDA devices only. CPU devices join only when the GPUs of the
   whole pool yield no candidate; then single-device candidates are skipped (a GPU + CPU split is
   faster than CPU alone). Why: within a tier a CPU counts as capacity like any GPU, and mixing it
   in from the start would put layers in host RAM (10x or more slower) while VRAM is free elsewhere.
2. **single_gpu**: every device where the whole model fits is a candidate.
3. **single_node**: per node, take devices by usable VRAM descending, adding one at a time (from
   two) until a feasible split exists.
4. **multi_node**: only when tiers 2 and 3 produced nothing. Candidates are the union of
   - greedy: add nodes by total usable descending (fewest nodes), then drop devices that are not
     needed; when CUDA and CPU mix, a second variant starts from all CUDA devices and drops only
     CPU devices, so a small GPU is not discarded for a CPU-only placement unless that is faster;
   - exhaustive: with at most 8 nodes, every feasible node subset of the smallest feasible size and
     one larger, fastest estimate first; with more nodes, a bandwidth-ordered fill (untrimmed, trimmed
     by bandwidth, trimmed by size, plus one more node);
   - deduplicated and capped at `MAX_MULTI_NODE_CANDIDATES` = 12, which bounds scoring work.
5. **Layer split** per device set: proportional to capacity (usable minus overhead), at least one
   layer each, then a repair pass moves one layer at a time from the most overfull device to the one
   with most slack, checked with exact per-layer bytes, until every device has `est_mb <= usable_mb`.
   `tensor_split` = layer counts.
6. **Head** = the node holding the most layers (found as a fixed point: re-order, re-split, repeat).
   **Device order**: the head's CUDA devices, the head's CPU, then every other node by total usable
   descending (devices within a node by usable descending). Non-local devices are named `RPC0`, `RPC1`...

Pins (`pin_devices`), GPUs switched off in the pool, and VRAM reserved by launching replicas are
applied by the reconciler before planning by setting `usable_mb` to 0 or lowering it, so the
planner itself stays unaware of them.

### 6.3 Scoring (`scoring.py`, `placement.py`)

Estimated decode speed is memory-bandwidth bound: every token streams all weights once, layers on
different devices run one after another, so

```
time per token = sum over devices( bytes on device / (bandwidth x 0.5) ) + n_rpc_hops x 2 ms
est_decode_tps = 1 / time per token
```

(0.5 = fraction of peak bandwidth llama.cpp reaches; calibrated on a GTX 1650, 160 GB/s, where
Qwen2.5-0.5B q4_k_m measured 182 tok/s.) Unknown bandwidth: CPU 25 GB/s; an unknown CUDA GPU ranks as
the slowest known one (100 GB/s when none is known).

Score of a candidate (higher wins):

| Term | Weight | Meaning |
| --- | --- | --- |
| speed | +100 x tps / best tps among candidates | relative speed |
| sharing | -10 per engine already on a chosen GPU, -30 x its busy fraction | prefer idle GPUs |
| same model | -40 per replica of the same model on a chosen GPU (spread `gpu` or `node`) | spread replicas |
| same node | -20 per replica of the same model on a chosen node (spread `node` only) | spread across servers |
| waste | -15 x mean(usable / biggest usable) | best fit: keep big GPUs free for big models |
| devices | -5 per extra device | fewer devices |
| hops | -10 per network hop | fewer RPC links |

Ties break by smaller tier, then node and device name, so the result is deterministic. The top
three to four **reasons** (speed vs the fastest option, shared GPUs, same-model neighbours, number of
GPUs, network hops) are stored in `Placement.reasons` and shown in the UI. Spread is soft: a shared
GPU is still used when nothing else fits. Occupants come from every live replica, draining ones
included, because they still hold memory.

### 6.4 Draft model reservation

With `speculative = "draft"` the draft runs inside the head's `llama-server` on the head's first
local CUDA device. For each possible head device `D` the split is solved on a pool where only `D`
gives up the draft's memory, and `D` is pinned as the head's first device; charging the draft to
every CUDA device would reject pools that do fit it once. Above 16 CUDA devices only the roomiest
8 are tried as heads. The draft's MB is added to that assignment's `est_mb` and to `est_total_mb`
(`draft_est_mb`). If the draft cannot fit on a local CUDA device of the head, there is no placement.

## 7. Multi-model policy

Rationale and the allocation algorithm are in [PLATFORM_DESIGN.en.md](PLATFORM_DESIGN.en.md); this
is the behaviour implemented.

| Field | Default | Effect |
| --- | --- | --- |
| `priority` 0..100 | 50 | Higher is placed first each tick, so it gets scarce VRAM first; among equal priorities every model gets a first replica before any gets a second, then name order. |
| `spread` | `gpu` | `gpu`: avoid GPUs already holding a replica of this model; `node`: also avoid servers; `none`: no preference. Soft penalty only. |
| `replicas` | 1 | The on/off switch: 0 stops the model whatever the bounds say. |
| `min_replicas`, `max_replicas` | unset = `replicas` | The autoscaler's range. Unset means a fixed count. |
| `autoscale` | unset = defaults | `target_busy` 0.7, `up_after_s` 30, `down_after_s` 300. |
| `idle_unload_s` | unset | Only with `min_replicas = 0`: unload after this long without a request. |
| `preemptible` | true | False: a higher-priority model may never stop this model's replicas. |
| `pin_devices` | empty | `node/device` entries the replicas may use; every other GPU is treated as unusable for this model, including during preemption and rebalancing. |

### 7.1 Autoscaler (`coordinator/autoscaler.py`)

The reconciler asks `desired(spec)` instead of reading `replicas`. Every `poll_s` the autoscaler
scrapes llama-server `/metrics` of each ready head (`requests_processing`, `requests_deferred`) and
decides per model:

- busy = processing slots / `parallel`, averaged over ready replicas; a scrape older than 3 poll
  intervals is not trusted and the router's outstanding count is used instead.
- **Scale up** one replica when requests are queueing or busy > `target_busy`, held for
  `up_after_s`, no replica is still launching, and desired < max.
- **Scale down** one replica when nothing queues and busy < `target_busy` x 0.5 (hysteresis against
  flapping), held for `down_after_s`, and desired > the floor.
- The floor is `max(min_replicas, 1)`; a model with no saved state starts there.
- **Scale to zero**: with `min_replicas = 0` and `idle_unload_s` set, a model with no request for
  that long and nothing in flight is unloaded (desired 0). The next request calls `note_request`,
  which sets desired to 1 and wakes the reconciler (**cold start**). The router holds that request
  up to `cold_start_timeout_s` (120 s) polling for a ready replica, then answers 503 with
  `Retry-After: 10`.
- Desired count, last request time and last decision are written to `control_state` (key
  `autoscaler:<model>`) whenever the count changes, so a restart keeps an unloaded model unloaded and a
  scaled-up model scaled up. The up/down timers are deliberately not persisted (a restart only delays a
  step). `last_request` is refreshed in the store at most every 60 s, not per request.

### 7.2 Preemption (`preemption.py`, `reconciler.py`)

When a model has no placement (`NoFit`) and is **below its running minimum**
(`active < min(wanted, max(min_replicas, 1))`), it may stop other replicas. Autoscale extras never evict.

- **Victims** must be `preemptible` and have a **strictly lower** priority (equal priority never
  preempts). Candidates are ordered: replicas above their model's minimum first, then least busy, then
  newest. Greedy: add victims until the model places on the freed memory, then drop any victim the rest
  made redundant. The result is the smallest sufficient set.
- Victims are **drained** (not killed); the preemptor is not placed in the same tick, because draining
  replicas still hold memory; a later tick places it once they stop.
- **Claim**: while victims drain, a draining replica is no longer active, so its own model would see a
  deficit and relaunch into the memory being freed (seen on real hardware: the evicted model came
  straight back and the preemptor stayed stuck). Models of lower priority therefore do not launch until
  the preemptor has its replica, or the claim expires after `drain_timeout_s` + 120 s.
- **Cooldown**: a model that evicted others may not do it again for 600 s, and not while its earlier
  victims are still live, so two models with overlapping needs cannot keep stopping each other. Both
  the cooldown and the victim set are persisted.
- A preemption never places a model where a normal launch could not go: pins and disabled GPUs are
  re-applied after memory is freed.

### 7.3 Simulation and recommendation

`POST /api/simulate` runs the same ordering, placement and preemption rules on a copy of the cluster and
reports `start`, `stop`, `preempt` and `unplaced` (with the planner's own `NoFit` text). It ignores the
cooldown and treats memory freed by stops as available at once. `POST /api/recommend` ranks placements
for a library file at a given ctx and options, reports the largest context that fits a single GPU (binary
search over multiples of 256), an option that requires preemption, or a `not_possible` answer with the
numbers. Neither changes anything.

## 8. Performance options

Per model, set in `ModelSpec`, planned by the scheduler and passed to the head by the reconciler.

### 8.1 KV-cache quantization (`kv_cache_type`: f16 | q8_0 | q4_0)

Adds `-ctk T -ctv T` to the head (and `-ctkd/-ctvd` for the draft). The estimate uses the matching
bytes per element, so a quantized cache really lets a longer context fit. Measured on Qwen2.5-3B at
ctx 8192: q8_0 saves 132 MB, q4_0 204 MB versus f16 (theory 142 / 217 MB).

### 8.2 Speculative decoding (`speculative`: none | ngram | draft)

Fewer target passes mean fewer RPC round trips, which matters most for multi-node placements.

- `ngram`: guesses continuations from the text so far; no extra memory. Flag `--spec-type ngram-mod`.
- `draft`: a small model with the **same tokenizer** runs on the head's first local CUDA device. Flags:
  `--spec-type draft-simple -md <file> -devd <CUDA device> -ngld 999 --spec-draft-n-max N`.
  **In b11342, `-md` alone loads a draft model but never uses it**, so `--spec-type draft-simple` is set
  explicitly. The API refuses a draft whose tokenizer differs, or whose vocabulary differs by more than
  128 tokens (llama.cpp refuses otherwise). The reconciler also refuses to launch if the first
  assignment is not a local CUDA device of the head.
- `draft_n_max` (1..16, default 4): measured on a GTX 1650 with Qwen2.5-3B plus a 0.5B draft, 4 drafted
  tokens gave +5 %, 8 was slower than none.

## 9. Router (`router/`)

- Candidates = `ready` replicas of the model whose head node is alive. The router reads a snapshot of the
  store that is rebuilt only when the store's version moves; liveness is judged on every call, because a
  node that goes silent triggers no write.
- **Prefix key** = sha256 of all messages but the last (canonical JSON, cut at 4 KB); with a single
  message, its first 512 characters; for `/v1/completions`, the first 512 characters of the prompt.
- Rendezvous hash(prefix, replica) picks the preferred replica; if it has more than 2 more outstanding
  requests than the least-loaded one, the least-loaded one is used.
- Sends `cache_prompt: true`; the head runs with `--cache-reuse 256 --metrics`.
- Retries on connection errors or 5xx **before the first byte**, at most twice, never after bytes were
  sent. Every error feeds `note_error`, which makes the reconciler check that replica's `/health` next tick.
- A mid-stream failure ends the SSE stream with an error event; the outstanding counter is released
  exactly once, including when the client disconnects.
- Bodies over `max_request_mb` (32) get 413, also while reading a chunked upload.
- Unknown model 404; no ready replica 503; all attempts failed 502; a model loading from zero is held
  (section 7.1).
- `/metrics`: requests by model and code, retries, TTFT sum and count, outstanding per replica, free and
  usable MB per device, node liveness, replicas per model and state.

## 10. Reconciler (`coordinator/reconciler.py`)

One loop every `reconcile_s` (2 s), or sooner when `wake()` is called after a change in desired state.
API handlers call `wake()`, not `tick()`, because a tick waits on the tick lock and on HTTP calls to
agents. One tick, in this order:

1. **Track nodes**: emit `node_offline` / `node_online` once per transition.
2. **Fail orphaned launches** (section 13).
3. **Detect failures**.
4. **Check suspects**: replicas the router reported errors for; `GET /health` on the head, fail on non-200.
5. **Process drains**.
6. **Clear stable backoff**.
7. **Advance the move in flight** (section 10.3).
8. **Enforce counts**: priority order, preemption, launches, surplus drains, low-free replacements.
9. **Rebalance if due**.
10. **Prune** terminal replica rows (keep 10 per model).

### 10.1 Failure detection

- **Node dead** = its report is older than `heartbeat_timeout_s` (10 s) AND at least 2 consecutive polls
  really failed. A stale report alone may be our own event loop stalling, so it never kills a node. A
  loop-lag watchdog logs a warning when the coordinator's event loop is blocked for more than 1 s. A
  replica using a dead node becomes `failed`; surviving engines of that replica are stopped.
- **GPU gone**: a replica whose assigned GPU (matched by uuid) is no longer reported by its live server
  is failed with a `gpu_missing` event.
- **Engine crashed**: an engine reported `exited` / `failed`, or missing from a report newer than the
  replica's ready time, fails the replica (`engine_crashed` event with the last log lines).
- **Realloc**: a failed replica records a pending re-allocation; `realloc_started` and `realloc_done`
  events carry the time the model was unserved. A placement failure emits `realloc_failed`.
- **Backoff**: a failed launch backs off 5 s, 10 s, 20 s... up to 300 s per model. A replica that
  crashes within 300 s of becoming ready (`model_fault`) feeds the same backoff and, from the second
  time, emits `crash_loop`. A replica that stays ready for 300 s clears it. A dead node or missing GPU
  is not the model's fault and does not count.

### 10.2 Desired count, launch and drain

- Per model in priority order: fewer active replicas than wanted launches one (planned, then spawned as a
  task, at most one new replica per model per tick); more drains the newest.
- **VRAM reservation**: when planning, `usable_mb` is reduced by the estimates of replicas in `launching`,
  and of replicas that just became `ready` until their node sends a report more than 5 s after the ready
  time (an older report may predate the model load, and the same VRAM would be handed out twice). A
  device with a `budget_mb` is also capped by budget minus the estimates of all live replicas on it.
- A ready replica on a device with `free_mb < low_free_mb` (256) is replaced first: a new replica is
  launched, and the old one drained once the new one is ready.
- **Launch**: start the rpc engines (each with `allowed_peers` = the head's host) and wait until each is
  running; `ensure` the model (and the draft) on the head; start the head; wait for `/health` 200
  (`launch_timeout_s`, 600 s); mark `ready`; calibrate (section 12). Any failure stops every engine
  created so far, so a half-launched replica never pins VRAM on a shared GPU.
- A replica drained or failed while launching is rolled back quietly (it is not a launch failure).
- **Drain**: wait for outstanding = 0 (at most `drain_timeout_s`, 60 s), stop engines, mark `stopped`.
- **Server removal** (`DELETE /api/servers/{id}`): replicas touching the server are marked `stopped` (not
  `failed`) so the next tick simply re-places them; launching tasks are cancelled first.

### 10.3 Rebalancing

Replicas are placed one at a time, so the cluster can drift: a model placed when the pool was full may sit
across a network hop after a big GPU frees up. Rebalancing finds and fixes that.

- **Candidates**: each `ready` replica is scored in one pass with its alternatives (`rank(..., extra=[its
  placement])`), with its own memory removed from the occupants but still subtracted from the reports,
  because the new place must fit **while the old replica is still running**. A move needs a score gain of
  at least 25 points (a move reloads a whole model, so it must be clearly better, not just better).
  Moves stay within the model's pins. Best gain first.
- **Make-before-break**: the target is planned with the model pinned to the target devices, a new replica
  launches, and the desired count is temporarily one higher for that model. When the new replica is
  ready, the old one is drained (`rebalanced` event). The old replica keeps serving the whole time.
- **Abandoned** (`rebalance_failed` event, old replica untouched) when the new replica fails, the old
  one is no longer ready, or the new one is not ready within `launch_timeout_s` + 60 s.
- **One move at a time, cluster-wide**, and only when the cluster is quiet: no move in progress, no
  replica pending or launching, no preemption waiting for its memory.
- **When**: every `rebalance_s` (600 s; 0 disables the periodic run), or on demand with
  `POST /api/rebalance` (`dry_run` true lists the moves, false starts the best one). The timer starts
  at boot, not at 0 and is not persisted: right after a restart the reports are stale and a move would
  be a guess.

## 11. Security

- **API**: bearer tokens on three levels (cluster token, admin key, API keys). `extra_args` is never
  accepted by the agent (the cluster token must not become arbitrary llama-server flags), and
  `model_path` / `draft_model_path` must lie inside the agent's model cache or have been returned by
  `/models/ensure`. Library files are served by item name, never by joining user input to a path.
- **`ggml-rpc-server` has no authentication.** Anyone who can reach its port can allocate GPU memory,
  run graphs and read or write tensors; and the traffic is unencrypted. Two layers:
  1. Engines bind only the agent's `host` (the node's internal address), never a wildcard. The agent logs
     a warning if that address is public or a wildcard and the firewall is off.
  2. **RPC firewall** (`rpc_firewall`, `agent/firewall.py`, needs root or `NET_ADMIN`; Docker
     `--cap-add NET_ADMIN`): iptables (and ip6tables) rules for each RPC port, in one dedicated chain
     `GPUPOOL-RPC` jumped to from `INPUT` once. Per port, in order: allow loopback, allow each of
     `allowed_peers` (the replica's head), allow the **agent's own bind address**, then drop everything
     else. Rules are appended before the process starts, so the port is never reachable unprotected, and
     are deleted when the engine stops.
- **Why the agent's own address is allowed**: the readiness probe connects to the bind address, and a
  connection to one's own IP has that IP (not 127.0.0.1) as its source. Without the rule the engine
  never looks `running` (seen in a real Docker cluster).
- At start the agent flushes the whole chain: engines of a previous agent were reaped, so every rule is
  stale. If iptables is unavailable the agent logs an error and runs unprotected; a failed rule rolls
  back that port's rules and leaves it unrestricted with an error log. Protect ports 9000-9999 with your
  own firewall in that case.

## 12. VRAM self-calibration

The formula in 6.1 is exact on small models only; other models, drivers and GPUs differ. After a replica
becomes ready, the reconciler asks the head's agent for the real buffers (`GET /engines/{id}/memory`,
parsed from the `-lv 4` log: model, KV and compute buffer MiB per device, last occurrence wins; host and
mapped buffers are ignored) and compares:

```
sample = measured buffers (head log, summed over the placement's devices)
         / (sum of est_mb / placement.mem_factor - runtime_context per device
            [- one more context if a draft is on the head])
```

`est_mb` was already multiplied by the factor the placement was planned with (`Placement.mem_factor`),
so it is divided back out: sampling against scaled estimates measures `true ratio / factor`, and the
EMA would then settle on the square root of the true ratio (1.2 instead of 1.44), under-reserving VRAM.
The runtime context is subtracted because llama.cpp does not report it as a buffer. If any device of the
placement is missing from the data, the sample is skipped (partial data would bias the ratio low).

- **EMA**: `factor = 0.5 x sample + 0.5 x previous` (first sample: the sample itself), stored raw per
  model in `model_calibration` with a sample count.
- **Clamp**: planning uses the factor clamped to **0.9..2.0**. A lucky measurement must never shrink the
  safety margin below 0.9, and a wild one must not make a model unplaceable.
- The planner multiplies every need of that model (and its draft) by the factor; a `calibrated` event is
  emitted when the clamped factor moves by more than 5 %. The factor and sample count are shown in
  `/api/state`.
- Calibration is bookkeeping about a replica that already serves: it never fails a launch, and old
  agents, logs without `-lv 4` or missing devices simply leave the factor alone.

## 13. State persistence and crash recovery

SQLite (WAL, busy timeout 5 s). Tables: `nodes`, `models`, `replicas`, `servers`, `removed_servers`,
`gpu_flags`, `events` (last 1000), `control_state`, `model_calibration`, plus the library's own tables.

| Persisted | Where | Survives restart |
| --- | --- | --- |
| Models, replicas, servers, removed servers, GPU on/off flags | tables above | yes |
| Events | `events` | yes; webhook delivery is best effort |
| Preemption cooldowns and victim sets, crash-loop backoffs, the move in flight | `control_state` keys `preempted`, `backoff`, `move` | yes |
| Autoscaler desired count, last request, last decision | `control_state` key `autoscaler:<model>` | yes (timers are not) |
| Calibration factors | `model_calibration` | yes |
| `_last_rebalance` timer | memory | no, on purpose (stale reports after boot) |

Control state is written through on every change (these are rare events) and times are wall-clock, so
they mean the same after a restart. A store problem while saving is logged and never breaks
reconciliation; an unreadable row is treated as absent. A saved move whose replicas no longer exist is
cleared quietly.

**Crash recovery**
- *Coordinator killed mid-launch* (kill -9, OOM, power loss; a clean shutdown cancels the launch tasks and
  marks them failed itself): a replica record and its launch task are created together, so a `launching`
  record with no task in this process means the coordinator died. `_fail_orphaned_launches` fails it at
  the first tick and stops its possibly half-started engines, and the next tick re-plans. Seen on a real
  cluster: after `docker kill` the replica stayed `launching` forever and, being counted as active, kept
  its model from ever being launched again.
- *Agent killed*: each engine has a pid file (with the process create time, which guards against PID
  reuse); a new agent on the same `log_dir` stops leftover llama.cpp children (they would keep holding VRAM
  that no coordinator knows about) and flushes the firewall chain.
- *Agent unreachable*: the node is judged dead only on evidence (section 10.1); the poller counts polls that
  ran to completion with an error, and an agent reporting a different `node_id` than registered is ignored.
- *Node lost*: replicas fail, engines on surviving nodes are stopped, the model is re-placed by the next
  tick (a `realloc_*` event sequence).

## 14. Code layout

```
src/gpupool/
  cli.py                      gpupool agent | coordinator | register | plan | scale | undeploy | status
  common/
    models.py                 wire contracts (pydantic)
    config.py                 AgentConfig, CoordinatorConfig, TOML + env (GPUPOOL_*), join string, secrets
    auth.py                   bearer-token dependency
    net.py                    internal vs external HTTP clients, proxy handling
  agent/
    app.py                    FastAPI: /report /engines /models/ensure; self-join; optional push heartbeat
    procs.py                  engine command lines, process supervision, pid files, orphan reaping
    firewall.py               iptables chain GPUPOOL-RPC for RPC ports
    memlog.py                 per-device buffer sizes from a llama-server log
    gpu.py                    NVML / psutil device probing, budgets, margins
    models_cache.py           download to .part, atomic rename, split GGUFs
  scheduler/
    gguf_meta.py              GGUF header parser (file or URL)
    estimate.py               memory estimate
    scoring.py                bandwidth-based decode-speed estimate
    placement.py              candidates, split, scoring, draft reservation, plan / rank
  coordinator/
    app.py                    FastAPI wiring, /internal/*, /admin/*, /metrics, UI mount, watchdog
    api.py                    /api/* for the web UI
    library.py, library_api.py  GGUF library (HF download, paths), /files/{name}
    store.py                  SQLite schema and accessors
    poller.py                 pulls /report from every registered agent
    reconciler.py             tick loop, launch, drain, failure, preemption, rebalance, calibration
    preemption.py             victim selection
    autoscaler.py             busy metrics, desired count, cold start, idle unload
    events.py                 events + webhook notifier
    agent_client.py           HTTP client for the agent API
  router/
    balancer.py               prefix key, rendezvous hash, outstanding counters
    proxy.py                  /v1/*, retries, streaming, cold-start wait, Prometheus metrics
  ui/                         index.html, app.js, styles.css, vendor/alpine.min.js, favicon.svg
tests/                        test_agent_*  test_scheduler_*  test_coordinator_*  test_router_*
                              test_library_unit  test_net  test_config_env  test_onecmd
                              test_ui_static  test_ci_e2e_static
scripts/                      e2e_local.py  ci_e2e.py  ui_mock_server.py
docker/                       agent.Dockerfile  coordinator.Dockerfile
docker-compose.*.yml          coordinator, agent, sim (simulated 3-server cluster), ci
docs/                         DESIGN, PLATFORM_DESIGN, API, QUICKSTART, TEST_REPORT, UI_DESIGN (en + vi)
```

## 15. Test plan

Unit tests: `uv run pytest` (default excludes the `real` marker); `uv run pytest -m real` needs
llama.cpp binaries, a GGUF and a GPU. CI runs the unit suite, then builds both Docker images.

Rule: tests written against mocks prove the mocks work. Every serious defect found so far survived a green
unit suite and appeared in the first real run, so each feature is also run against reality once.

Real-engine runs (results in [TEST_REPORT.en.md](TEST_REPORT.en.md)):

- `scripts/e2e_local.py`: the dev machine has one GTX 1650 4 GB, so 3 servers are emulated by 3 agents on
  127.0.0.1/.2/.3: agent A has the real `CUDA0` with `budget_mb` capped to force a split; agents B and C expose
  a `CPU` device running a real `ggml-rpc-server -d CPU`, so RPC is real over TCP.
- `scripts/ci_e2e.py` with `docker-compose.ci.yml`: a real coordinator and 3 real agents (Docker images, CPU
  only) serve a tiny model split over several servers; catches Dockerfile errors and wrong Python in images.
- `docker-compose.sim.yml`: simulated 3-server cluster for demos; `GPUPOOL_FAKE_DEVICES` makes agents
  report different GPUs while sharing the one real card.
- `scripts/ui_mock_server.py`: in-memory coordinator API for working on the UI without a cluster.

| Test | How | Passes when |
| --- | --- | --- |
| Baseline | Qwen2.5-0.5B Q4_K_M on CUDA0 | prefill and decode tokens/s, TTFT recorded |
| Estimate vs reality | compare `est_mb` with the measured buffers | error within 20 % |
| Real split | Qwen2.5-3B Q4_K_M, A capped, split over B and C | correct answer via `/v1/chat/completions`, tokens/s recorded |
| Failover | kill the agent of one replica | later requests still succeed, the replica is re-placed |
| Preemption, rebalance, autoscaling, cold start | unit tests with fake agents (`test_coordinator_preemption`, `_reconciler`, `_autoscaler`) | victims chosen and drained, move is make-before-break, scaling follows the thresholds |
| RPC firewall | unit tests with a fake iptables runner (`test_agent_firewall`), plus a real Docker cluster | only the head (and the agent itself) reaches an RPC port; the engine reaches `running` |

## 16. Answered open questions

- Servers can reach each other; internet is not guaranteed, so the `coordinator://<file>` source exists
  and llama.cpp binaries are passed via `llama_dir` or baked into the image.
- No model size limit; the goal is to use the pool's total usable VRAM, so multi-node split is the main
  path for big models. Margins are configurable per node (`margin_pct`, `margin_min_mb`, `budget_mb`).
