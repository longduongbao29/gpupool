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
| Web UI API | admin key | `/api/state`, `/api/servers`, `/api/servers/{id}/gpus/{dev}` (enable/disable a GPU in the pool), `/api/models/{name}` (PUT, start, stop, scaling, plan, delete), `/api/capacity`, `/api/simulate`, `/api/rebalance`, `/api/recommend`, `/api/events`, `/api/library`, `/api/hf/files`, `/api/convert*` (Hugging Face to GGUF conversion, section 17) |
| OpenAI | API keys | `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions` (`stream: true` supported) |
| Metrics | none | `GET /metrics` (Prometheus text) |

## 6. Scheduler

`src/gpupool/scheduler/`: `gguf_meta.py`, `estimate.py`, `scoring.py`, `placement.py`. All pure
functions of (model metadata, spec, node reports, occupants); ports are allocated only for the
winning placement.

### 6.1 Memory estimate (`estimate.py`)

For a device holding layer range `L` (calibrated against llama.cpp b11342 verbose load logs):

```
need = ceil( sum(layer_bytes[i] + cache_bytes[i] for i in L) [+ output_bytes if last device] )
       + compute_buffer + runtime_context
cache_bytes[i]     = cells[i] x (k_row[i] + v_row[i]) x bytes_per_element(kv_cache_type)
                     + state_bytes[i] x parallel
cells[i]           = ctx_size, or for a sliding-window layer
                     parallel x pad256(min(pad256(ctx_size / parallel), n_swa + ubatch))
compute_buffer     = ceil(21 x 512 x n_embd x 4 bytes)      (default ubatch 512)
runtime_context    = 128 MB (CUDA) | 32 MB (CPU)
```

- Bytes per KV element: f16 2, q8_0 34/32, q4_0 18/32 (ggml block layouts).
- The per-layer layout follows llama.cpp b11342's loaders: `k_row = n_head_kv[i] x key_length`
  (`key_length_swa` on SWA layers), `v_row` likewise, 0 with MLA (`key_length_mla`: only the latent K
  is cached). Recurrent layers of hybrid models (`full_attention_interval` for Qwen3-Next / Qwen3.5,
  `recurrent_layers`, or 0 KV heads) have no KV but a per-sequence f32 state from the `ssm.*` keys.
  SWA layers come from a `sliding_window_pattern` array, or for gemma2/3/3n, gpt-oss, cohere2 and
  olmo2 from the loader's own period when `sliding_window` is set; any other architecture counts
  full layers (over-estimate, never an OOM). Without layout data (old metadata) every layer uses
  `2 x ctx_size x n_head_kv x head_dim`.
- MTP (`nextn`) blocks that llama.cpp loads only for `--spec-type draft-mtp` are kept apart
  (`nextn_bytes`) and count, with their cache, only when MTP is on.
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
planner itself stays unaware of them. A pin list is an **allowed set**, not a placement: every device
outside it gets `usable_mb = 0` (`Reconciler._apply_pins`), and the planner then picks the best
placement among what remains. An entry is `"<node>/<device>"` or `"<node>/*"`; the wildcard allows every
device of that server, including GPUs that register later, so a "this whole server" choice does not go
stale when the server grows. Pins and the pool's enable/disable switch are independent: a GPU left out of a
model's set stays enabled for other models.

### 6.3 Scoring (`scoring.py`, `placement.py`)

Estimated decode speed is memory-bandwidth bound: every token streams all weights once, layers on
different devices run one after another, so

```
time per token = sum over devices( bytes on device / (bandwidth x 0.5) ) + n_rpc_hops x 2 ms
est_decode_tps = 1 / time per token
```

(0.5 = fraction of peak bandwidth llama.cpp reaches; calibrated on a GTX 1650, 160 GB/s, where
Qwen2.5-0.5B q4_k_m measured 182 tok/s.) For a MoE layer the bytes per token are its shared weights plus
`expert_used_count / expert_count` of its routed experts (`ffn_*_exps`). Unknown bandwidth: CPU 25 GB/s; an unknown CUDA GPU ranks as
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
| `pin_devices` | empty | The allowed set: `node/device` or `node/*` (every device of that server, GPUs added later included) entries the replicas may use; every other GPU is treated as unusable for this model, including during preemption and rebalancing. Empty = all. It limits, the scheduler still chooses. |

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

### 8.2 Speculative decoding (`speculative`: none | ngram | draft | mtp)

Fewer target passes mean fewer RPC round trips, which matters most for multi-node placements.

- `ngram`: guesses continuations from the text so far; no extra memory. Flag `--spec-type ngram-mod`.
- `draft`: a small model with the **same tokenizer** runs on the head's first local CUDA device. Flags:
  `--spec-type draft-simple -md <file> -devd <CUDA device> -ngld 999 --spec-draft-n-max N`.
  **In b11342, `-md` alone loads a draft model but never uses it**, so `--spec-type draft-simple` is set
  explicitly. The API refuses a draft whose tokenizer differs, or whose vocabulary differs by more than
  128 tokens (llama.cpp refuses otherwise). The reconciler also refuses to launch if the first
  assignment is not a local CUDA device of the head.
- `mtp`: the model's own multi-token-prediction (`nextn`) blocks draft the tokens, in a second llama.cpp
  context on the same devices. Flags: `--spec-type draft-mtp --spec-draft-n-max N`. llama.cpp loads those
  blocks only in this mode, so the estimate adds them (at their layer position in the split), their KV and
  a second compute buffer on the last device only with `mtp`. The API refuses `mtp` for a GGUF without
  `nextn_predict_layers`.
- `draft_n_max` (1..16, default 4): measured on a GTX 1650 with Qwen2.5-3B plus a 0.5B draft, 4 drafted
  tokens gave +5 %, 8 was slower than none.

## 9. Router (`router/`)

- Candidates = `ready` replicas of the model whose head node is alive. The router reads a snapshot of the
  store that is rebuilt only when the store's version moves; liveness is judged on every call, because a
  node that goes silent triggers no write.
- **Prefix key** = sha256 of canonical JSON cut at 4 KB: for a single turn (system messages + one user
  message), all messages but the last, so a shared system prompt meets on one replica; for a multi-turn
  chat, the system messages and the first user message, which every later turn repeats, so the conversation
  stays with its KV cache. With a single message, its first 512 characters; for `/v1/completions`, the
  first 512 characters of the prompt.
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
- **Launch**: `ensure` the model (and the draft) on the head while the rpc engines start (each with
  `allowed_peers` = the head's host) and become running; once both are done, start the head; wait for `/health` 200
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
`gpu_flags`, `events` (last 1000), `control_state`, `model_calibration`, `convert_jobs` (section 17.6), plus the library's own tables.

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
    library.py, library_api.py  GGUF library (HF download, paths, converted files), /files/{name}
    convert_api.py            /api/convert/* (9 routes of the conversion feature)
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
  converter/                  Hugging Face -> GGUF conversion (section 17)
    models.py                 API contracts: SourceSpec, InspectResult, ConvertRequest, ConvertJob, Validation
    quant.py                  quantization table, size/VRAM estimates, recommendation, output names
    source.py                 HfClient, file selection, inspect
    jobs.py                   ConvertManager: SQLite jobs, single worker, pipeline, cleanup
    toolchain.py              tool discovery, command builders, run_tool, supported architectures
    validate.py               header, tokenizer and generation checks
    hf_tokenize.py            standalone script run by the converter's Python (HF token ids)
  ui/                         index.html, app.js, styles.css, vendor/alpine.min.js, favicon.svg
tests/                        test_agent_*  test_scheduler_*  test_coordinator_*  test_router_*
                              test_library_unit  test_net  test_config_env  test_onecmd
                              test_ui_static  test_ci_e2e_static  test_converter_*  test_coordinator_convert_api
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
  Its last stages run three conversions inside the coordinator (33 checks in all): a Hugging Face model
  (SmolLM2-135M-Instruct) to `Q4_K_M`, served split over RPC with a chat; a folder source to `Q8_0` (the source
  folder must stay untouched); and `IQ2_XS`, which needs an importance matrix, checking the stages and the
  matrix flag (`--skip-convert` leaves them out).
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

## 17. Hugging Face to GGUF conversion (`converter/`)

Only GGUF files can be served, but many models are published only as safetensors / PyTorch weights. The
coordinator therefore runs **conversion jobs**: source files (a Hugging Face repo or a folder) →
llama.cpp's `convert_hf_to_gguf.py` → `llama-quantize` → validation → the model library. Usage:
[QUICKSTART.en.md](QUICKSTART.en.md#serving-a-model-that-has-no-gguf-convert); routes:
[API.en.md](API.en.md#12-conversion-apiconvert).

The toolchain is optional. `Toolchain.problem()` returns what is missing (`GPUPOOL_CONVERT_DIR`,
`GPUPOOL_CONVERT_PYTHON`, `GPUPOOL_LLAMA_TOOLS_DIR`); without it everything else works and `POST /api/convert`
answers 503 with that explanation. `convert_api.py` imports the converter lazily so the router loads either way.
`llama-imatrix` is the one optional part inside the toolchain (`Toolchain.has_imatrix()`, reported as
`imatrix_available` by `GET /api/convert/options`), see 17.8.

| Module | Job |
| --- | --- |
| `converter/models.py` | pydantic contracts: `SourceSpec`, `InspectResult`, `QuantOption`, `ConvertRequest`, `ConvertJob`, `Validation`, the state sets. Every field is part of the HTTP API and the UI |
| `converter/quant.py` | the quantization table (bpw, tier, quality note), size and VRAM estimates, `recommend`, `plan_steps`, output naming |
| `converter/source.py` | `HfClient` (listing, model info, `config.json`, download), `select_files`, `local_files`, `inspect_source` |
| `converter/jobs.py` | `ConvertManager`: job table in SQLite, the single worker, the pipeline, cleanup |
| `converter/toolchain.py` | where the tools are, command builders, `run_tool` (low priority, process-tree kill), supported architectures |
| `converter/data/calibration.txt` | the built-in calibration text for importance matrices (package data, with a README explaining its origin) |
| `converter/validate.py` | header, tokenizer and generation checks |
| `converter/hf_tokenize.py` | runs *under the converter's Python*: token ids of the probe texts from Hugging Face (gpupool is not installed there) |
| `coordinator/convert_api.py` | the 9 routes, mapping `ConvertError.status` to HTTP |

### 17.1 Pipeline

```
queued → downloading → converting → [calibrating →] quantizing → validating → done
                                                                  └→ needs_review → (accept) → done
(any active state) → failed | cancelled
```

`calibrating` only exists for jobs that compute an importance matrix (17.8). When a job fails or is cancelled, the
stage it was in is stored as `failed_stage`, so the UI can mark the step and a reader of the API can tell a download
failure from a disk or calibration failure without parsing the error text.
1. **Source files.** For Hugging Face the tree listing gives names and sizes; `select_files` keeps `config.json`,
   tokenizer files and the weights, and drops everything else. Safetensors win over `pytorch_model*.bin` when both
   exist; `consolidated.*` (the Mistral-native duplicate of the weights, which doubled the download) is skipped;
   `*.py` is kept only with `allow_remote_code`; names that could escape the download folder (absolute, `..`,
   backslash, drive letter, hidden or `onnx`/`openvino`/... directories) are dropped, and `_check_rel` re-checks the
   selection before use.
2. **Download** (Hugging Face only). One file at a time into `.part`, size checked against the listing and the
   `Content-Length`, then an atomic rename; the writes run in a thread so the event loop that also serves inference
   is never blocked. A file already in the cache with the expected size is not fetched again.
3. **Stage.** The selected files are linked (symlink, else hardlink, else copy) into `.convert/<job>/src`.
4. **Convert.** `convert_hf_to_gguf.py src --outfile <outtype>.gguf --outtype <t>` with `plan_steps`: `F16`, `BF16`
   and `Q8_0` are written by the converter directly; every other type first writes a 16-bit intermediate (`auto` =
   bf16 for bf16 weights, otherwise f16).
5. **Calibrate** (only with an importance matrix, 17.8). `llama-imatrix` over the 16-bit intermediate writes
   `imatrix.gguf` into the job's scratch folder.
6. **Quantize.** `llama-quantize [--imatrix imatrix.gguf] [flags] intermediate out.gguf TYPE [threads]`; the intermediate is deleted at once.
   The advanced flags are checked against a whitelist of ggml type names so a user value can never become an extra
   command-line argument.
7. **Validate** (17.4). **Publish**: move the file to `models_dir/<name>` and register it (`LibraryItem.source`
   `"convert"`), then clean up, and only then report `done`.

Progress comes from the tools' own output: tqdm percentages for download and conversion, `[ i/ n]` lines of
llama-quantize; a progress bar that redraws one line is kept as its latest state, and the row is written to SQLite
at most once a second. The last 200 lines are kept in memory and 50 in the database.

### 17.2 One worker, FIFO, low priority

One asyncio worker takes one job at a time, in submission order. Reasons: conversion and quantization use all the
CPUs and a lot of RAM and disk, two at once would only slow each other down and could exhaust the disk; and the
coordinator also serves the router. Every tool is started in its own process group at low priority
(`BELOW_NORMAL_PRIORITY_CLASS` on Windows, `nice 10` elsewhere). Cancelling kills the whole process tree (the
converter forks helpers), waits for any thread work still running, and only then deletes the scratch folder, so a
cancelled job leaves no process and no open file. The converter runs with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

Names are checked at submit time (library, a file in `models_dir`, an unfinished job) with no `await` between the
check and the insert, so two submits cannot both pass; `retry` and `accept` check again.

### 17.3 Caching and disk

Downloads go to `models_dir/.hf/<owner>__<name>@<revision>/`, shared by retries and by other jobs for the same model
and revision (several quantization types queued together, or later ones when `keep_source` kept the files): a file
already there is not downloaded again. The cache is dropped when the job ends unless `keep_source` is set or another active job needs it; a failed or cancelled job keeps it for the retry, and
deleting that job releases it. The scratch folder of a job is `models_dir/.convert/<job>/`.

Free disk space is checked twice. First **at submit** (`_early_disk_check`): from the uncached part of the download
(files already in the cache do not count), the intermediate and the estimated output plus the margin, and when the
disk obviously cannot hold the job `POST /api/convert` answers 507 and queues nothing. Without it the same job would
be accepted, wait in the queue, download for minutes and only then fail in the worker. Second, **before each heavy
stage** in the worker, with a 512 MB margin, because free space can shrink while jobs wait: before the download
(`remaining download + intermediate + output`), before converting (`intermediate + output`) and before quantizing
(`output`). The intermediate is estimated as `params × 2` bytes (×4 for f32), or the source size when the count is
unknown. Failing early with the numbers beats a half-written multi-GB file; the check raises `ConvertError` with
status 507, which the worker turns into a *failed* job whose `error` explains it.

### 17.4 Validation and why the policy is what it is

`validate.py` never raises for a bad model; problems land in the `Validation` result and `jobs.py` applies the policy:

| Result | Outcome | Why |
| --- | --- | --- |
| `header_ok` false (unreadable GGUF, no architecture, no tokenizer) | job **failed** | such a file cannot be served at all; nothing is published |
| `tokenizer_ok` false or `generation_ok` false | **needs_review**, file kept, not in the library | it may be a converter bug or an acceptable quirk: only a person can tell, so the decision (accept or delete) is theirs. A wrong tokenizer silently degrades answers, which is why it is checked at all |
| a check that could not run (`null`) | warning only | not being able to check (no RAM for the CPU run, Hugging Face tokenizer failed to load, timeout) says nothing against the model, and blocking on it would make small coordinators unusable |

The tokenizer check tokenizes 8 fixed probe texts (English, Vietnamese, code, numbers, emoji with a ZWJ sequence,
odd spacing, blank lines, CJK) with Hugging Face (`hf_tokenize.py`, `add_special_tokens=False`) and with
`llama-tokenize --no-bos --no-parse-special --no-escape -f <file>` on the GGUF, and compares the ids. These are the
places converted tokenizers usually go wrong. The generation check runs `llama-simple -ngl 0` (always on the CPU, to
leave the GPUs to inference) for 16 tokens of "The capital of France is"; it is skipped when free RAM is below
1.2 × the file size, and its timeout is `120 s + 60 s per GB`. The GGUF is read with `gguf` and every derived object
is dropped before returning, because on Windows a lingering memory map blocks moving the file afterwards.

### 17.5 Size and VRAM estimates

`inspect` lists every type with an estimated file size, VRAM (file + f16 KV cache at context 4096 + 300 MB) and
whether it fits one GPU or the pool. The first version used one bits-per-weight average per type, taken from
llama-quantize's Llama-3-8B sizes; on Qwen2.5-0.5B it came out **24 % low**. Two effects were missing:

- **Embeddings.** llama-quantize keeps the token embedding and output matrices near 8 bits in the low-bit types.
  They are about 13 % of Llama-3-8B but 28 % of Qwen2.5-0.5B. `estimate_bytes` costs them separately
  (`vocab × hidden`, twice when input and output are not tied, at 8.5 bpw) and derives the bpw of the remaining
  weights from the whole-model reference.
- **K-quant fallback.** K-quants and `IQ4_XS` need rows that are a multiple of 256 values. When the hidden size is
  not (Qwen2.5-0.5B: 896, SmolLM2-135M: 576), llama-quantize falls back per tensor to a legacy type (`Q4_K` →
  `Q5_0`, `Q5_K` → `Q5_1`, `Q6_K` → `Q8_0`, the others → `IQ4_NL`), so the table lists the bpw of those fallbacks
  (4.5 to 8.5).

With both, the estimate is within about 2 % of the real files (Qwen2.5-0.5B `Q4_K_M`: 390.7 MB estimated, 397.8 MB
real; SmolLM2-135M: 103.1 against 105.5 MB). The same estimate is stored on the job (`est_output_bytes`) and drives
the disk checks.

The recommendation (`quant.recommend`) walks the ladder `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M` and takes the first that
fits one GPU, then the first that fits the pool (split over RPC is slower, and the reason says so). Models under
3 B parameters use only `Q8_0`, `Q6_K`, `Q5_K_M`, because small models lose quality fastest. Without GPUs there is
no fit information and a size-based default applies (`Q8_0` below 3 B, `Q5_K_M` below 15 B, else `Q4_K_M`).
The `IQ` types (`IQ1_*`, `IQ2_*`, `IQ3_*`) are listed as options but never the default; the
recommendation only mentions them when even `Q4_K_M` does not fit, with the note that they need an importance matrix.

The IQ rows were checked against real files, because their bpw is not the format's nominal figure (llama-quantize
keeps the output matrix and the most sensitive tensors at higher types, so whole files are bigger). The table uses
whole-file averages and the dialog's estimate came out within +6 % (`IQ2_XS`), -4 % (`IQ3_M`) and -1 % (`Q8_0`) of
the real files. The 256-block IQ types also fall back to `IQ4_NL` (4.5 bpw) for rows that are not a multiple of 256,
like the K-quants.

### 17.6 Persistence, restart and cleanup boundaries

Job rows live in the coordinator's SQLite database (table `convert_jobs`, its own connection like the library,
WAL), written on every state change and at most once a second for progress. On startup, jobs a previous process
left in an active state other than `queued` go back to `queued` with progress reset (their partial outputs are
discarded, finished downloads stay in the cache) and the worker starts; a sweep removes scratch folders no job
owns. A clean shutdown cancels the worker and leaves the row active, so the next start requeues it.

What gets deleted is deliberately narrow:

- Only paths **inside** `models_dir/.convert` and `models_dir/.hf` are ever removed (`_rmtree` resolves the path
  and refuses anything else, never follows links). Nothing else under `models_dir` is touched except the library
  file of a deleted `convert` item, and only when it is a regular file directly inside `models_dir`.
- A folder given as the source is never modified: the converter reads a staging folder of links, and `local_files`
  does not enter symlinked directories.
- A `needs_review` job keeps only its output file in `.convert/<job>/`; its cache is released.
- If registering the file in the library fails after the move, the file is removed again, so no unregistered file
  stays under a taken name. A cleanup problem after a successful publish is logged and does not fail the job.

### 17.7 Security

`convert_hf_to_gguf.py` loads tokenizers with `trust_remote_code=True`, which executes `*.py` files found next to
the weights. A repository's Python code would therefore run inside the coordinator, with its permissions. So repo
`*.py` files are never downloaded or staged unless the request sets `allow_remote_code`, the converter runs offline
and sees only the staging folder, and the UI marks the option as dangerous. The same flag is passed to the Hugging
Face tokenizer of the validation step. The image runs
the toolchain from `/opt` (llama.cpp's converter and a CPU-only PyTorch venv, plus static CPU builds of
`llama-quantize`, `llama-tokenize`, `llama-simple`, `llama-imatrix`); `WITH_CONVERT=0` leaves all of it out.

### 17.8 Importance matrices and the calibrate stage

**What and why.** Low-bit quantization has few levels per weight, so which weights get the precise ones matters. An
importance matrix is a per-weight estimate of how much each weight affects the output on typical text, measured by
running the model (`llama-imatrix`) over a calibration text; `llama-quantize --imatrix` then spends its precision where
it counts. llama-quantize b11342 even refuses the IQ1, IQ2, `IQ3_XXS` and `IQ3_XS` types without one (`IQ2_M` and
`IQ3_XS` files contain `IQ2_XS` / `IQ3_XXS` tensors), which is `QuantOption.needs_imatrix`. The matrix is computed from
the **16-bit intermediate**, not from the quantized file, so it sees the model's real activations; the CPU runs it
(`-ngl 0`, GPUs stay for inference, like the rest of the toolchain) at low priority.

**Policy** (`quant.imatrix_wanted`, `ConvertManager._decide_imatrix`, decided at submit and stored as
`imatrix_used`):

| `imatrix` | Types written by the converter (`F16`, `BF16`, `Q8_0`) | A type that needs a matrix | Other types |
| --- | --- | --- | --- |
| `auto` | none | yes | yes when `bpw` < 4.0, else no |
| `on` | none | yes | yes |
| `off` | none | **422**, the quantizer would fail | no |

Why `auto` is "needs it or under 4 bits per weight": the benefit grows as bits shrink (the fewer levels there are, the more the choice of
which weights stay precise matters), while at 4 bits and above the quality gain is small and the cost
is large: calibration is by far the slowest stage on a CPU (measured, Qwen2.5-1.5B to `IQ3_M`: 1175 s of 1687 s).
So `Q4_K_M` and up are not slowed down by default, and `on` stays available for anyone who wants it. `Q8_0`, `F16`
and `BF16` have no llama-quantize step to feed.

**The tool is optional, and the failure is where it can be explained.** `llama-imatrix` is the one part of the
toolchain a coordinator can lack while the rest works (a hand-built install; `imatrix_available` is false and the UI
disables the types that need it). A job that cannot work without it, a type that needs a matrix or `on`, is refused at
submit with 503 and a clear message, not minutes later inside `llama-quantize`. A job that merely would have
benefited (`auto` on a type like `Q3_K_S`) runs without it, silently, because failing a conversion for lack of an
optional improvement would be worse than the missing improvement. The same reasoning refuses bad calibration text at
submit (422: not a `.txt`, empty, over 20 MB) and a missing built-in text with 503.

**The calibration text and why it is original.** The usual public calibration sets are an English encyclopedia
dump (`wikitext`) or community collections (such as `calibration_datav3`) whose licences are unclear or
share-alike, which a project that ships and redistributes the file should not take on. The text is also a design
choice: a matrix computed on narrow text (one language, no code) makes the quantizer protect the weights that text
uses and be careless with the rest, so a model calibrated on English prose can get worse in Vietnamese or code. So gpupool
ships `converter/data/calibration.txt` (about 115 KB, written for gpupool, no real personal data): English prose in
many registers, Vietnamese with full diacritics, other languages, code in many languages, math and structured
data, chat-formatted dialogue, and edge content (emoji, odd whitespace, mixed scripts). Its README gives the
composition and the licence (the project's). A different `.txt` of at most 20 MB can be given as
`advanced.calibration_path` (translated like library paths); either way the file is **copied into the job's scratch
folder** when the stage starts, so one stable text is read even if the original changes or disappears meanwhile.
(The text sits under `data/`, which `.gitignore` also uses for runtime data folders; the first commit missed it for
exactly that reason, and a fresh checkout would have refused every matrix job with 503. The ignore rule now has an
exception for the package folder; see TEST_REPORT.)

**The run.** `llama-imatrix -m intermediate -f calibration.txt -o imatrix.gguf --chunks N -c 512 --no-ppl -ngl 0
[-t threads]`. `N` is `advanced.imatrix_chunks`, default 100; the context is 512 tokens because short sequences keep
a CPU run feasible; `--no-ppl` skips the perplexity pass, which only costs time. With `--no-ppl` the tool prints no
per-chunk lines (in b11342 they are inside the perplexity branch), so progress is estimated from the time of the first
pass and the ETA it prints, advanced once a second (never reaching 100 % from the clock alone: the file is written
after the last chunk); per-chunk lines, if a future version prints them, take over. The result must exist and
be non-empty, else the job fails.

**Memory and time (measured).** One job's peak resident memory was 1045 MiB (Qwen2.5-0.5B, `Q4_K_M`) and 1697 MiB
(Qwen2.5-1.5B, `IQ3_M`, with calibration); the cgroup peaks of 3.0 and 3.4 GB include page cache. See TEST_REPORT.
