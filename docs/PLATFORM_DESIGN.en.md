# gpupool — Multi-model, multi-server platform design (implemented)

> English version. Vietnamese version: [PLATFORM_DESIGN.vi.md](PLATFORM_DESIGN.vi.md). Keep both in sync.

**Status: implemented.** All four rollout phases (section 8) are in the code, plus two later additions that
the design called for in its open questions: per-model VRAM self-calibration and persisted control state.
This is a design-rationale document: why the system behaves as it does. For the exact HTTP interface see
[API.en.md](API.en.md).

Goal: gpupool manages **many models on many servers and GPUs** at once, shares resources deliberately
(priority, spreading, load-based scaling) and **recommends GPUs/servers** for each model with reasons
and estimates. This document covers the behaviour that motivated the design, the resource model, the
allocation algorithm, a summary of the API, the data changes and the rollout.

## 1. Baseline: behaviour with several models before this design

Measured with the **real** reconciler and scheduler as they were before phase 1 (fake agents, real GGUF
metadata of Qwen2.5 0.5B = 587 MB and 3B = 2191 MB at ctx 4096). Each problem below has since been
addressed by the section that follows it.

| Scenario | What happened | Problem |
| --- | --- | --- |
| 2 replicas of `chat`, server a: 2×8 GB, b: 1×8 GB | both replicas on **a/CUDA0** | no fault tolerance (one GPU dies = model gone); two replicas compete for one GPU while two GPUs idle |
| `alpha` and `zeta` (3B), one 4 GB GPU fits one | `alpha` runs, `zeta` NoFit | the winner is decided by **name order**; no way to say which model matters |
| `big` 3B + `small` 0.5B, a: 8 GB + 8 GB | both on **a/CUDA0**, CUDA1 empty | the scheduler only sees memory; two active models halve one GPU's bandwidth |
| `small` then `big`, a: 3 GB + 2.4 GB | `big` → CUDA0, `small` → CUDA1 | correct: memory best-fit works |

Root causes, from the code at that time:

- `Reconciler._enforce_counts` walked models `ORDER BY name`, one replica per model per tick. There was no
  priority and no preemption.
- `scheduler.placement.plan` was a pure function that only saw `usable_mb`. It did not know where other
  replicas run, which GPU is faster or which GPU is busy. Tier `single_gpu` picked the **smallest GPU that
  still fits** (best-fit), which naturally piles everything onto the same GPU.
- `replicas` was a fixed number. An unused model held VRAM forever, and an overloaded model never gained a
  replica.
- NoFit was all-or-nothing: it did not suggest a smaller `ctx_size`, nor say how much is missing, or where.

## 2. Resource model

A GPU has two kinds of resources, and they behave differently:

| Resource | Nature | Source | Use |
| --- | --- | --- | --- |
| VRAM | **hard**: short means it cannot load | `usable_mb`, `est_mb` estimate (`scheduler/estimate.py`), corrected per model by the calibration factor (section 4.7) | hard constraint |
| Memory bandwidth | **soft**: sharing slows it down | NVML: bus width × mem clock (measured on GTX 1650 Ti: 128 bit × 5001 MHz ≈ 160 GB/s) | tok/s estimate, performance score |
| Busyness | **soft**, changes over time | llama-server `/metrics`: `requests_processing`, `requests_deferred`, `predicted_tokens_seconds` (checked on b11342); router `outstanding` | co-location penalty, autoscale signal |

**Decode speed estimate.** Each generated token reads all weights once, so decode is bandwidth-bound:

```latex
t_{token} = \sum_{d \in \text{devices}} \frac{\text{bytes}_d}{BW_d \cdot \eta} + n_{rpc} \cdot t_{hop}
\qquad \text{tok/s} \approx 1 / t_{token}
```

`bytes_d` is the sum of `layer_bytes` of the layers placed on device d (already in `ModelMeta`; the output
tensors count on the last device). η is the real-world efficiency: the numbers in `TEST_REPORT` give
η ≈ 0.45 for 0.5B (182 tok/s against ~400 in theory) and ≈ 0.6 for 3B (51 tok/s against ~84). The code uses
a constant **η = 0.5** and `t_hop` = 2 ms per RPC hop (`scheduler/scoring.py`). A GPU without a known
bandwidth gets a default. Prefill depends on compute more than on bandwidth; the estimate ranks by decode
only. Measured decode speed is read from `/metrics` and shown next to the estimate in
`GET /api/models/{name}/scaling`, but it does **not** feed back into η (see open questions).

**Sharing a GPU.** Two busy models on one GPU split its bandwidth, each getting about half its speed. A
model that is mostly idle can share fine. The co-location penalty is therefore based on **measured
busyness**, not a hard ban.

## 3. Per-model policy

All fields are optional, and their defaults keep the behaviour of a plain fixed-replica model. The full
field table, with types and limits, is in [API.en.md](API.en.md#2-modelspec).

| Field | Default | Meaning |
| --- | --- | --- |
| `priority` | 50 | 0–100. Higher priority places first and, when short on room, may take the place of lower priority |
| `min_replicas` | = `replicas` | replicas always kept (0 = may be unloaded when idle) |
| `max_replicas` | = `replicas` | > `min_replicas` enables autoscaling |
| `autoscale` | `{target_busy: 0.7, up_after_s: 30, down_after_s: 300}` | thresholds for adding/removing replicas by busy-slot ratio |
| `idle_unload_s` | null | only with `min_replicas = 0`: unload after N s without requests; the first request loads it again |
| `spread` | `"gpu"` | `gpu`: replicas prefer different GPUs; `node`: different servers; `none`: don't care |
| `pin_devices` | empty | `"node_id/device_id"` entries a replica may use; empty = free choice |
| `preemptible` | true | false = never preempted |
| `kv_cache_type`, `speculative`, `draft`, `draft_n_max` | `f16`, `none`, null, 4 | memory/speed options that the estimate and the planner take into account (added after this design) |

`replicas` is the on/off switch: 0 stops the model whatever `min/max` say. With `min_replicas` and
`max_replicas` unset both equal `replicas`, so a fixed-count model behaves as before.

**Decided not to build.** The first draft also proposed `share_gpu`, `gpu_selector` (labels,
`min_vram_mb`, `min_bandwidth_gbps`), server/GPU **labels** and a per-GPU `reserve_mb`. They are not in the
code. `pin_devices` covers explicit placement, and the per-device `budget_mb` on the agent (which also
subtracts the VRAM gpupool's own replicas hold) covers keeping VRAM back for other work. Labels would be
added only if a cluster appears where pinning single devices is too coarse.

## 4. Allocation algorithm

```mermaid
flowchart LR
  A[Autoscaler: desired replicas per model] --> B[Order models: priority, models with no replica first]
  B --> C[Generate feasible placements]
  C --> D[Score and pick the best]
  D -->|none feasible| E[Try preempting lower priority]
  D --> F[Launch]
  E --> F
  G[Rebalancer, infrequent] -->|make-before-break| F
```

**4.1 Order.** Each tick, sort models by `priority` descending. Within a level, models with **no replica
yet** go before models that already have one (every model gets 1 replica before any gets a second), then by
**name** (a stable tie-break; the first draft said creation time). Name order only breaks ties now.

**4.2 Candidate placements.** `scheduler/placement.py` generates several feasible candidates instead of one
per tier: every single GPU that fits, multi-GPU combinations within a node, and multi-node placements (only
when needed; the greedy variants first, then subsets by bandwidth). `plan()` returns the best one with
ports allocated; `rank()` returns the top `k` without ports, which `recommend`, `simulate` and rebalancing
use. Real clusters have few GPUs (tens), so enumeration is cheap. The layer split algorithm (`_split`) is
unchanged. A speculative `draft` model is placed whole on the head's first CUDA device and counted in that
device's `est_mb`.

**4.3 Scoring.** Each candidate gets a score; higher wins. The weights are constants in
`scheduler/placement.py`, not configuration:

```latex
\text{score} = 100 \cdot \frac{tps}{tps_{best}} - \sum_{o \in \text{engines on chosen GPUs}} (10 + 30 \cdot busy_o) - 40 \cdot same_{gpu} - 20 \cdot same_{node} - 15 \cdot waste - 5 \cdot (n_{dev}-1) - 10 \cdot n_{rpc}
```

- `tps / tps_best`: estimated decode speed relative to the fastest candidate.
- Each engine already on a chosen GPU costs 10, plus 30 times its busyness (0–1).
- `same_gpu`, `same_node`: replicas of the same model already on that GPU / server (per `spread`: `gpu` and
  `node` penalise a shared GPU; only `node` penalises a shared server).
- `waste`: mean of `usable / biggest usable` over the chosen GPUs. This keeps best-fit's strength: do not
  carve up a large GPU while a snug one is free.
- `n_dev`, `n_rpc`: extra devices and network hops.

Ties break by smaller tier, then the first device's name, so results are deterministic. The winning
placement is stored with the replica (`score`, `est_decode_tps`, `reasons`), so the UI can explain why a
model runs where it does.

**4.4 Preemption.** A model below its running minimum (`max(min_replicas, 1)`, bounded by the desired
count, so autoscale extras never evict) and with no feasible placement may stop lower-priority replicas:

1. Eligible victims: replicas of `preemptible` models with priority **strictly lower** than P (equal never
   preempts). Order: replicas above their model's running minimum first, then the least busy, then the
   newest.
2. Add victims one by one to a simulated "removed" set until a placement exists, then drop any victim that
   turns out to be unnecessary. The result is a small set, not necessarily the smallest possible.
3. Drain them (the existing `drain`: wait for in-flight requests, at most `drain_timeout_s`). The
   preempting model is not placed in the same tick; draining replicas still hold their memory, so a later
   tick places it. A `preempted` warning event is emitted per victim.
4. **Cooldown** of 10 minutes per preempting model, and no further eviction while its earlier victims still
   drain, so two models cannot keep evicting each other.
5. **Claim.** While a preempting model has no replica yet (at most `drain_timeout_s` + 120 s) models of
   lower priority do not launch. Without this, the evicted model relaunches straight into the memory being
   freed (a bug found on real hardware).

**4.5 Autoscaler.** Each replica has `busy = requests_processing / parallel`, read from llama-server
`/metrics` every `poll_s` (a scrape older than 3 polls is replaced by the router's outstanding count).
`requests_deferred > 0` means requests are queueing.

- Add 1 replica when average `busy` > `target_busy`, or requests queue, for `up_after_s` and no replica is
  still launching. Never above `max_replicas`.
- Remove 1 replica when `busy` < `target_busy / 2` and nothing queues for `down_after_s`. Never below the
  running floor `max(min_replicas, 1)`: going to zero is only done by `idle_unload_s`.
- `idle_unload_s`: no request for that long and nothing in flight, with `min_replicas = 0` → desired 0.
- **Cold start**: when the router gets a request for a started model with `min_replicas = 0` and no replica,
  the autoscaler sets desired to 1, emits `cold_start` and wakes the reconciler; the router holds the request
  up to `cold_start_timeout_s` (default 120 s). Past that, it returns 503 with `Retry-After`.
- **Persistence.** The desired count, last request time and last decision are written to the store
  (`control_state`, key `autoscaler:<model>`) when they change, so a restart keeps an unloaded model
  unloaded and a scaled-up model scaled up. The up/down timers are not saved: a restart only delays a step by
  `up_after_s`/`down_after_s`.

**4.6 Rebalancing.** Runs rarely (every `rebalance_s`, default 600 s, 0 = off; the first run is one period
after boot because reports are stale right after a restart) or on demand (`POST /api/rebalance`). Target:
replicas that now score at least 25 points higher somewhere else (e.g. a model on CPU or split over 2 GPUs
that now fits on 1 GPU); a move stays within the model's `pin_devices`. The method is **make-before-break**:
launch the new replica (planned on the target devices while the old one still holds its memory), wait for
ready, then drain the old one. Only one replica moves at a time cluster-wide, and only when the cluster is
quiet: no replica launching, no preemption waiting for its memory. A move whose new replica fails, or does
not become ready within `launch_timeout_s` + 60 s, is abandoned with a `rebalance_failed` warning and the old
replica keeps serving.

**4.7 VRAM self-calibration.** The memory estimate is exact to the MiB for the cases measured on a GTX 1650,
but was never measured for large models over several GPUs. Instead of trusting it, each model learns a
correction. After a replica is ready, the coordinator reads what llama.cpp reported when it loaded
(`GET /engines/{id}/memory` on the head's agent), sums the measured buffers over the placement's devices and
compares with the estimate (divided by the factor it was planned with, minus the runtime context llama.cpp does not report). The ratio is folded into a
per-model factor by exponential moving average (weight 0.5) and stored in `model_calibration`. Planning
multiplies the model's device needs by the factor, clamped to 0.9–2.0. A `calibrated` event is emitted when
the factor moves by more than 5 %; a sample with partial data is skipped rather than biasing the factor. The
factor is visible as `calibration` in `GET /api/state`.

**4.8 Persisted control state.** Besides the autoscaler's state, the reconciler writes to `control_state`
the preemption cooldowns and claims (`preempted`), crash-loop backoffs (`backoff`) and the replica move in
flight (`move`), and loads them at boot. A coordinator restart therefore does not forget a cooldown (letting
two models evict each other again), a backoff (hammering a crashing model) or half a move (leaving two
replicas of the same model). A move whose replicas no longer exist is cleared quietly. Times are wall-clock.
The last-rebalance time is deliberately not saved.

## 5. GPU/server recommendation

Answers "where should this model run?" **before** deploying, with the same scoring as the scheduler, so
what is recommended is what the scheduler would do.

- Ranks up to `limit` options, each with expected VRAM, estimated tok/s, score and reasons.
- Says whether an option fits now or needs to preempt someone (`requires_preemption`, listing the
  replicas to stop).
- Reports the largest `ctx_size` that fits one GPU now (`max_ctx_single_gpu`).
- When nothing fits, says why: the memory needed, the largest free single GPU and node, and the largest
  `ctx_size` that would fit (`not_possible`).

## 6. API summary

The full reference, with request and response shapes, is [API.en.md](API.en.md). All design endpoints live
under `/api` and take the admin key; new request fields are optional, so older clients keep working.

| Capability | Endpoints |
| --- | --- |
| Model policy (sections 3, 4.5) | `PUT /api/models/{name}`, `POST .../start`, `POST .../stop`, `DELETE /api/models/{name}`, `GET .../scaling`, `POST .../plan` |
| Capacity and recommendation (section 5) | `GET /api/capacity`, `POST /api/recommend` |
| Simulation (section 4.4) | `POST /api/simulate`: starts, stops and preemptions the reconciler would do under hypothetical changes |
| Rebalancing (section 4.6) | `POST /api/rebalance` (`dry_run` defaults to true) |
| State and events | `GET /api/state` (includes `calibration`, `rebalance`), `GET /api/events`, `POST /api/events/read` |
| Router | `/v1/*` cold start returns 503 + `Retry-After` past `cold_start_timeout_s` |

Events added by the design: `preempted` (warning), `scaled_up`, `scaled_down`, `unloaded_idle`,
`cold_start`, `rebalance_started`, `rebalanced`, `calibrated` (info), `rebalance_failed` (warning).

Differences from the first draft: there are no label endpoints (`PUT /api/servers/{id}/labels`) and
`PUT .../gpus/{device_id}` only takes `enabled`; `POST /api/simulate` reports `start`, `stop`, `preempt` and
`unplaced` but no `move` (rebalance moves are previewed with `POST /api/rebalance`); `not_possible` carries
`need_mb`, `largest_single_gpu_mb`, `largest_single_node_mb` and `max_ctx_that_fits`.

## 7. Data changes

- `ModelSpec`: the policy fields (JSON in the `models` table, no migration), later also `kv_cache_type`,
  `speculative`, `draft`, `draft_n_max`.
- `Device`: `bandwidth_gbps` (the agent reads it from NVML), `uuid`, `budget_mb`. Old agents omit them, and
  their GPUs count as equal.
- `Placement`: `score`, `est_decode_tps`, `reasons`, `draft_est_mb`.
- `gpu_flags` is keyed by the card's uuid when reported. There is no `server_labels` table and no
  `labels`/`reserve_mb` column (see section 3).
- Table `control_state(key, value, updated_at)`: autoscaler state per model, `preempted`, `backoff`, `move`.
- Table `model_calibration(model, factor, samples, updated_at)`.
- The last request time per model lives in the autoscaler (persisted at a slow cadence); the router only
  reports requests to it.

## 8. Rollout

| Phase | Content | Risk | Status |
| --- | --- | --- | --- |
| 1. Foundation | `priority` and fair ordering; `spread`; busy-GPU co-location penalty; `bandwidth_gbps` + tok/s estimate; candidate generation + scoring; `GET /api/capacity`; `POST /api/recommend`; recommendation panel in the "New model" form | low: never removes a running replica | done |
| 2. Scaling | read llama-server `/metrics`; `min/max_replicas` + autoscaler; `idle_unload_s` + router cold start; `GET /scaling` | medium: replicas added/removed automatically | done |
| 3. Preemption | preemption + cooldown + claim; `POST /api/simulate` | medium: removes serving replicas, draining must be right | done |
| 4. Rebalancing | make-before-break rebalance; `POST /api/rebalance` | highest: reloading large models takes time | done |
| After 4 | VRAM self-calibration (4.7); persisted control state (4.8) | low | done |

**Phase 1 status:** done. Real run on a GTX 1650: a `priority` 80 model wins over one whose name sorts first; replicas and new models spread to other GPUs (simulated with the real scheduler); estimated vs measured decode 41.6 vs 51.8 tok/s (3B) and 204 vs 182 (0.5B). `budget_mb` now also subtracts the VRAM gpupool's own replicas hold. Spreading has not been checked on real multi-GPU hardware.

**Phase 2 status:** done. Real run on a GTX 1650: an on-demand 3B model unloaded after 20 s without requests, and the next request cold-started it and got its answer after 2.9 s; a 0.5B model (min 1, max 2) went to 2 replicas about 9 s after requests started queueing, 82 requests split 49/33 between them, and it went back to 1 after 16 s without load. When this was measured, autoscaling state was in memory only and a coordinator restart sent every model back to its running minimum; it is now persisted (4.5, 4.8).

**Phase 3 status:** done. Real run on a GTX 1650 (budget 2500 MB): `/api/simulate` predicted exactly that `q05` (priority 20) would be stopped to run `q3b` (priority 80), `/api/recommend` returned a `fits_now: false` option naming the replica to stop, and with `q05` non-preemptible it reported `q3b` as unplaceable. Real preemption: `q3b` ran 23.5 s after start, and `q05` could not evict it back. The first run exposed a bug: the evicted replica turns draining, so its model relaunched straight into the memory being freed. Fixed with a claim: while the preempting model has no replica yet (at most `drain_timeout_s` + 120 s), lower priorities do not launch.

**Phase 4 status:** done. Real run with two agents on one machine (a: GPU capped at 1000 MB, b: CPU only): replica `y` running on b/CPU (~30 tok/s estimated) was moved to a/CUDA0 (~204 tok/s) once the GPU freed up, score +100, in about 9.5 s. During the move a client sent 23 requests back to back and none failed. Afterwards a new check proposed nothing. The periodic run follows `rebalance_s` (default 600 s, 0 = off).

Each phase is checked on a real cluster, with at least 2 servers and 2 models, before the next one starts.

## 9. Open questions and decisions

Decided:

- **VRAM estimate accuracy for large models over several GPUs.** Not measured by hand; instead every model
  calibrates itself from what llama.cpp really allocated (4.7). The estimate stays the starting point, the
  measured factor corrects it per model.
- **Labels, `share_gpu`, `gpu_selector`, `reserve_mb`.** Not built (section 3).
- **Restart behaviour.** Control state is persisted (4.8); only the rebalance timer restarts on purpose.

Still open:

- **η per GPU architecture.** The only numbers come from one GTX 1650 and the code uses a fixed η = 0.5. The
  measured decode speed is collected, so per-model η self-calibration is possible, but it is not
  implemented. Datacenter GPUs may differ a lot, so it should be done before relying on tok/s ranking there.
- **Calibration on large models.** The mechanism exists, but a real ≥ 7B model over several GPUs has not been
  run through it yet; the clamp 0.9–2.0 and weight 0.5 are untuned.
- **Prefill** is compute-bound (SMs × clock), not bandwidth-bound. Models that mostly take long prompts may
  need a separate performance score.
- **Scoring weights** in 4.3 are a starting point, constants in code, to be tuned against real
  measurements.
- **Spreading and rebalancing on real multi-GPU, multi-server hardware** are verified only with simulated
  agents and a single-GPU machine.
