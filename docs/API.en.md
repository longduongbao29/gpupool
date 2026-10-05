# gpupool HTTP API reference

> English version. Vietnamese version: [API.vi.md](API.vi.md). Keep both in sync.

This reference was written from the code (`src/gpupool/coordinator/api.py`, `app.py`, `library_api.py`,
`src/gpupool/coordinator/convert_api.py`, `src/gpupool/converter/models.py`, `src/gpupool/router/proxy.py`,
`src/gpupool/agent/app.py`, `src/gpupool/common/models.py`). It lists every
HTTP route the two services expose: **51 routes** (coordinator 44, agent 7), plus the static web UI that the
coordinator mounts at `/`. The full list is in section 11.

| Scanned file | Routes |
| --- | --- |
| `coordinator/api.py` (`/api`) | 16 |
| `coordinator/library_api.py` | 6 (5 library routes, 1 file server) |
| `coordinator/app.py` | 10 (`/healthz`, `/metrics`, 2 internal, 6 `/admin`) |
| `coordinator/convert_api.py` (`/api/convert`) | 9 |
| `router/proxy.py` (`/v1`) | 3 |
| `agent/app.py` | 7 |


## 0. Conventions

### Authentication

Three independent secrets exist. A route that needs a secret that is configured empty is **open** (dev mode,
`require_bearer` returns without checking when no non-empty key is configured).

| Name | Config | Header | Protects |
| --- | --- | --- | --- |
| API key | `api_keys` / `GPUPOOL_API_KEYS` (list) | `Authorization: Bearer <key>` | `/v1/*` (any one key of the list works) |
| Admin key | `admin_key` / `GPUPOOL_ADMIN_KEY` | `Authorization: Bearer <admin key>` | `/api/*`, `/admin/*` |
| Cluster token | `cluster_token` / `GPUPOOL_CLUSTER_TOKEN` | `Authorization: Bearer <token>` | `/internal/*`, `/files/{name}`, and every agent route except `GET /health` |

When the coordinator is started with `gpupool coordinator`, an empty admin key and cluster token are
generated and stored in `secrets.json` next to the database. A wrong or missing token gives
`401 {"detail": "invalid or missing bearer token"}` (for `/v1/*` the OpenAI error shape below).

`GET /healthz` (coordinator), `GET /health` (agent), `GET /metrics` and the static UI need no token.

### Errors

- Coordinator and agent routes use FastAPI errors: `{"detail": "<message>"}` with the status code; request
  validation failures are `422` with the FastAPI `detail` list.
- `/v1/*` routes use the OpenAI error shape: `{"error": {"message": "...", "type": "...", "code": "..."}}`.
- Planning failures (`/api/models/{name}/plan`, `/api/simulate`, `/api/recommend`) map the exception to a
  status: `NoFit` is `409`, a missing model file is `404`, anything else `400`. The detail is
  `"<ExceptionType>: <message>"`.

### Examples

```bash
export COORD=http://localhost:8080
export ADMIN=...   # admin key
curl -s -H "Authorization: Bearer $ADMIN" $COORD/api/state | jq .summary
```

## 1. OpenAI-compatible API (`/v1`)

Served by the coordinator's router. Auth: API key (open when `api_keys` is empty).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/v1/models` | list the model names (every registered model, running or not) |
| POST | `/v1/chat/completions` | chat completion, proxied to a ready replica |
| POST | `/v1/completions` | text completion, proxied to a ready replica |

**`GET /v1/models`** returns `{"object": "list", "data": [{"id": "<name>", "object": "model", "owned_by":
"gpupool"}]}`.

**`POST /v1/chat/completions` and `POST /v1/completions`**: the body is the llama-server (OpenAI) request,
forwarded unchanged except that `cache_prompt` defaults to `true`. Rules the router adds:

- The body must be a JSON object with a string `model` that names a registered model.
- Body size is capped by `max_request_mb` (default 32 MB), also while a chunked upload streams in.
- `"stream": true` returns `text/event-stream`, passed through as received. If the upstream breaks after the
  stream started, one SSE `data:` frame with an `error` object is sent, then the stream ends.
- Replica choice: prompt-prefix affinity first (rendezvous hash on the conversation prefix, so the same prefix goes to the same
  replica, for the llama.cpp prompt cache; a multi-turn chat keeps its replica), weighted by each replica's estimated speed so a
  faster replica gets a larger share, falling back to the least-loaded replica (requests in flight relative to speed) when the
  preferred one is more than 2 ahead.
- A failed attempt (connection error or upstream status >= 500) is retried on another replica, at most
  2 retries (3 attempts), only while nothing has reached the client.
- **Cold start**: if the model has `min_replicas = 0`, is started (`replicas > 0`) and has no ready replica,
  the request is held up to `cold_start_timeout_s` (default 120 s, config) while the model loads. The client
  disconnecting ends the wait.

| Status | `error.code` | When |
| --- | --- | --- |
| 400 | `invalid_json` | body is not valid JSON |
| 400 | `missing_model` | not an object, or `model` missing or not a string |
| 401 | `invalid_api_key` | wrong or missing API key |
| 404 | `model_not_found` | `model` is not registered |
| 413 | `request_too_large` | body over `max_request_mb` |
| 502 | `upstream_error` | all attempts failed (`upstream failed: ...` / `all replicas failed: ...`) |
| 503 | `no_replica` | the model has no ready replica and is not allowed to cold start |
| 503 | `model_loading` | cold start did not finish in `cold_start_timeout_s`; has `Retry-After: 10` |

```bash
curl -s $COORD/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen-3b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
```

## 2. ModelSpec

`ModelSpec` (in `common/models.py`) is what the coordinator stores per model. The API accepts it in two
shapes: the full `ModelSpec` on `POST /admin/models`, and the friendlier `ModelBody` on
`PUT /api/models/{name}` (section 3). `GET /api/state` returns it as `models[].spec`.

| Field | Type, default | Valid values and meaning |
| --- | --- | --- |
| `name` | string, required | alias served at `/v1/models`. Through `/api/models/{name}`: `[A-Za-z0-9._-]{1,64}` |
| `source` | string, required | `https://...gguf`, an absolute path on the coordinator, or `coordinator://<file in library>`. `/api` always writes `coordinator://<file>` |
| `ctx_size` | int, 4096 | total context in tokens, divided across `parallel` slots (llama.cpp `-c`); `>= 1` through `/api` |
| `parallel` | int, 1 | concurrent slots (llama.cpp `-np`); `>= 1` through `/api`. Autoscaling busyness is `processing / parallel` |
| `replicas` | int, 1 | desired count; **`0` = stopped**. It is the on/off switch: with autoscaling it stays `> 0` and `min/max` bound the count |
| `pin_devices` | list of `"node_id/device_id"` or `"node_id/*"`, empty | the **allowed set** of devices: replicas may only use these; empty = every server and GPU. `"node_id/*"` allows every device of that server, **including GPUs added to it later**. It is a limit, not a placement: the scheduler still chooses the best placement among the allowed devices (it applies during preemption and rebalancing too), and the Servers tab (GPUs enabled or disabled in the pool) is unaffected. `/api` rejects entries that do not match `node/device` (or `node/*`) or name an unregistered server (422). The UI shows this as "All servers and GPUs / Only selected ones" |
| `priority` | int 0..100, 50 | higher places first each tick and may preempt lower priorities |
| `preemptible` | bool, true | `false`: a higher-priority model never stops this model's replicas |
| `spread` | `"gpu"` / `"node"` / `"none"`, `"gpu"` | soft: replicas of this model avoid sharing a GPU / a server (a shared GPU is still used when nothing else fits) |
| `min_replicas` | int `>= 0` or null, null | autoscale floor; null = `replicas` (fixed count). `0` allows scale-to-zero |
| `max_replicas` | int `>= 1` or null, null | autoscale ceiling; null = `replicas`. Autoscaling is active when `max > min` |
| `autoscale` | object or null, null | `{target_busy: 0.7 (0,1], up_after_s: 30 >= 0, down_after_s: 300 >= 0}`; null = those defaults. Add a replica when average busy slots / slots stays above `target_busy` for `up_after_s` (or requests queue); remove one when below `target_busy / 2` for `down_after_s` |
| `idle_unload_s` | float `> 0` or null, null | only with `min_replicas == 0`: unload after this many seconds without a request; the next request cold-starts it |
| `kv_cache_type` | `"f16"` / `"q8_0"` / `"q4_0"`, `"f16"` | KV cache element type (`-ctk/-ctv`). Bytes per element 2 / 34/32 / 18/32, so q8_0 / q4_0 roughly halve / quarter the KV memory |
| `speculative` | `"none"` / `"ngram"` / `"draft"` / `"mtp"`, `"none"` | speculative decoding: `ngram` guesses from the text so far (no extra memory); `draft` runs a small model with the same tokenizer on the head's first GPU; `mtp` drafts with the model's own multi-token-prediction (nextn) layers, 422 when the GGUF has none |
| `draft` | string or null, null | source of the draft model, `coordinator://<file>`; only for `speculative: "draft"` (dropped otherwise) |
| `draft_n_max` | int 1..16, 4 | tokens drafted per step. On a GTX 1650 (3B + 0.5B draft) 4 gave +5 %, 8 was slower than none |
| `flash_attn` | `"auto"` / `"on"` / `"off"`, `"auto"` | llama.cpp `-fa`. Auto turns it on where the GPU supports it. A quantized `kv_cache_type` needs it (`off` with q8_0/q4_0 is a 422) |
| `ubatch` | int 32..8192, 512 | micro-batch (`-ub`): prompt tokens per pass. Bigger reads long prompts faster on GPUs with tensor cores (cc 7.0+); the compute buffer, charged on every device, grows with it |
| `batch` | int 32..16384, 2048 | logical batch (`-b`); raised to `ubatch` when smaller |
| `kv_unified` | bool, false | `-kvu`: the `parallel` slots share one KV pool, so one request may use up to `ctx_size` tokens while the others are short (false: each slot owns `ctx_size / parallel`); same memory |

Validation applied by `PUT /api/models/{name}` and by `/api/simulate` (the same checks):

- `min_replicas <= max_replicas` (422).
- `idle_unload_s` needs `min_replicas == 0` (422).
- `speculative: "draft"` needs a `draft_file` that is a **ready** library item, different from the model,
  with a tokenizer model equal to the model's and a vocabulary size at most 128 tokens away (422 otherwise).

Not part of `ModelSpec` (they exist in the design discussion but are not implemented): `share_gpu`,
`gpu_selector`, server/GPU labels, `reserve_mb`.

## 3. Models (`/api/models`)

All under `/api`, admin key.

| Method | Path | Purpose |
| --- | --- | --- |
| PUT | `/api/models/{name}` | create or update a model spec (does **not** start it) |
| POST | `/api/models/{name}/start` | set `replicas` (start) |
| POST | `/api/models/{name}/stop` | set `replicas = 0` (replicas drain) |
| DELETE | `/api/models/{name}` | drain replicas and delete the spec |
| GET | `/api/models/{name}/scaling` | autoscaler view |
| POST | `/api/models/{name}/plan` | dry-run placement of one replica |

### `PUT /api/models/{name}`

Body `ModelBody`. Fields marked "keep" keep the stored value when omitted (or `null`) on an existing
model; a new model gets the default.

| Field | Type | Default / keep |
| --- | --- | --- |
| `file` | string, required | library item name; must be `ready` (422 otherwise). Stored as `source = coordinator://<file>` |
| `ctx_size` | int `>= 1` | 4096 (always overwritten) |
| `parallel` | int `>= 1` | 1 (always overwritten) |
| `pin_devices` | list of string | empty (always overwritten, duplicates removed); `"node_id/*"` is allowed, see section 2 |
| `priority` | int 0..100 or null | keep, else 50 |
| `spread` | `gpu`/`node`/`none` or null | keep, else `gpu` |
| `min_replicas`, `max_replicas` | int or null | keep, else unset |
| `autoscale` | object or null | keep, else unset |
| `idle_unload_s` | float `> 0` or null | keep, else unset |
| `preemptible` | bool or null | keep, else true |
| `kv_cache_type` | enum or null | keep, else `f16` |
| `speculative` | enum or null | keep, else `none` |
| `draft_file` | string or null | keep (stored draft), else none |
| `draft_n_max` | int 1..16 or null | keep, else 4 |
| `flash_attn`, `ubatch`, `batch` | as above, or null | keep, else `auto` / 512 / 2048 |
| `kv_unified` | bool or null | keep, else false |

`replicas` is not in the body: a new model starts with `replicas = 0` and an existing model keeps its value.
Returns the stored `ModelSpec` (JSON). Errors: 422 (bad name, file not ready, bad pin, min > max, idle
without min 0, draft checks, `mtp` for a GGUF without nextn layers), 401. The reconciler is woken.

```bash
curl -s -X PUT $COORD/api/models/qwen-7b -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" -d '{
  "file": "qwen2.5-7b-instruct-q4_k_m.gguf", "ctx_size": 8192, "parallel": 4, "priority": 80,
  "min_replicas": 1, "max_replicas": 3, "spread": "node", "kv_cache_type": "q8_0"}'
```

### `POST /api/models/{name}/start`

Optional body `{"replicas": 1}` (`int >= 1`, default 1). Sets `replicas` and emits `model_started`. Returns
the `ModelSpec`. 404 if the model is unknown. With autoscaling the count is then governed by `min/max`.

```bash
curl -s -X POST $COORD/api/models/qwen-7b/start -H "Authorization: Bearer $ADMIN"
```

### `POST /api/models/{name}/stop`

No body. Sets `replicas = 0` (emits `model_stopped`); the reconciler drains the replicas. Returns the
`ModelSpec`. 404 if unknown.

### `DELETE /api/models/{name}`

Drains every active replica, then deletes the spec. Returns `{"ok": true}`. 404 if unknown. The library
file is kept (but can then be deleted from the library).

### `GET /api/models/{name}/scaling`

Autoscaler state of one model. 404 if the model is unknown (or `autoscaling not available`).

```json
{"model": "q05", "min": 0, "max": 2, "desired": 1, "ready": 1, "launching": 0,
 "avg_busy": 0.25, "queued": false, "idle_s": 12.4,
 "state": "steady", "last_decision": {"...": "..."},
 "replicas": [{"replica_id": "q05-1", "busy": 0.25, "requests_processing": 1, "requests_deferred": 0,
               "measured_decode_tps": 182.0, "est_decode_tps": 204.0, "metrics_ok": true}]}
```

`state` is one of `stopped` (`replicas == 0`), `fixed` (min == max, no idle unload), `unloaded` (desired 0),
`scaling_up`, `scaling_down`, `steady`. `busy` and the counters come from llama-server `/metrics` of the
head (`requests_processing`, `requests_deferred`, `predicted_tokens_seconds`); when a scrape is older than
3 poll intervals the router's own outstanding count is used and `metrics_ok` is false.

### `POST /api/models/{name}/plan`

No body. Where would one more replica go right now? No side effects (ports are only chosen within the call).
Returns a `Placement`:

```json
{"model": "qwen-7b", "replica_id": "...", "tier": "single_gpu", "head_node": "b", "head_port": 9001,
 "assignments": [{"node_id": "b", "device_id": "CUDA1", "llama_device": "CUDA0", "rpc_endpoint": null,
                  "layers": 28, "est_mb": 6120, "device_uuid": "GPU-..."}],
 "tensor_split": [1.0], "est_total_mb": 6120, "score": 91.0, "est_decode_tps": 41.2,
 "reasons": ["fastest GPU with room"], "draft_est_mb": null, "mem_factor": 1.0}
```

Errors: 404 unknown model, 409 `NoFit`, 404 model file missing, 400 other.

## 4. Servers, nodes and GPUs

All under `/api`, admin key. A "server" is a registered agent; `node_id` is its id.

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/servers` | register an agent by URL |
| DELETE | `/api/servers/{node_id}` | stop its replicas and unregister it |
| PUT | `/api/servers/{node_id}/gpus/{device_id}` | enable or disable one GPU |

### `POST /api/servers`

Body `{"agent_url": "http://host:7070"}` (must be `http(s)://host[:port]`). The coordinator probes the
agent's `/report` with the cluster token. Returns the server entry (same shape as `servers[]` in
`/api/state`). Errors: `400` (bad URL, agent unreachable, wrong token, invalid report), `409` already
registered. Clears an earlier removal mark (see `DELETE`).

```bash
curl -s -X POST $COORD/api/servers -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"agent_url": "http://10.0.0.5:7070"}'
```

### `DELETE /api/servers/{node_id}`

Marks every replica that touches the node as stopped (the desired count is re-placed elsewhere by the next
tick), forgets the server and remembers the removal, so the agent's `--join` does **not** bring it back
(`/internal/join` answers 403) until it is added again with `POST /api/servers`. Returns `{"ok": true}`.
404 if unknown.

### `PUT /api/servers/{node_id}/gpus/{device_id}`

Body `{"enabled": true|false}`. A disabled GPU receives no new replicas (it counts as usable 0). The flag is
stored by the card's UUID when the agent reports one (device ids shift when a GPU disappears), else by
`device_id`. Returns `{"node_id", "device_id", "enabled"}`. 404 if the server is unknown.

## 5. Model library and files

Library routes: admin key. `/files/{name}`: cluster token (it is how agents download a model).

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/api/hf/files?repo=owner/name` | admin | list `.gguf` files of a Hugging Face repo |
| GET | `/api/library` | admin | list library items |
| GET | `/api/library/browse` | admin | list `.gguf` files found under the model roots |
| POST | `/api/library` | admin | add an item (HF download or local path) |
| DELETE | `/api/library/{name}` | admin | remove an item and its files |
| GET | `/files/{name}` | cluster token | download a library file (internal) |

**`GET /api/hf/files?repo=owner/name`** returns `[{"file": "path/in/repo.gguf", "bytes": 123}]`, sorted.
Errors: `400` repo not `owner/name`, `403` gated or private (set `HF_TOKEN`), `404` repo not found, `502` HF
unreachable or unexpected answer.

**`GET /api/library`** returns a list of `LibraryItem`:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | unique file name, e.g. `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| `path` | string | absolute path on the coordinator |
| `source` | `"hf"` / `"path"` / `"convert"` | downloaded from HF, registered local file, or written by a conversion job ([section 12](#12-conversion-apiconvert)) |
| `hf_repo`, `hf_file` | string or null | HF origin |
| `bytes` | int or null | total size when known |
| `downloaded` | int | bytes written so far |
| `status` | `"downloading"` / `"ready"` / `"failed"` | only `ready` items can be used by a model |
| `error` | string or null | failure reason |
| `created_at` | float | unix time |

**`GET /api/library/browse`** returns `{"roots": [{"path", "exists", "host_path"}], "files": [{"path", "name",
"bytes", "in_library", "split_part", "broken_link", "host_path"}], "truncated": bool}`. It walks the model
roots (`model_roots`, default `models_dir`) up to depth 6 and 1000 files.

**`POST /api/library`** body: either `{"hf_repo": "owner/name", "hf_file": "file.gguf"}` (starts a background
download into `models_dir`) or `{"path": "/abs/path/model.gguf"}` (registers an existing file, translated
through `path_map` when the coordinator runs in Docker). Giving both forms, or an incomplete one, is a 422.
Returns the `LibraryItem`. Errors: `400` invalid name/path/not `.gguf`/split file by path, `409` name already
in the library, `404`/`403`/`502` from HF. Split GGUF (`-00001-of-NNNNN`) is added from HF via its first part
and downloaded whole; its parts are measured and served together.

**`DELETE /api/library/{name}`** cancels a running download and deletes the item and its files. Returns
`{"ok": true}`. `404` unknown, `409` if a model uses the file.

**`GET /files/{name}`** returns the file as `application/octet-stream` (also a single part of a split item).
`404 no such model file` otherwise.

```bash
curl -s -X POST $COORD/api/library -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "hf_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf"}'
```

## 6. Capacity, recommendation and simulation

All under `/api`, admin key. None of them changes the cluster.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/capacity` | per-GPU free memory, bandwidth, busyness and tenants |
| POST | `/api/recommend` | rank GPUs/servers for a model before deploying it |
| POST | `/api/simulate` | what the reconciler would do under hypothetical spec changes |

### `GET /api/capacity`

```json
{"gpus": [{"node_id": "a", "device_id": "CUDA0", "uuid": "GPU-...", "name": "NVIDIA ...", "kind": "cuda",
           "enabled": true, "alive": true, "total_mb": 16384, "usable_mb": 9800, "free_for_new_mb": 5500,
           "reserved_mb": 4300, "bandwidth_gbps": 320.0, "busy": 0.42,
           "replicas": [{"replica_id": "chat-1", "model": "chat", "est_mb": 4300, "busy": 0.42}]}],
 "summary": {"gpus": 6, "free_for_new_mb": 31200, "largest_single_gpu_mb": 9800, "largest_single_node_mb": 18100}}
```

`free_for_new_mb` is 0 for a disabled GPU or a dead node; `reserved_mb = usable_mb - free_for_new_mb`.
`summary` counts CUDA devices only.

### `POST /api/recommend`

Body `RecommendBody`:

| Field | Type, default | Meaning |
| --- | --- | --- |
| `file` | string, required | library item (must be `ready`, else 422) |
| `ctx_size` | int `>= 1`, 4096 | |
| `parallel` | int `>= 1`, 1 | |
| `priority` | int 0..100, 50 | used to decide which replicas could be preempted |
| `spread` | enum, `gpu` | |
| `pin_devices` | list, empty | |
| `limit` | int 1..10, 3 | number of options |
| `kv_cache_type` | enum, `f16` | |
| `speculative` | enum, `none` | |
| `draft_file` | string or null | for `speculative: "draft"` (same 422 checks as `PUT /api/models`) |
| `draft_n_max` | int 1..16, 4 | |
| `flash_attn`, `ubatch`, `batch` | as in `PUT /api/models`, `auto` / 512 / 2048 | |
| `kv_unified` | bool, false | |

Response:

```json
{"need_mb": 6120,
 "options": [{"rank": 1, "score": 91.0, "tier": "single_gpu", "fits_now": true,
              "assignments": [{"node_id": "b", "device_id": "CUDA1", "layers": 28, "est_mb": 6120}],
              "est_decode_tps": 41.0, "est_total_mb": 6120, "reasons": ["..."]}],
 "max_ctx_single_gpu": 16384,
 "not_possible": null,
 "tips": [{"id": "parallel", "kind": "throughput", "title": "Serve 4 requests at once (4 slots)",
           "detail": "...", "apply": {"parallel": 4, "ctx_size": 32768},
           "tier": "single_gpu", "est_decode_tps": 41.0, "est_total_mb": 7400}]}
```

- `options`: up to `limit` placements ranked by the scheduler's own score. When none fits now but stopping
  lower-priority replicas would help, one extra option is appended with `fits_now: false` and
  `requires_preemption: [{"replica_id", "model", "priority"}]`.
- `max_ctx_single_gpu`: largest `ctx_size` (multiple of 256, up to 131072) that fits on a single GPU now, or
  null.
- `not_possible`: when nothing fits even with preemption:
  `{"need_mb", "largest_single_gpu_mb", "largest_single_node_mb", "max_ctx_that_fits"}`, else null.
- `tips`: settings that would make this model faster or let it serve more users, each checked with the
  ranker against the live pool (a tip never needs more GPUs than the request as sent). `apply` holds the
  `PUT /api/models` fields to change; `kind` is `speed`, `throughput` or `fix`; `tier` / `est_*` describe the
  best placement with the tip applied. Ids: `kv_cache`, `smaller_quant` (a smaller quantization of the same
  model in the library), `ctx_single` (to fit on one GPU instead of several), `parallel`, `kv_unified` (share the context between slots), `ctx_per_slot`,
  `mtp` (the GGUF has multi-token-prediction layers), `draft` (a compatible small model in the library), `ngram`, `ubatch`, `flash_attn`, and for GPUs without
  tensor cores (compute capability below 7.0) `flash_attn_old_gpu` / `ubatch_old_gpu`. The GPU generation
  comes from each device's `compute_cap`; `ubatch` is only suggested when every GPU of the placement has
  tensor cores. Empty when nothing would help.

```bash
curl -s -X POST $COORD/api/recommend -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"file": "qwen2.5-7b-instruct-q4_k_m.gguf", "ctx_size": 8192, "parallel": 4, "priority": 80}'
```

### `POST /api/simulate`

Runs the reconciler's own ordering, placement and preemption rules over the whole cluster with hypothetical
changes. Body:

```json
{"changes": [{"model": "qwen-7b", "min_replicas": 2, "priority": 90}],
 "add": [{"name": "new", "file": "x.gguf", "replicas": 1, "ctx_size": 4096}]}
```

- `changes[]`: `model` (existing, else 404) plus any of `replicas`, `min_replicas`, `max_replicas`,
  `priority`, `preemptible`, `ctx_size`, `parallel`, `spread`, `pin_devices`, `kv_cache_type`, `speculative`,
  `draft_file`, `draft_n_max`, `flash_attn`, `ubatch`, `batch`, `kv_unified`. Absent = unchanged.
- `add[]`: `name` (new, `[A-Za-z0-9._-]{1,64}`), `file` (ready library item), plus the same optional fields.
  A new model starts at its floor (`max(min_replicas, 1)`) when `replicas > 0`.
- Validation as `PUT /api/models` (422); the result of each model is checked in full.

Response:

```json
{"start":   [{"model": "qwen-7b", "tier": "single_gpu", "est_decode_tps": 41.0,
              "assignments": [{"node_id": "b", "device_id": "CUDA1", "layers": 28, "est_mb": 6120}]}],
 "stop":    [{"replica_id": "chat-2", "model": "chat", "reason": "2 running, 1 wanted"}],
 "preempt": [{"replica_id": "q05-1", "model": "q05", "priority": 20, "for_model": "q3b"}],
 "unplaced":[{"model": "qwen-7b", "missing": 1, "why": "NoFit: ..."}]}
```

Differences from a real tick: the preemption cooldown is ignored, memory freed by stops and evictions is
available at once, and a changed `ctx_size`/`parallel` only shapes new replicas (running ones are not
restarted). It does not report moves; use `/api/rebalance` for those.

## 7. Scaling and rebalancing

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/models/{name}/scaling` | see section 3 |
| POST | `/api/rebalance` | find (and optionally start) a better replica placement |

### `POST /api/rebalance`

Optional body `{"dry_run": true}` (default **true**, also with no body). Candidates are ready replicas that
would score at least 25 points higher somewhere else, best first. With `dry_run: false` the best one is
started **make-before-break**: a new replica is launched, and the old one drains only when the new one is
ready. Refused (nothing started) unless the cluster is quiet: no move in flight, no replica launching, no
preemption waiting for its memory. One move at a time cluster-wide.

```json
{"moves": [{"replica_id": "y-1", "model": "y", "from": [{"node_id": "b", "device_id": "CPU"}],
            "to": [{"node_id": "a", "device_id": "CUDA0"}], "current_score": -20.0, "new_score": 80.0,
            "gain": 100.0, "reasons": ["..."]}],
 "started": null,
 "in_progress": null}
```

`started` is `{"replica_id", "model"}` when a move began. `in_progress` is the move in flight
(`{"model", "old", "new", "since"}`) or null. A periodic run also happens every `rebalance_s` (default 600 s,
0 disables) when the cluster is quiet.

```bash
curl -s -X POST $COORD/api/rebalance -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" -d '{"dry_run": false}'
```

## 8. State and events

### `GET /api/state`

Everything the web UI shows, in one call (admin key).

| Key | Content |
| --- | --- |
| `summary` | `servers_total`, `servers_online`, `gpus_total`, `gpus_enabled`, `pool_total_mb`, `pool_usable_mb` (enabled CUDA GPUs of live servers), `models_running` |
| `servers[]` | `node_id`, `agent_url`, `added_at`, `alive`, `last_seen`, `report` (the agent's `NodeReport` or null), `gpu_enabled` (`{device_id: bool}`) |
| `models[]` | see below |
| `library[]` | `LibraryItem` list |
| `settings` | `public_url`, `cluster_token`, `api_keys_set` (bool) |
| `events[]` | the 50 newest events |
| `unread_events` | count of unread events |
| `rebalance` | `{"in_progress": {...} or null, "next_run_ts": unix time or null}` |
| `speed_model` | `{"eta": float, "hop_ms": float}`: the decode-speed model the scheduler ranks with (fraction of peak memory bandwidth, milliseconds per RPC server per token), learned from measured speed |

`models[]` entry:

| Field | Meaning |
| --- | --- |
| `spec` | the `ModelSpec` |
| `file` | library file name when the source is `coordinator://`, else null |
| `state` | `running`, `starting`, `idle` (started but unloaded, next request loads it), `stopping`, `stopped`, `failed` |
| `error` | reason for `failed`, or the newest replica error while `starting` |
| `scaling` | `{"min", "max", "desired", "avg_busy", "unloaded"}` |
| `calibration` | the **VRAM self-calibration** block, `{"factor": 1.04, "samples": 3}`, or `null` until a replica of the model was measured. See below |
| `replicas[]` | `ReplicaRecord` fields (`replica_id`, `model`, `placement`, `state`, `created_at`, `updated_at`, `error`) plus `outstanding` (requests in flight). Lists live replicas and the newest failed one |

**`calibration`.** After a replica becomes ready, the coordinator asks the head's agent for what
llama.cpp really allocated (`GET /engines/{engine_id}/memory`, section 10), compares it with the estimate
(minus the per-device runtime context llama.cpp does not report) and folds the ratio into a per-model factor
(exponential moving average, weight 0.5 per sample). Placement multiplies that model's device needs by
`factor`, clamped to 0.9..2.0, so later plans use real numbers. `factor` here is the clamped value planning
uses, `samples` the number of measurements. An info event `calibrated` is emitted when the factor moves by
more than 5 %. A sample is skipped when the agent is old, the log has no buffer lines, or a device is
missing, and then the factor stays as it was.

### Events

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/events?limit=200&after_id=` | list events, newest first |
| POST | `/api/events/read` | mark events read |

**`GET /api/events`**: `limit` is clamped to 1..1000 (default 200); `after_id` returns only events with a
larger id. Returns `{"events": [...], "unread": n}`. An event is
`{"id", "ts", "level": "info"|"warning"|"error", "kind", "message", "node_id", "model", "read"}`.

**`POST /api/events/read`** body `{"up_to_id": 123}` marks events up to that id read. Returns `{"unread": n}`.

Event kinds in the code: `server_added`, `server_removed`, `node_online`, `node_offline`, `model_started`,
`model_stopped`, `launch_failed`, `engine_crashed`, `crash_loop`, `gpu_missing`, `realloc_started`,
`realloc_failed`, `realloc_done`, `preempted`, `scaled_up`, `scaled_down`, `unloaded_idle`, `cold_start`,
`rebalance_started`, `rebalanced`, `rebalance_failed`, `calibrated`, `mtp_unavailable` (an `mtp` model's head cannot draft with MTP:
served without speculation), `llama_version_mismatch` (registered servers
run different llama.cpp builds; a split model needs the same RPC protocol everywhere). Warnings and errors are also POSTed to
`webhook_url` when configured.

## 9. Coordinator: health, metrics and legacy admin API

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/healthz` | none | `{"ok": true}` |
| GET | `/metrics` | none | Prometheus text |
| POST | `/admin/models` | admin | create or replace a model from a full `ModelSpec` |
| DELETE | `/admin/models/{name}` | admin | drain and delete a model |
| POST | `/admin/models/{name}/scale?replicas=N` | admin | set `replicas` (`N >= 0`, else 422) |
| POST | `/admin/deploy/{model}?dry_run=0` | admin | `dry_run=1`: the `Placement` (as `/plan`); else run one reconcile tick and return the model's replicas |
| DELETE | `/admin/replicas/{replica_id}` | admin | drain one replica (404 unknown) |
| GET | `/admin/status` | admin | raw nodes (devices, engines), models and replicas |

`/admin/*` is the older, lower-level interface, used by the CLI and scripts. It takes the raw `ModelSpec` (no
library check, no draft or pin validation, no events), so prefer `/api/models`.
`POST /admin/models` returns the stored spec. `GET /admin/status` returns `{"nodes": [{"node_id", "alive",
"last_seen", "agent_url", "host", "devices", "engines"}], "models": [...], "replicas": [... plus
"outstanding"]}`.

`/metrics` exposes the router's counters and histograms (requests per model and status, retries, time to
first token, outstanding requests) and: `gpupool_device_free_mb{node,device}`,
`gpupool_device_usable_mb{node,device}`, `gpupool_node_alive{node}`, `gpupool_replicas{model,state}`.

## 10. Internal API (do not call by hand)

### Coordinator, cluster token

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/internal/heartbeat` | agent pushes its `NodeReport` (only when `push_heartbeat` is on; default is that the coordinator polls `GET /report`) |
| POST | `/internal/join` | agent self-registration (`gpupool agent --join ...`) |

`POST /internal/heartbeat` body: a `NodeReport`. `403` when the node is not registered (a removed server
cannot come back by itself). Returns `{"ok": true}`.

`POST /internal/join` body `{"agent_url": "http://10.0.0.5:7070"}`. The coordinator probes the agent like
`POST /api/servers`; failures are `502` (not 400). `403` if the server was removed in the UI. Returns
`{"node_id", "agent_url", "added_at", "new": true|false}`; if the node is known under another URL, the URL is
updated (`new: false`).

### Agent API

The agent (default port 7070) is driven by the coordinator. Auth: cluster token, except `/health`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | `{"ok": true}`, no auth |
| GET | `/report` | `NodeReport`: devices, engines, llama.cpp version and CUDA archs, cached model files, `features` (by the llama.cpp build: `rpc_multi_device` one rpc engine may serve several devices, `spec_mtp`, `kv_unified`), CPU/RAM |
| POST | `/engines` | start one llama.cpp process from an `EngineSpec` |
| GET | `/engines/{engine_id}` | `EngineStatus` |
| GET | `/engines/{engine_id}/memory` | per-device buffers llama.cpp reported at load |
| DELETE | `/engines/{engine_id}` | stop an engine, returns its `EngineStatus` |
| POST | `/models/ensure` | make sure a model file is in the local cache |

**`POST /engines`** body `EngineSpec`: `engine_id` (`"<replica_id>-head"` or `"<replica_id>-rpc-<first device_id>"`),
`kind` (`"rpc"`/`"server"`), `port`, `devices` (rpc: one or more distinct local devices, served by one process; server: ordered list such as
`["CUDA0","RPC0"]`), `model` (alias), `model_path` (GGUF on the head), `rpc_endpoints` (`"host:port"`, order of
`RPC0..`; each RPC server once), `tensor_split`, `ctx_size` (4096), `parallel` (1), `extra_args`, `cache_type` (`f16`),
`spec_type` (`none`; `ngram` → `--spec-type ngram-mod`, `draft` → `draft-simple`, `mtp` → `draft-mtp`),
`draft_model_path`, `draft_device`, `draft_n_max` (4), `flash_attn` (`auto`, `-fa`),
`batch` (2048, `-b`), `ubatch` (512, `-ub`), `kv_unified` (false, `-kvu`), `allowed_peers` (hosts allowed
to reach an rpc engine; enforced only when the agent runs with `rpc_firewall`). Returns `EngineStatus`
(`engine_id`, `kind`, `state` `starting|running|exited|failed`, `pid`, `port`, `exit_code`, `log_tail` of at
most 50 lines). Errors: `422` for `extra_args` (not accepted by this agent, so the token cannot become
arbitrary llama-server flags), a server engine without `model_path`, a `model_path` or `draft_model_path`
outside the model cache and not returned by `/models/ensure`, a missing file, an rpc engine without devices
or with a device listed twice, or a port in use; `409` engine already running; `500` binary not found.

**`GET /engines/{engine_id}/memory`**: llama.cpp prints its buffer sizes when it loads a model; the agent
parses the head of the engine's log (first 2 MiB) and returns, per device (`CUDA0`, `RPC0`, ...) in MiB, the
last value seen of each kind:

```json
{"engine_id": "chat-1-head",
 "devices": {"CUDA0": {"model_mb": 2090.5, "kv_mb": 144.0, "compute_mb": 80.5, "total_mb": 2315.0},
             "RPC0":  {"model_mb": 1003.2, "kv_mb": 72.0, "compute_mb": 80.5, "total_mb": 1155.7}}}
```

Host buffers (`CPU`, `*_Host`, `*_Mapped`) are left out. `devices` is `{}` when the log has no buffer lines.
`404 unknown engine`. This is the data behind the `calibration` block of `/api/state`.

**`DELETE /engines/{engine_id}`**: `404 unknown engine`.

**`POST /models/ensure`** body `{"name": "file.gguf", "source": "coordinator://file.gguf" | "https://...gguf"}`.
Downloads (from the coordinator's `/files/{name}` with the cluster token, or from the URL) into the agent's
cache if missing; split GGUF parts are fetched together. Returns `{"path": "...", "bytes": n}`. Errors:
`422` file not found, `502` download failed.

## 11. Route index

All 51 routes, for a completeness check:

| # | Method | Path | Section |
| --- | --- | --- | --- |
| 1 | GET | `/v1/models` | 1 |
| 2 | POST | `/v1/chat/completions` | 1 |
| 3 | POST | `/v1/completions` | 1 |
| 4 | GET | `/api/state` | 8 |
| 5 | POST | `/api/servers` | 4 |
| 6 | DELETE | `/api/servers/{node_id}` | 4 |
| 7 | PUT | `/api/servers/{node_id}/gpus/{device_id}` | 4 |
| 8 | PUT | `/api/models/{name}` | 3 |
| 9 | POST | `/api/models/{name}/start` | 3 |
| 10 | GET | `/api/models/{name}/scaling` | 3 |
| 11 | POST | `/api/models/{name}/stop` | 3 |
| 12 | DELETE | `/api/models/{name}` | 3 |
| 13 | POST | `/api/models/{name}/plan` | 3 |
| 14 | GET | `/api/capacity` | 6 |
| 15 | POST | `/api/simulate` | 6 |
| 16 | POST | `/api/rebalance` | 7 |
| 17 | POST | `/api/recommend` | 6 |
| 18 | GET | `/api/events` | 8 |
| 19 | POST | `/api/events/read` | 8 |
| 20 | GET | `/api/hf/files` | 5 |
| 21 | GET | `/api/library` | 5 |
| 22 | GET | `/api/library/browse` | 5 |
| 23 | POST | `/api/library` | 5 |
| 24 | DELETE | `/api/library/{name}` | 5 |
| 25 | GET | `/files/{name}` | 5 |
| 26 | GET | `/healthz` | 9 |
| 27 | POST | `/internal/heartbeat` | 10 |
| 28 | POST | `/internal/join` | 10 |
| 29 | POST | `/admin/models` | 9 |
| 30 | DELETE | `/admin/models/{name}` | 9 |
| 31 | POST | `/admin/models/{name}/scale` | 9 |
| 32 | POST | `/admin/deploy/{model}` | 9 |
| 33 | DELETE | `/admin/replicas/{replica_id}` | 9 |
| 34 | GET | `/admin/status` | 9 |
| 35 | GET | `/metrics` | 9 |
| 36 | GET | `/health` (agent) | 10 |
| 37 | GET | `/report` (agent) | 10 |
| 38 | POST | `/engines` (agent) | 10 |
| 39 | GET | `/engines/{engine_id}` (agent) | 10 |
| 40 | GET | `/engines/{engine_id}/memory` (agent) | 10 |
| 41 | DELETE | `/engines/{engine_id}` (agent) | 10 |
| 42 | POST | `/models/ensure` (agent) | 10 |
| 43 | GET | `/api/convert/options` | 12 |
| 44 | POST | `/api/convert/inspect` | 12 |
| 45 | POST | `/api/convert` | 12 |
| 46 | GET | `/api/convert` | 12 |
| 47 | GET | `/api/convert/{job_id}` | 12 |
| 48 | POST | `/api/convert/{job_id}/cancel` | 12 |
| 49 | POST | `/api/convert/{job_id}/retry` | 12 |
| 50 | POST | `/api/convert/{job_id}/accept` | 12 |
| 51 | DELETE | `/api/convert/{job_id}` | 12 |

The count matches the route decorators found in the code: 16 + 6 + 10 + 9 + 3 + 7 = 51.

## 12. Conversion (`/api/convert`)

Converts a Hugging Face model (or a folder of weights on the coordinator machine) to GGUF and adds it to the model
library. All 9 routes need the admin key and live in `coordinator/convert_api.py`; the shapes come from
`converter/models.py`. How it works: [DESIGN.en.md](DESIGN.en.md#17-hugging-face-to-gguf-conversion-converter); how
to use it: [QUICKSTART.en.md](QUICKSTART.en.md#serving-a-model-that-has-no-gguf-convert).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/convert/options` | is conversion available, every quantization type, what the cluster can hold |
| POST | `/api/convert/inspect` | look at a source without downloading weights: facts, warnings, per-type estimates |
| POST | `/api/convert` | start a conversion job |
| GET | `/api/convert` | list jobs, newest first |
| GET | `/api/convert/{job_id}` | one job |
| POST | `/api/convert/{job_id}/cancel` | cancel a queued or running job |
| POST | `/api/convert/{job_id}/retry` | queue a failed or cancelled job again |
| POST | `/api/convert/{job_id}/accept` | add the file of a `needs_review` job to the library |
| DELETE | `/api/convert/{job_id}` | delete a finished job and its leftovers |

Status codes used by these routes (the body is always `{"detail": "<message>"}`):

| Code | Meaning here |
| --- | --- |
| 400 | invalid repo id (not `owner/name`), invalid output name, unknown ggml type in `advanced`, invalid folder path |
| 403 | the Hugging Face repo is gated or private: set `HF_TOKEN` and accept the licence |
| 404 | repo or revision not found on Hugging Face, or unknown job id |
| 409 | the output name is taken (library, a file in `models_dir`, or another unfinished job); the job is not in a state that allows the action |
| 422 | the source cannot be converted: architecture not supported by the pinned converter (llama.cpp b11413), pre-quantized in a format the converter cannot read (AWQ, bitsandbytes...), no `config.json`, no safetensors / PyTorch weights; also a malformed body (for example both `hf_repo` and `path`); a type that needs an importance matrix with `advanced.imatrix: "off"`; an unusable `advanced.calibration_path` (not a `.txt`, empty, larger than 20 MB) |
| 502 | Hugging Face unreachable or answered with an unexpected error |
| 503 | the conversion toolchain is not set up (`problem` of `/api/convert/options`); `llama-imatrix` is missing while the job needs an importance matrix (a type that needs one, or `advanced.imatrix: "on"`); the built-in calibration text is missing from the install and no `calibration_path` was given |
| 507 | not enough free disk space. `POST /api/convert` answers 507 at submit when the disk obviously cannot hold *uncached download + 16-bit intermediate + output* (nothing is queued). The worker repeats the check before each heavy stage, because space can shrink while a job waits; that 507 appears as the `error` of a *failed* job (with `failed_stage` set) |

### Types

**`QuantOption`** (an element of `quant_options` and of `InspectResult.options`):

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | string | one of `F16`, `BF16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q5_K_S`, `Q4_K_M`, `Q4_K_S`, `IQ4_XS`, `Q4_0`, `Q3_K_L`, `Q3_K_M`, `IQ3_M`, `IQ3_S`, `Q3_K_S`, `IQ3_XS`, `IQ3_XXS`, `Q2_K`, `IQ2_M`, `IQ2_S`, `IQ2_XS`, `IQ2_XXS`, `IQ1_M`, `IQ1_S` |
| `bpw` | float | bits per weight used for the estimate |
| `tier` | string | `lossless`, `near_lossless`, `balanced`, `small` or `tiny` |
| `note` | string | quality note, e.g. the perplexity change against F16 |
| `via` | `"convert"` / `"quantize"` | `F16`, `BF16` and `Q8_0` are written by the converter directly, the rest by llama-quantize |
| `needs_imatrix` | bool | llama-quantize refuses this type without an importance matrix: `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ2_M`, `IQ3_XXS`, `IQ3_XS` |
| `est_bytes` | int or null | estimated file size; only filled by inspect |
| `est_vram_mb` | int or null | file + KV cache at context 4096 + runtime overhead |
| `fits_single_gpu`, `fits_pool` | bool or null | against the largest single GPU / the whole pool; null when there is no GPU server or the parameter count is unknown |
| `recommended` | bool | the type inspect recommends |

**`SourceSpec`**: exactly one of `hf_repo` (`"owner/name"`) or `path` (absolute folder on the coordinator machine,
translated through `path_map` like library paths); `revision` (default `"main"`) applies to `hf_repo`.

**`InspectResult`**:

| Field | Type | Meaning |
| --- | --- | --- |
| `source` | `SourceSpec` | the input |
| `architecture`, `model_type` | string or null | `architectures[0]` and `model_type` of `config.json` |
| `supported` | bool or null | by the pinned converter; null = toolchain missing, so unknown |
| `params`, `n_layers`, `context_length` | int or null | |
| `weight_format` | `"safetensors"` / `"pytorch_bin"` / `"none"` | `none` means nothing to convert |
| `prequantized` | string or null | `quantization_config.quant_method` (`awq`, `gptq`, `fp8`...) |
| `prequant_supported` | bool or null | can the converter read that format |
| `source_bytes` | int | bytes of the selected files (the download size for Hugging Face) |
| `files`, `skipped` | `[{name, bytes}]`, `[string]` | files the conversion will use / files ignored |
| `remote_code` | bool | the source ships `*.py` or `auto_map`; not downloaded or run unless allowed |
| `gated` | bool | |
| `base_model` | string or null | the card's base model: convert that instead of a pre-quantized repo |
| `gguf_alternatives` | `[string]` | Hugging Face repos with ready GGUF builds |
| `options` | `[QuantOption]` | every type with estimates |
| `recommended`, `recommend_reasons` | `QuantType`, `[string]` | the recommended type and why |
| `name_stem` | string | the sanitized last part of the repo or folder name: the default output name is `<name_stem>-<QUANT>.gguf` (so clients need not repeat the naming rule) |
| `warnings` | `[string]` | |

**`ConvertRequest`** (body of `POST /api/convert`):

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `source` | `SourceSpec` | required | |
| `quant` | `QuantType` | `"Q4_K_M"` | |
| `name` | string or null | `<model>-<QUANT>.gguf` | output file name: letters, digits, `.`, `_`, `-`, ending in `.gguf`, not a split-part name |
| `keep_source` | bool | `false` | keep the downloaded Hugging Face files after success (folders are never touched) |
| `advanced.intermediate` | `auto` / `f16` / `bf16` / `f32` | `auto` | 16/32-bit file written before quantizing |
| `advanced.output_tensor_type`, `advanced.token_embedding_type` | ggml type name or null | null | `llama-quantize --output-tensor-type` / `--token-embedding-type` (e.g. `q8_0`) |
| `advanced.leave_output_tensor` | bool | `false` | `llama-quantize --leave-output-tensor` |
| `advanced.pure` | bool | `false` | `llama-quantize --pure` |
| `advanced.allow_remote_code` | bool | `false` | download and run the repo's own `*.py` inside the coordinator (dangerous) |
| `advanced.imatrix` | `auto` / `on` / `off` | `auto` | importance matrix mode. `auto` computes one when the type needs it (`needs_imatrix`) or has `bpw` below 4.0; `on` always (for types written by llama-quantize); `off` never. Details below |
| `advanced.calibration_path` | string or null | null | absolute path of a `.txt` file on the coordinator machine (at most 20 MB, not empty; host paths are translated like library paths) to calibrate on; null = the multilingual text shipped with gpupool. The file is copied when the job starts |
| `advanced.imatrix_chunks` | int >= 0 | `0` | number of 512-token chunks `llama-imatrix` processes; `0` = 100. Fewer is faster and rougher |
| `advanced.validate_generation` | bool | `true` | generate a few tokens on the CPU after converting |
| `advanced.threads` | int >= 0 | `0` | `0` = `GPUPOOL_CONVERT_THREADS` |

The `advanced` quantize options (including the importance matrix ones) are ignored for `F16`, `BF16` and `Q8_0`.

**Importance matrix policy (decided at submit).** The job computes a matrix when `quant` is not `F16`/`BF16`/`Q8_0` and
`imatrix` is `on`, or `auto` with `needs_imatrix` true or `bpw` < 4.0 (so `Q4_K_M` and up go without under `auto`).
`off` on a type with `needs_imatrix` is refused (422). If `llama-imatrix` is not installed: `on`, or a type with
`needs_imatrix`, is refused (503); `auto` on a type that only benefits from a matrix quietly runs without one
(`imatrix_used: false`). The calibration text is checked at submit (`.txt`, not empty, at most 20 MB, else 422).
The matrix is computed by `llama-imatrix -c 512 --no-ppl -ngl 0 --chunks <N>` on the 16-bit intermediate, on the CPU,
in the `calibrating` stage, which takes most of the time (measured: about 20 of 28 minutes for Qwen2.5-1.5B to
`IQ3_M` on a laptop CPU).

**`ConvertJob`**:

| Field | Type | Meaning |
| --- | --- | --- |
| `id` | string | 12 hex characters |
| `request` | `ConvertRequest` | as submitted |
| `state` | string | `queued`, `downloading`, `converting`, `calibrating`, `quantizing`, `validating`, `needs_review`, `done`, `failed`, `cancelled` |
| `stage_progress` | float or null | 0..1 inside the current stage, null when unknown |
| `bytes_done`, `bytes_total` | int, int or null | download stage |
| `output_name` | string | file name in the library |
| `imatrix_used` | bool | an importance matrix is (or was) computed for this job: the job goes through `calibrating`. Decided at submit by the policy above |
| `failed_stage` | state or null | the stage that was running when the job failed or was cancelled (for example `calibrating`); null otherwise and after a retry. Databases from the first release are migrated in place |
| `output_bytes`, `est_output_bytes` | int or null | real and estimated size |
| `validation` | `Validation` or null | see below |
| `error` | string or null | why it failed, or the *needs review* note |
| `log_tail` | `[string]` | last lines of the current or last tool |
| `created_at`, `started_at`, `finished_at` | float or null | unix time |

Active states are `queued`, `downloading`, `converting`, `calibrating`, `quantizing`, `validating`; `done`, `failed`
and `cancelled` are final; `needs_review` waits for `accept` or delete. A job from a folder passes through
`downloading` too (it only links the files). `calibrating` only occurs when `imatrix_used` is true. `F16`, `BF16` and
`Q8_0` skip `calibrating` and `quantizing`.

**`Validation`**: `header_ok` (bool or null), `architecture`, `n_layers`, `vocab_size`, `chat_template` (bool),
`tokenizer_ok` (bool, null = could not run), `tokenizer_cases` (`[{text, hf, gguf, match}]`: the token ids of 8 probe
texts from Hugging Face and from the GGUF), `generation_ok` (bool, null = skipped), `generation_sample` (string),
`warnings`, `errors`. `header_ok: false` makes the job *failed*; `tokenizer_ok: false` or `generation_ok: false`
make it `needs_review`; null values are not failures.

### `GET /api/convert/options`

Returns `{"available": bool, "problem": string or null, "quant_options": [QuantOption], "imatrix_available": bool,
"cluster": {"largest_gpu_mb", "pool_mb"}}`. `problem` says what is missing when the toolchain is not set up (the UI
shows it). `imatrix_available` is whether `llama-imatrix` is installed: when false, types with `needs_imatrix` cannot
be converted (the UI greys them out) and the importance matrix is unavailable. The estimates in
`quant_options` are empty here; use inspect for a concrete model.

### `POST /api/convert/inspect`

Body: a `SourceSpec`. Reads the configuration and the file listing (for a folder, the safetensors headers) without
downloading weights; returns an `InspectResult` with per-type sizes, VRAM, fit and the recommendation. It works
without the toolchain, then `supported` is null. Errors: `400`, `403`, `404`, `422` (no `config.json`), `502`.

### `POST /api/convert`

Body: a `ConvertRequest`. Checks the toolchain, inspects the source, refuses what cannot work, and queues the job.
Returns the `ConvertJob` (state `queued`). Errors: `400`, `403`, `404`, `409`, `422`, `502`, `503`, `507` as in the
table above (`507` is the early disk check). Jobs run one at a time, in submission order.

### `GET /api/convert`, `GET /api/convert/{job_id}`

A list of `ConvertJob` (newest first), or one job (`404` unknown id). Poll one job to follow progress.

### `POST /api/convert/{job_id}/cancel`

Cancels a queued or running job: the running tool is killed and the job's scratch folder removed. Returns the job
(state `cancelled`). `409` if the job is not active.

### `POST /api/convert/{job_id}/retry`

Queues a `failed` or `cancelled` job again with the same request; a finished download in the cache is reused. Returns
the job (state `queued`). `409` if the job is in another state or its output name is taken meanwhile.

### `POST /api/convert/{job_id}/accept`

For a `needs_review` job: moves the converted file into the library despite the failed check. Returns the job
(state `done`). `409` if the job is not in `needs_review`, the file is gone, or the name is taken.

### `DELETE /api/convert/{job_id}`

Deletes the job and its scratch folder (a *needs_review* file is discarded). Returns `{"ok": true}`. `409` while the
job is active (cancel it first), `404` unknown.

```bash
# 1. look first
curl -s -X POST $COORD/api/convert/inspect -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct"}' | jq '{architecture, supported, params, recommended, recommend_reasons}'

# 2. convert (a folder works too: "source": {"path": "/models/hf/my-model"})
curl -s -X POST $COORD/api/convert -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"source": {"hf_repo": "Qwen/Qwen2.5-0.5B-Instruct"}, "quant": "Q4_K_M"}' | jq .id

# 3. follow it, then deploy the file from the library
curl -s -H "Authorization: Bearer $ADMIN" $COORD/api/convert/<job id> | jq '{state, stage_progress, error}'

# a type that needs an importance matrix: calibrated automatically (state goes through "calibrating");
# fewer chunks = faster. "imatrix": "off" would be refused for IQ2_XS with 422
curl -s -X POST $COORD/api/convert -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -d '{"source": {"hf_repo": "Qwen/Qwen2.5-1.5B-Instruct"}, "quant": "IQ3_M", "advanced": {"imatrix": "auto", "imatrix_chunks": 30}}' | jq '{id, imatrix_used}'

# needs_review: accept it (or DELETE the job)
curl -s -X POST -H "Authorization: Bearer $ADMIN" $COORD/api/convert/<job id>/accept | jq .state
```

A finished job adds a `LibraryItem` with `source: "convert"` (`hf_repo` set for Hugging Face sources), listed by
`GET /api/library` like any other file and removable with `DELETE /api/library/{name}`.
