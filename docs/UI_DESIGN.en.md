# gpupool — Management UI + Docker: design

> English version. Vietnamese version: [UI_DESIGN.vi.md](UI_DESIGN.vi.md). Keep both in sync.

Goal: one web console, served by the coordinator on one machine, to add/remove servers, enable or
disable each GPU, bring in models (Hugging Face or a file path), and start/stop serving a model with
one button. Everything ships as Docker images.

## 1. Decisions

| Topic | Decision | Why |
| --- | --- | --- |
| Adding a server | Register an agent that already runs: the user enters its URL; the coordinator calls it, reads its `node_id`, and adds it | Answer to the open question. No SSH credentials stored anywhere. |
| Node discovery | **Coordinator pulls** `GET /report` from every registered server every 2 s | Removing a server must stick. With push heartbeats a deleted node would re-appear on its next heartbeat. The push endpoint stays, but accepts registered nodes only. |
| GPU selection | Both: (1) per-GPU on/off switch in the pool; (2) optional GPU pick when starting a model, else auto | Answer to the open question. |
| Frontend | Static files (HTML + CSS + [Alpine.js](https://alpinejs.dev), vendored, ~45 KB), served by the coordinator at `/` | No Node.js or build step, works offline, one origin so no CORS. |
| UI login | Admin key, kept in the browser's localStorage, sent as a bearer token | Reuses the existing `admin_key`. |
| Live data | The UI polls `GET /api/state` every 2 s | Simple; the cluster is small. |
| Model downloads | The coordinator downloads from Hugging Face into `models_dir`; heads fetch from the coordinator (`coordinator://`) | Only the coordinator needs internet; servers stay as they are. |
| Start / Stop | Start = desired replicas ← N (default 1); Stop = desired replicas ← 0 (drain, then the engines stop) | Built on the existing reconciler. |
| Docker | `gpupool-coordinator` (Python slim, no GPU) and `gpupool-agent` (CUDA runtime + llama.cpp built with CUDA and RPC) | Agents run with `--gpus all --network host`; host networking keeps RPC ports and IPs simple. |

## 2. Screens (as in the mockup)

- **Overview**: cards for servers (online/total), GPUs (enabled/total), free VRAM in the pool, models running.
- **Servers**: one card per server: status, IP, CPU/RAM bars, GPU table (utilization, memory used/total,
  temperature, power, process count, status), a checkbox per GPU = in pool on/off. **Add Server** opens
  a dialog with the URL field and a ready-to-copy `docker run` command for the agent. Delete asks for
  confirmation, stops this server's engines, and removes it.
- **GPU detail panel** (click a GPU): utilization ring, memory, temperature, power, running processes
  (PID, name, GPU memory), static details (model, CUDA, driver).
- **Models**: model library (files) + deployments.
  - *Add model*: tab **Hugging Face** (repo id → list of `.gguf` files from the HF API → pick one →
    download with a progress bar) or tab **Path** (absolute path to a `.gguf` on the coordinator machine;
    in Docker, a mounted folder).
  - Each model row: name, size, status (stopped / starting / running / failed + error), endpoint,
    replicas, ctx, **GPU picker** (Auto or tick GPUs across servers), **Start** / **Stop** buttons.
- **Settings**: shows the API base URL, the OpenAI client snippet, the agent install command.

## 3. Contract changes (`common/models.py`, all new fields optional, so old agents keep working)

```python
class GpuProcess(BaseModel):
    pid: int
    name: str            # "" when the OS hides it (other users' processes)
    used_mb: int | None

class Device(BaseModel):
    ...                  # existing fields
    temp_c: int | None = None
    power_w: int | None = None
    processes: list[GpuProcess] = []
    driver: str | None = None
    cuda: str | None = None      # max CUDA version the driver supports

class NodeReport(BaseModel):
    ...                  # existing fields
    cpu_pct: float | None = None
    ram_used_mb: int | None = None
    ram_total_mb: int | None = None

class ModelSpec(BaseModel):
    ...                  # existing fields
    pin_devices: list[str] = []  # "node_id/device_id"; empty = scheduler chooses
```

New store tables: `servers(node_id, agent_url, added_at)`, `gpu_flags(node_id, device_id, enabled)` (the second column holds the GPU's `uuid` when the agent reports one, so a flag follows the physical card; legacy rows keyed by `CUDA<i>` are migrated on the first report carrying uuids),
`library(name, path, source, bytes, status, progress, error)`.

Scheduler input changes (no change to `plan()` itself): before planning, the reconciler sets
`usable_mb = 0` on disabled GPUs and, when `pin_devices` is set, on every device not in it. Disabling a
GPU affects new placements only; replicas already running there keep running until stopped.

## 4. HTTP API used by the UI (admin key)

| Method | Path | Body → Result |
| --- | --- | --- |
| GET | `/api/state` | → `{servers:[{node_id, agent_url, alive, last_seen, report, gpu_enabled:{device_id:bool}}], models:[{spec, file, replicas:[ReplicaRecord + outstanding]}], library:[LibraryItem], summary:{...}}` |
| POST | `/api/servers` | `{agent_url}` → server (400 if unreachable / wrong token, 409 if node_id exists) |
| DELETE | `/api/servers/{node_id}` | stops that server's replicas, removes it |
| PUT | `/api/servers/{node_id}/gpus/{device_id}` | `{enabled: bool}` |
| GET | `/api/hf/files?repo=owner/name` | → `[{file, bytes}]` (`.gguf` only; HF token from env if set) |
| POST | `/api/library` | `{hf_repo, hf_file}` or `{path}` → LibraryItem (HF: download starts in background) |
| DELETE | `/api/library/{name}` | removes the entry (deletes the file only if it was downloaded by us; 409 if a model uses it) |
| PUT | `/api/models/{name}` | `{file, ctx_size, parallel, replicas, pin_devices}` → spec (create or update) |
| POST | `/api/models/{name}/start` | `{replicas?}` → spec |
| POST | `/api/models/{name}/stop` | → spec (replicas 0) |
| DELETE | `/api/models/{name}` | stop + remove |
| POST | `/api/models/{name}/plan` | → Placement (dry run; shows which GPUs would be used) |

`/files/{name}` (cluster token) serves library entries by name: downloaded files and registered paths,
nothing else (no path traversal possible: names are looked up, not joined).

## 5. Docker

- `docker/coordinator.Dockerfile`: `python:3.12-slim` + uv, app installed, `EXPOSE 8080`, volume `/data`
  (database + `models`). Config from environment: `GPUPOOL_ADMIN_KEY`, `GPUPOOL_CLUSTER_TOKEN`,
  `GPUPOOL_API_KEYS`, `HF_TOKEN`.
- `docker/agent.Dockerfile`: build stage `nvidia/cuda:12.4.1-devel-ubuntu22.04` compiles llama.cpp
  (pinned tag, `GGML_CUDA=ON GGML_RPC=ON`, build args `LLAMA_CPP_REF`, `CUDA_ARCHS`); runtime stage
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04` + uv + the app. Config from environment:
  `GPUPOOL_NODE_ID`, `GPUPOOL_HOST` (reachable IP), `GPUPOOL_CLUSTER_TOKEN`, `GPUPOOL_PORT`.
  CUDA 12.4 runs on drivers ≥ 525 (CUDA minor-version compatibility).
- `docker-compose.coordinator.yml` and `docker-compose.agent.yml`.
- Note: Docker needs the `docker` group or root on each server. The uv install path stays supported for
  servers without Docker access.

## 6. Implementation split (parallel, disjoint files)

| Owner | Files |
| --- | --- |
| Lead | `common/models.py` (contract above), `cli.py`, integration, Docker files, docs |
| Agent A: agent metrics | `agent/gpu.py`, `agent/app.py` (report fields), tests |
| Agent B: coordinator servers + models | `coordinator/store.py`, `coordinator/poller.py` (new), `coordinator/reconciler.py`, `coordinator/api.py` (new, `/api/servers*`, `/api/models*`, `/api/state`), tests |
| Agent C: library + Hugging Face | `coordinator/library.py` (new: store table, HF listing, background download with progress, path registration, `/files` lookup), `coordinator/library_api.py` (new router), tests |
| Agent D: frontend | `src/gpupool/ui/` (index.html, app.js, styles.css, vendor/alpine.min.js) |

`coordinator/app.py` wiring is the lead's job after the agents finish.

## 7. Test plan

- Unit tests per module (mocked HF API, fake agents).
- Real run on this laptop: coordinator + 3 agents; in the browser: add the servers by URL, disable a
  GPU, download a small GGUF from Hugging Face, Start → chat answers → Stop → engines gone, delete a
  server. UI screenshots for the report.
- Docker: images built in GitHub Actions (this laptop has no Docker). Running the agent image with a GPU
  is **not** testable here and will be reported as untested.

## 8. Failure detection and notifications (added 2026-10-02)

A server can lose power, or a single GPU can fall off the bus, while models run on it.

| Situation | Detected by | Action |
| --- | --- | --- |
| Server down (power, network, agent crash) | no successful `/report` for `heartbeat_timeout_s` (10 s; polled every 2 s) | replicas touching it → `failed`; surviving engines on other servers stopped; re-placed on the remaining GPUs |
| One GPU gone, server alive | the GPU is missing from the server's report | replicas using it → `failed`, re-placed |
| llama.cpp process crashed | engine `exited`/`failed` in the report | replica → `failed`, re-placed |
| Not enough capacity left | `plan()` raises `NoFit` | model shown as failed with "needs X GB, pool has Y GB"; retried automatically when capacity returns |

Every transition is recorded as an **event** (`info` / `warning` / `error`): `node_offline`, `node_online`,
`gpu_missing`, `engine_crashed`, `realloc_started`, `realloc_done`, `realloc_failed`, `launch_failed`,
plus server/model actions. Events are emitted once per transition, not every reconcile tick.

Users are told through: a bell with an unread count and toasts in the UI, an Events page, optional
desktop notifications (browser Notification API), and an optional webhook (`webhook_url`, Slack or
Discord compatible) for warnings and errors, so alerts arrive even with the UI closed.

Requests in flight on a replica that dies are lost unless they had not received a byte yet (the router
retries those on another replica). New requests go to the remaining replicas immediately; if a model had
a single replica, it is unavailable until the re-allocation is ready (seconds for small models, longer
when weights must travel over RPC).
