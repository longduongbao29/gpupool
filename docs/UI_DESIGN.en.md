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
    replicas, ctx, **allowed servers/GPUs** (all, or only the ticked ones; see section 10), **Start** / **Stop** buttons.
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
    pin_devices: list[str] = []  # allowed set: "node_id/device_id" or "node_id/*"; empty = all
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

## 9. Model conversion UI (Hugging Face / folder to GGUF)

The coordinator turns Hugging Face weights into GGUF (see PLATFORM_DESIGN for the pipeline). The UI has
two parts: the **Convert dialog** and the **Conversions panel** on the Models page. Both use the
existing design tokens (`--surface-2`, `--accent`, `--warning-soft`, `--danger-soft`, `--info-soft`,
`.pill`, `.chip`, `.notice`, `.seg`, `.tabs`); no literal colours.

### 9.1 Entry points

- **Convert a model** button in the library header and in the Conversions panel header.
- In *Add model*: when a Hugging Face repo has no `.gguf` files, a notice offers **Convert to GGUF**
  (the repo is prefilled and inspected at once); a "Not a GGUF repository? Convert it" link shows
  while nothing is listed yet.
- Library rows of converted files show the source **Converted**.

### 9.2 Convert dialog

1. **Source tabs**: *Hugging Face repo* (repo id, optional revision, Inspect) and *Folder on the
   server* (absolute path; host paths are translated like library paths when the coordinator is in
   Docker). Enter in a field starts the inspection. Errors (404, gated, bad folder, **507 not enough
   disk**: the message names the folder, needed and free GB) appear in a red block under the tabs.
2. **Inspect facts**: architecture, parameters, layers, context length, weight format, download size
   (or size on disk) in a six-cell grid, a status pill (Supported / Not supported / Support unknown /
   Cannot convert), a collapsible list of the files used and skipped.
3. **Notices** (only those that apply): cannot convert (with *Inspect the base model instead* when the
   repo is pre-quantized), gated repository, already quantized, ships its own Python code, ready-made
   GGUF alternatives (*download instead* hands over to Add model), converter warnings.
4. **Quantization picker**: a radio list, best quality first. Each row: type, tier pill (Lossless,
   Near-lossless, Balanced, Small, Tiny), *Recommended*, **Needs calibration** badge (types that need
   an importance matrix), a fit badge (*Fits one GPU* / *Needs several GPUs* / *Does not fit the
   cluster*), the quality note, estimated file size and VRAM. The recommendation and its reasons are
   shown above the list. Types below 3 bits per weight (IQ2, IQ1) are folded behind **Show smaller,
   lower-quality types (n)**; the selected type is always visible. When `imatrix_available` is false
   in `GET /api/convert/options`, the types that need an importance matrix are greyed, disabled and
   say why.
5. **Output file name**: built from `name_stem` of the inspection plus the type
   (`<name_stem>-<QUANT>.gguf`); it follows the type until the user edits it. Invalid names block
   Start.
6. **Keep downloaded source** (Hugging Face only) and the collapsible **Advanced** section:
   intermediate precision, output tensor type, token embedding type, leave output tensor, pure
   (these four are disabled for F16/BF16/Q8_0, which the converter writes directly),
   **Importance matrix**, validate generation, allow remote code (red warning), threads.
   - *Importance matrix* is a three-way control **Auto / On / Off**. Help text: what it is (a pass
     over sample text that records which weights matter, so quantizing keeps those precise), that
     Auto means on for types under about 4 bits and for types that need it, that On costs extra time
     (roughly one pass of the model over the text, on the CPU), and that Off is not possible for
     types that need it (the button is disabled; picking such a type while Off is set switches to
     Auto). A line under the control says whether a Calibrate step will run. For F16/BF16/Q8_0 the
     whole block is disabled with an explanation. Without llama-imatrix the control is fixed to Off.
   - **Calibration text** (optional): absolute path of a `.txt` file on the server, at most 20 MB;
     empty = gpupool's built-in multilingual text. Validated in the browser (absolute, `.txt`).
   - **Calibration chunks**: number of 512-token pieces; 0 = default (100).
7. Footer: the reason Start is disabled (toolchain missing, invalid name, bad calibration path), and
   **Start conversion**. On success the dialog closes, the Models page opens and a toast confirms.

### 9.3 Conversions panel

One card per job, newest first:

- Header: output name, state pill (spinner while running), quant chip, an *importance matrix* chip
  when the job computes one; source and timing line (*queued 2m ago*, *started 2m ago*, or
  *ran 3m 20s, finished 5m ago* from `started_at` / `finished_at`).
- **Stepper**: Download, Convert, **Calibrate** (only when `imatrix_used`), Quantize, Validate.
  Done steps are green, the running step pulses, a failed job marks `failed_stage` in red, a
  cancelled job marks it in the warning colour; Download is dashed (skipped) for folders and Quantize
  for the direct types. Jobs without `failed_stage` fall back to the last stage seen while polling.
  On mobile only the running or failed step keeps its label.
- Progress bar (determinate for download and for stages with `stage_progress`, otherwise a moving
  bar) with text (bytes, percent, "Computing the importance matrix on the CPU: 45%").
- Meta: estimated size, final size, source files kept.
- Error box (`error`), a *Needs your review* notice for `needs_review`.
- **Validation panel** (collapsible, open by itself for needs_review): check chips (GGUF header,
  tokenizer, generated text, chat template), architecture / layers / vocabulary, errors and warnings,
  the tokenizer table (HF ids against GGUF ids, mismatching rows highlighted) and the generation
  sample.
- **Log** (collapsible): `log_tail`.
- Actions by state: *Deploy this model* (done, opens the New model form), *Accept anyway*
  (needs_review, asks for confirmation), *Cancel* (active), *Retry* (failed / cancelled), delete icon
  (not active), each with a confirmation where it destroys something.

### 9.4 Polling, toasts, states

- `GET /api/convert` every 2 s while a job is active, every 10 s otherwise while the Models page is
  open. Transitions seen between two polls raise toasts: converted (and the library refreshes),
  needs review, failed.
- Empty state: "No conversions yet". Unavailable state: a warning notice with the reason from
  `options.problem` (inspecting still works, starting is disabled); 404 means the coordinator has
  no conversion support. Polling errors show in a red block, never as an empty list.

### 9.5 Mobile and accessibility

- The dialog is a bottom-fitting modal; quant rows wrap (numbers move under the text), the facts
  grid collapses, no horizontal scroll (checked at 375 px).
- The picker is a `radiogroup` with a label per radio; the low-bit toggle has `aria-expanded`; the
  importance-matrix control is a `radiogroup`; progress bars use `role="progressbar"`; stepper steps
  carry a screen-reader state; errors use `role="alert"`; the dialog has `aria-modal` and a title.
  Colour is never the only signal (labels and text accompany every pill).

## 10. Model card: copy buttons, and limiting a model to some servers or GPUs

### 10.1 Copy

The endpoint row of a model card has two lines: **endpoint** with a *Copy endpoint* icon button, and
**model** (the name clients put in `"model"`) with a *Copy model name* icon button and a small
*Copy curl* action that copies a ready `curl` for this model (with an `Authorization` header
placeholder when API keys are set). Each shows a "Copied" toast. The lines wrap instead of overflowing
on mobile.

### 10.2 Allowed servers and GPUs (`pin_devices`)

`pin_devices` is an **allowed set**, not a manual placement: the scheduler still picks the best
placement, but only on devices in the set. Entries are `"<node>/<device>"` (one GPU) or
`"<node>/*"` (every device of that server, GPUs added later included). Empty = everything. It does not
change the Servers tab: GPUs stay enabled in the pool for other models.

In the New / Edit model form:

- **Use**: a two-way control *All servers and GPUs* (default) / *Only selected ones*, with the help
  text above.
- **Selection tree** (shown for *Only selected ones*): a checkbox per server (checked = `"<node>/*"`,
  indeterminate when only some GPUs are picked) with its GPUs below, each with free and usable
  memory. Checking a server stores `"<node>/*"` and drops that server's single-GPU pins; unchecking
  one GPU of a whole-server pick turns it into the explicit remaining `"<node>/<device>"` pins.
  Offline servers and disabled GPUs are greyed with the reason but stay selectable.
- A summary ("3 of 7 GPUs on 2 servers"); with nothing selected a warning is shown and Save refuses
  (an empty list would mean "everything").
- Saved pins, including wildcards, load back into the tree when the form is reopened. Recommend,
  Check placement and Preview impact send the same `pin_devices`.
- The model card shows a chip such as *Limited to server-a, server-b/CUDA1* (`/*` is shown as the
  server name).
