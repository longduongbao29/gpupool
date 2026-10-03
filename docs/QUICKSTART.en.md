# gpupool — Quick start

> English version. Vietnamese version: [QUICKSTART.vi.md](QUICKSTART.vi.md). Keep both in sync.

Three steps, one command each. Nothing to configure by hand: keys are generated, servers find their
own name and IP, and they join the pool by themselves.

```
1. coordinator machine   docker run ... gpupool-coordinator        (once)
2. every GPU server      docker run ... gpupool-agent  (join cmd)   (once per server)
3. your application     base_url = http://<coordinator>:8080/v1    (any OpenAI client)
```

## 1. Start the coordinator (one machine, no GPU needed)

```bash
docker run -d --name gpupool --restart unless-stopped -p 8080:8080 -v gpupool:/data \
  ghcr.io/longduongbao29/gpupool-coordinator
docker logs gpupool
```

The log prints the **admin key**. Open `http://<this machine's IP>:8080` in a browser and log in with it.
Keys are generated on first start and stored in the `gpupool` volume, so they stay the same after a restart.

The log also prints a join command, but inside Docker it shows the container's internal IP
(`172.17.x.x`). Take the join command from the UI instead: it uses the address you opened the UI with.

## 2. Add each GPU server (one command per server)

In the UI open **Servers → Add Server**, copy the join command and run it on the GPU server.
It looks like this:

```bash
docker run -d --name gpupool-agent --restart unless-stopped --gpus all --network host --pid host \
  -v gpupool-agent:/data \
  -e GPUPOOL_JOIN="http://10.0.0.1:8080#<cluster-token>" \
  ghcr.io/longduongbao29/gpupool-agent
```

The server appears in the UI within a few seconds with all its GPUs. Its name is the hostname and its
IP is detected automatically.

Requirements on the GPU server: an NVIDIA driver ≥ 525, Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Check with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 3. Serve a model (in the UI)

1. **Models → Add model**: type a Hugging Face repo (e.g. `Qwen/Qwen2.5-7B-Instruct-GGUF`), pick a
   `.gguf` file, **Download**. Or the **Path** tab for a file already on the coordinator machine.
2. **New model**: give it a name (this is the `model` your clients will use), pick the file.
   GPUs: leave **Auto**, or tick the GPUs to use.
3. **Start**. The status goes *starting → running*. **Stop** frees the GPUs.

If the model is bigger than any single GPU, it is split across GPUs and servers automatically.

## 4. Connect your application

Any OpenAI-compatible client works. Only two values change:

| Setting | Value |
| --- | --- |
| Base URL | `http://<coordinator>:8080/v1` |
| Model | the name you gave the model in the UI |
| API key | none needed by default; if you set `GPUPOOL_API_KEYS`, one of those keys |

curl:

```bash
curl http://10.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

Python:

```python
from openai import OpenAI

client = OpenAI(base_url="http://10.0.0.1:8080/v1", api_key="none")
r = client.chat.completions.create(model="qwen7b", messages=[{"role": "user", "content": "Hello"}])
print(r.choices[0].message.content)
```

Tools like Open WebUI, LangChain, LlamaIndex or Continue: choose "OpenAI-compatible" and enter the same
base URL. Streaming (`stream: true`) is supported. The Settings page of the UI shows these snippets with
your real address.

To require a key from clients, start the coordinator with `-e GPUPOOL_API_KEYS=key1,key2`.

## Try it on one machine (simulated 3-server cluster)

`docker-compose.sim.yml` starts a coordinator and three "servers" joined exactly like a real install
(`GPUPOOL_JOIN`), so you can try the UI, multi-server placement and RPC splits with one GPU. Build both
images first (`docker compose -f docker-compose.coordinator.yml build` and the agent compose file), then:

```bash
GPUPOOL_HOST_MODELS_DIR=/srv/gguf docker compose -f docker-compose.sim.yml up -d
```

UI: `http://<docker host>:8080`, admin key `sim-admin`. server-a reports the real GPU; server-b and
server-c report a simulated Tesla T4 and A100 (`GPUPOOL_FAKE_DEVICES`) but run on the same real GPU. The
budgets (`SIM_BUDGET_A/B/C`, default 1300/1100/1100 MB) must add up to less than the real GPU memory.
Each server gets its own IP on a private Docker network instead of `--network host`. The RPC firewall is on
in all three (`cap_add: [NET_ADMIN]`), so the multi-server splits also exercise it. Stop and wipe it with
`docker compose -f docker-compose.sim.yml down -v`. Overrides: `SIM_ADMIN_KEY`, `SIM_CLUSTER_TOKEN`,
`GPUPOOL_AGENT_IMAGE`, `GPUPOOL_COORDINATOR_IMAGE`.

## CI end-to-end test (no GPU needed)

`scripts/ci_e2e.py` runs the real Docker images against `docker-compose.ci.yml`: a coordinator and three
CPU-only servers (`GPUPOOL_INCLUDE_CPU`, RPC firewall on). Each server offers one CPU device with a budget of
`CI_BUDGET_MB` (default 120 MB), smaller than the test model, so SmolLM2-135M (Q8_0, about 140 MB) must be
split over at least two servers through llama.cpp RPC. It needs Docker, Python 3 (stdlib only) and internet
for the first model download (verified by SHA-256, then cached). Build the images first (or set
`GPUPOOL_AGENT_IMAGE` / `GPUPOOL_COORDINATOR_IMAGE`), then:

```bash
python scripts/ci_e2e.py                       # options: --project gpupool-ci --port 8080 --keep
```

It checks: the three agents self-join, the model library, the API surface, start to *running*, a chat
completion, `/metrics`, and stop (no engine left on any agent). Exit code 0 means all passed; on failure
it prints `docker compose logs` and exits 1. The cluster is removed afterwards unless `--keep` is given.
The model is cached in `GPUPOOL_CI_MODELS_DIR` (default `<repo>/.cache/ci-models`).

## Performance options (per model, in the deploy form)

| Option | Values | Effect | Measured (GTX 1650, Qwen2.5-3B) |
| --- | --- | --- | --- |
| KV cache | f16, q8_0, q4_0 | smaller KV cache, so a model can fit on fewer GPUs | ctx 8192: −132 / −204 MB, speed unchanged (51.9 / 51.2 / 50.8 tok/s) |
| Speculative | none, ngram, draft | fewer passes of the big model per token, i.e. fewer RPC round trips when split | split over 2 servers: none 48.9, ngram 53.6, draft 0.5B 53.9 tok/s |

In the API these are `kv_cache_type` (`f16`, `q8_0`, `q4_0`), `speculative` (`none`, `ngram`, `draft`),
`draft_file` (a library model, for `draft`) and `draft_n_max` (1 to 16, default 4); see [API.en.md](API.en.md).
They apply the next time the model starts.

N-gram needs no extra memory but only helps when the output repeats earlier text. A draft model must share
the tokenizer of the main model (checked when saving) and runs on the head's GPU (its memory is planned for).
Drafting 4 tokens is the default; 8 was slower in our measurements.

## Without Docker

Coordinator:

```bash
git clone https://github.com/longduongbao29/multi-gpu-inference && cd multi-gpu-inference
uv sync && uv run gpupool coordinator          # prints the same admin key and join command
```

GPU server (needs [uv](https://docs.astral.sh/uv/) and a llama.cpp build with CUDA and RPC, b11342):

```bash
uv run gpupool agent --join "http://10.0.0.1:8080#<cluster-token>" --llama-dir /path/to/llama.cpp/build/bin
```

## Network

| Port | Open between | Used for |
| --- | --- | --- |
| 8080 | clients and GPU servers → coordinator | UI, API, model downloads |
| 7070 | coordinator → GPU servers | starting and stopping engines |
| 9000–9999 | coordinator and GPU servers → GPU servers | llama.cpp (HTTP and RPC between servers) |

llama.cpp RPC is not encrypted: keep the GPU servers on a trusted internal network.

## Security

- **RPC ports are unauthenticated.** `ggml-rpc-server` (ports 9000–9999 on every GPU server) accepts any
  connection: whoever can reach it can allocate GPU memory and read or write tensors. Restrict it in one of two ways:
  - set `GPUPOOL_RPC_FIREWALL=1` on the GPU server. For every RPC engine the agent adds iptables (and ip6tables)
    rules in its own chain that allow loopback, the head node of the replica and the agent's own address on that
    port, and **drop everything else**. The rules are removed when the engine stops, and rules left by a crashed
    agent are flushed at startup. It needs root and iptables (included in the image); in Docker add
    `--cap-add NET_ADMIN` (compose: `cap_add: [NET_ADMIN]`, already set in the agent, sim and CI compose files).
    Without that the agent logs an error and runs unprotected (see Troubleshooting);
  - or use your own firewall: allow 9000–9999 only between cluster hosts.
- The agent warns at startup when the firewall is off.
- Keys: the **admin key** (`GPUPOOL_ADMIN_KEY`) protects the UI and admin API; the **cluster token**
  (`GPUPOOL_CLUSTER_TOKEN`) authenticates servers to the coordinator and is part of the join command; **API keys**
  (`GPUPOOL_API_KEYS`) are what clients of `/v1` send. Admin key and cluster token are generated on first start and
  stored in the data volume (`secrets.json`); set your own to override.
- Without `GPUPOOL_API_KEYS`, `/v1` is open to anyone who reaches port 8080: set them.
- The cluster token and API keys travel over plain HTTP (no TLS): keep the cluster on a private network or VPN, and
  put a TLS-terminating proxy in front of the coordinator if clients connect from outside.

## VRAM self-calibration

The planner's memory estimates are corrected from reality. Once a replica is running, the coordinator compares
the memory the engines really use with the estimate and keeps a smoothed factor per model (measured / estimated).
The next placement of that model uses it, so a model that needs more or less than estimated stops
over-committing or wasting GPUs. You see it in `/api/state`: each model has `calibration`, either
`{"factor": ..., "samples": ...}` or `null` until the first measurement; a `calibrated` event is logged when the
factor moves noticeably. There is nothing to configure. Removing a model forgets its factor.

## What survives a coordinator restart

Everything in the data volume (`/data`): files, secrets, the database. Engines run on the servers, so they keep
running while the coordinator restarts. The control state is persisted and loaded again at boot:

- autoscaler state per model: desired replica count, last request time and last decision (an unloaded on-demand
  model stays unloaded, a scaled-up model stays scaled up); only the "up/down since" timers restart, which just
  delays a scaling step;
- preemption claims and cooldowns, crash-loop backoff counters, and the rebalance move in flight;
- not kept: the periodic rebalance timer, which restarts at boot because the first reports are stale.

A launch interrupted by a hard kill (`kill -9`, OOM, power loss) cannot be resumed: at the next start that replica
is marked *failed* ("coordinator restarted during launch"), its half-started engines are stopped on the servers,
and the next reconcile tick plans the model again. A clean shutdown does the same itself. Hugging Face downloads
interrupted by a restart are resumed.

## Behind an HTTP proxy

If your servers reach the internet through a proxy, pass the host's proxy variables into the containers
(lowercase `http_proxy`, `https_proxy`, `no_proxy` are read first, uppercase also works):

```bash
docker run -d ... -e http_proxy -e https_proxy -e no_proxy ghcr.io/longduongbao29/gpupool-agent
```

With docker compose they are passed from the host environment or `.env` automatically.

- Through the proxy: Hugging Face (file listing and downloads), model downloads from `https://` URLs on the
  servers, and the alert webhook.
- Never through the proxy: traffic between the coordinator, the servers and llama.cpp (heartbeats, engine
  control, model files from the coordinator, requests to your models).
- `no_proxy` is honoured: comma-separated hosts, domain suffixes (`.corp.local`), IPs, CIDRs (`10.0.0.0/8`) or `*`.

To build the images locally behind a proxy:

```bash
docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy   --build-arg no_proxy=$no_proxy -f docker/agent.Dockerfile -t gpupool-agent .
```

(`docker compose ... --build` passes them for you.) The proxy values are not stored in the image.

What you still have to set up yourself on every machine that runs Docker:

1. **The Docker daemon's proxy**, for pulling base images (`nvidia/cuda`, `python`, the uv image). The proxy
   variables of your shell and the build args are not used for that, the daemon pulls. Create
   `/etc/systemd/system/docker.service.d/http-proxy.conf`:

   ```ini
   [Service]
   Environment="HTTP_PROXY=http://proxy.corp.local:3128"
   Environment="HTTPS_PROXY=http://proxy.corp.local:3128"
   Environment="NO_PROXY=localhost,127.0.0.1,.corp.local,10.0.0.0/8"
   ```

   then `sudo systemctl daemon-reload && sudo systemctl restart docker`. Check with
   `systemctl show --property=Environment docker`.
2. **Optional: `~/.docker/config.json`.** The Docker client then injects the proxy into every build
   (as build args) and every container (as environment) by itself, so you do not need `-e` or `--build-arg`:

   ```json
   { "proxies": { "default": {
       "httpProxy": "http://proxy.corp.local:3128",
       "httpsProxy": "http://proxy.corp.local:3128",
       "noProxy": "localhost,127.0.0.1,.corp.local,10.0.0.0/8" } } }
   ```
3. **`no_proxy` must list the coordinator and every GPU server** (IPs, hostnames, `.domain` suffixes or CIDRs
   such as `10.0.0.0/8`). gpupool itself never proxies cluster traffic, but a proxy set for other tools (curl, apt, ...)
   in the same container or on the host would.
   Always include `localhost,127.0.0.1`.
4. `git clone` (during the agent build) honours `http_proxy` / `https_proxy`. If the proxy blocks GitHub
   completely, use [Building without GitHub](#building-without-github).

| Symptom | Cause | Fix |
| --- | --- | --- |
| `docker pull` / `FROM` step: `i/o timeout`, `TLS handshake timeout`, `connection refused` | the daemon has no proxy | step 1 (daemon proxy), restart docker |
| Build step `git clone` hangs or `Failed to connect to github.com` | GitHub blocked even through the proxy | put `vendor/llama.cpp-b11342.tar.gz` in the repo, or set `LLAMA_CPP_URL` |
| `uv sync`: `Failed to download ... cpython-3.12` | python-build-standalone is hosted on GitHub | set `UV_PYTHON_INSTALL_MIRROR` (see below) |
| `failed to resolve source metadata for ghcr.io/astral-sh/uv` | ghcr.io unreachable | set `UV_IMAGE` to a mirror; coordinator: `UV_FROM_PYPI=1` |
| `apt-get` or `pip` fails inside the build | proxy build args missing | pass `--build-arg http_proxy=... https_proxy=...` or use compose / `config.json` |
| `407 Proxy Authentication Required` | the proxy wants credentials | use `http://user:pass@host:port`; URL-encode special characters in the password |
| Hugging Face search or download fails in the UI, other things work | the coordinator container has no proxy variables | set `http_proxy` / `https_proxy` in `.env` (compose) or `-e` (docker run) and recreate the container |
| Servers go offline, model requests fail, only when a proxy is set | cluster traffic is sent through the proxy | add the coordinator and server IPs or CIDR to `no_proxy` |
| Proxy works for `curl` but not for the container | lowercase vs uppercase variables | set both spellings (compose files already pass both) |

## Building without GitHub

On a server that cannot reach GitHub (or git), the images can be built from files you copy in. Nothing here
is needed when the proxy lets GitHub through.

1. **llama.cpp source** (agent image). On a machine that can reach GitHub, download the tag archive into
   `vendor/` of this repo and copy the repo to the server:

   ```bash
   curl -L -o vendor/llama.cpp-b11342.tar.gz https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11342.tar.gz
   ```

   The build uses that file first, then `git clone`, then `curl` of `LLAMA_CPP_URL` (an internal mirror of the
   same archive). Without `.git` the build number is passed to CMake explicitly, so `llama_version` stays correct.
2. **Python 3.12** (agent image). The CUDA Ubuntu base has no Python 3.12, so uv downloads one from
   python-build-standalone. Put the archive in `vendor/python/<release>/` (layout and the exact file name:
   [vendor/README.md](../vendor/README.md)) and build with `UV_PYTHON_INSTALL_MIRROR=file:///vendor/python`, or
   point it at an internal HTTP mirror.
3. **The uv image** (both images). `UV_IMAGE=registry.corp.local/astral-sh/uv:0.10` for a registry mirror. The
   coordinator image can use `UV_FROM_PYPI=1` instead: uv is installed with pip, and PyPI usually works through the
   proxy.
4. Optional: `LLAMA_USE_PREBUILT_UI=OFF` skips llama.cpp's download of its web UI (gpupool does not use it).

With docker compose put them in `.env` (see `.env.example`), then:

```bash
docker compose -f docker-compose.agent.yml up -d --build            # agent
docker compose -f docker-compose.coordinator.yml up -d --build      # coordinator
```

Or with plain docker:

```bash
docker build -f docker/agent.Dockerfile -t gpupool-agent \
  --build-arg UV_PYTHON_INSTALL_MIRROR=file:///vendor/python \
  --build-arg UV_IMAGE=registry.corp.local/astral-sh/uv:0.10 .
docker build -f docker/coordinator.Dockerfile -t gpupool-coordinator --build-arg UV_FROM_PYPI=1 .
```

The vendored files need BuildKit (the default since Docker 23; older: `DOCKER_BUILDKIT=1`). Do not commit them.
Python packages still come from PyPI (through the proxy, or `UV_INDEX_URL` / `PIP_INDEX_URL` for an internal index).

## Using model files already on the server

Do not copy a model into the container: mount the folder that holds it. The coordinator is the only machine
that needs the files: it serves them to the GPU servers, so nothing has to be copied to them.

1. In `.env` on the coordinator machine set the folder, as an **absolute path** with forward slashes:

   ```
   GPUPOOL_HOST_MODELS_DIR=/srv/gguf
   ```

   `docker-compose.coordinator.yml` mounts it read-only at `/models` and tells the coordinator the mapping
   (`GPUPOOL_PATH_MAP`). With plain docker: `-v /srv/gguf:/models:ro -e GPUPOOL_PATH_MAP='{"/srv/gguf": "/models"}'`.
2. In the UI (**Models → Add model → Path**) browse `/models` (the coordinator lists `.gguf` files there), or type
   the path. Both `/srv/gguf/qwen.gguf` (the host path) and `/models/qwen.gguf` work: a host path that does not
   exist inside the container is rewritten to `/models/...`.
3. **Symlinks must point inside the mounted folder.** A link whose target is outside the mount is broken inside
   the container. This is how the Hugging Face cache is laid out (`snapshots/<rev>/file.gguf` links to
   `../../blobs/<hash>`): mount the whole `hub/models--<org>--<repo>` folder (or `hub`), not only
   the `snapshots` folder, or copy the files with `cp -L` / `cp --dereference`.
4. Files on the GPU servers are not needed.

Several folders: add more `-v` mounts, extend `GPUPOOL_MODEL_ROOTS` (comma-separated) and `GPUPOOL_PATH_MAP`
(one JSON entry per folder).

## Settings (environment variables)

Every setting can be given as `GPUPOOL_<NAME>` (the field name in upper case), in a TOML file (`--config`) or,
for a few, a CLI flag. Precedence: CLI flag > environment > TOML file > default. Lists are comma-separated,
dicts (`GPUPOOL_PATH_MAP`, `GPUPOOL_BUDGET_MB`) are JSON, booleans accept `1/0/true/false`, a port range is
`9000-9999`. Everything below is optional except `GPUPOOL_JOIN` on an agent.

### Coordinator (22 settings)

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUPOOL_HOST` | `0.0.0.0` | address the coordinator listens on |
| `GPUPOOL_PORT` | `8080` | listening port |
| `GPUPOOL_DB_PATH` | `.gpupool/coordinator.db` (image: `/data/coordinator.db`) | SQLite database; `secrets.json` is stored next to it |
| `GPUPOOL_ADMIN_KEY` | generated | key for the UI and the admin API |
| `GPUPOOL_CLUSTER_TOKEN` | generated | token servers use to join |
| `GPUPOOL_API_KEYS` | empty (open) | comma-separated keys clients must send to `/v1` |
| `GPUPOOL_MODELS_DIR` | `.gpupool/models` (image: `/data/models`) | model library, served to the servers at `/files/<name>` |
| `GPUPOOL_PATH_MAP` | `{}` | JSON `{"<host dir>": "<dir in container>"}` so users can type host paths |
| `GPUPOOL_MODEL_ROOTS` | empty (image: `/models`) | comma-separated folders the UI may browse for `.gguf` (empty = models dir only) |
| `GPUPOOL_HEARTBEAT_TIMEOUT_S` | `10` | seconds without a report before a server is considered down |
| `GPUPOOL_RECONCILE_S` | `2` | how often desired and actual state are reconciled |
| `GPUPOOL_MAX_REQUEST_MB` | `32` | largest `/v1` request body in MB (bigger gets HTTP 413) |
| `GPUPOOL_COLD_START_TIMEOUT_S` | `120` | how long a request waits for an unloaded (on-demand) model before HTTP 503 |
| `GPUPOOL_REBALANCE_S` | `600` | how often replicas with a clearly better placement are moved (one at a time, new one first); `0` = only on demand (`POST /api/rebalance`) |
| `GPUPOOL_LAUNCH_TIMEOUT_S` | `600` | a replica not ready after this long is marked failed |
| `GPUPOOL_LOW_FREE_MB` | `256` | a device with less free memory than this while hosting an engine makes the replica move |
| `GPUPOOL_DRAIN_TIMEOUT_S` | `60` | how long a stopping replica gets to finish its requests |
| `GPUPOOL_PORT_RANGE` | `9000-9999` | ports handed to engines on the servers (open them between servers) |
| `GPUPOOL_POLL_S` | `2` | how often the coordinator polls each server's `/report` |
| `HF_TOKEN` (or `GPUPOOL_HF_TOKEN`) | empty | for gated or private Hugging Face repos |
| `GPUPOOL_PUBLIC_URL` | detected | address servers use to reach the coordinator, if detection is wrong (used in the join command) |
| `GPUPOOL_WEBHOOK_URL` | empty | Slack/Discord/generic JSON webhook for alerts (server down, GPU lost, not enough VRAM) |

The Docker image sets `GPUPOOL_HOST`, `GPUPOOL_PORT`, `GPUPOOL_DB_PATH`, `GPUPOOL_MODELS_DIR` and
`GPUPOOL_MODEL_ROOTS` as shown. `docker-compose.coordinator.yml` also reads `GPUPOOL_HOST_MODELS_DIR` (host folder
mounted at `/models`) and the proxy variables.

### Agent, one per GPU server (17 settings)

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUPOOL_JOIN` | none | `http://<coordinator>:8080#<cluster-token>`: replaces the next two (`--join`) |
| `GPUPOOL_COORDINATOR_URL` | `http://127.0.0.1:8080` | coordinator address (wins over `GPUPOOL_JOIN`) |
| `GPUPOOL_CLUSTER_TOKEN` | empty | cluster token (wins over `GPUPOOL_JOIN`) |
| `GPUPOOL_AUTO_JOIN` | `true` | register with the coordinator by itself (`--no-auto-join` turns it off; then add the server in the UI) |
| `GPUPOOL_NODE_ID` | hostname | name of the server in the pool |
| `GPUPOOL_HOST` | auto-detected | address other servers reach this one on, and engines bind to; default is the local IP used to reach the coordinator |
| `GPUPOOL_PORT` | `7070` | agent API port |
| `GPUPOOL_LLAMA_DIR` | required (image: `/opt/llama`) | folder with `llama-server` and `rpc-server` (`--llama-dir`) |
| `GPUPOOL_CACHE_DIR` | `.gpupool/cache` (image: `/data/cache`) | downloaded GGUF files |
| `GPUPOOL_LOG_DIR` | `.gpupool/logs` (image: `/data/logs`) | one log file per engine |
| `GPUPOOL_MARGIN_PCT` | `0.10` | share of each GPU's memory always left free for other users |
| `GPUPOOL_MARGIN_MIN_MB` | `512` | ...but never less than this |
| `GPUPOOL_BUDGET_MB` | `{}` | JSON cap per device, e.g. `{"CUDA0": 1200, "CPU": 2000}`; also gives a CPU device its size |
| `GPUPOOL_INCLUDE_CPU` | `false` | also offer a `CPU` device (served through `rpc-server -d CPU`) |
| `GPUPOOL_HEARTBEAT_S` | `2` | heartbeat interval (only used with push heartbeats) |
| `GPUPOOL_PUSH_HEARTBEAT` | `false` | also push heartbeats, for coordinators older than pull mode; normally leave off |
| `GPUPOOL_RPC_FIREWALL` | `false` | `1` = restrict each RPC port with iptables, see [Security](#security); needs root and `NET_ADMIN` |

Read by the programs but not part of the settings above: `GPUPOOL_LOG` (log level, default `INFO`, both
programs), `GPUPOOL_FAKE_DEVICES` (agent: JSON list of simulated GPUs, used by `docker-compose.sim.yml`), and
`GPUPOOL_URL` (default `http://127.0.0.1:8080`) plus `GPUPOOL_ADMIN_KEY` for the CLI commands
`gpupool register | plan | scale | undeploy | status`. `docker-compose.agent.yml` reads `GPUPOOL_JOIN` (required),
`GPUPOOL_NODE_ID`, `GPUPOOL_HOST`, `GPUPOOL_MARGIN_PCT`, `GPUPOOL_RPC_FIREWALL`, `CUDA_ARCHS` and the build args
described above. HTTP API: [API.en.md](API.en.md). Internals: [DESIGN.en.md](DESIGN.en.md).

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Server never appears in the UI | `docker logs gpupool-agent`: "connection refused" → port 8080 blocked or wrong address in the join command (use the coordinator's LAN IP); "wrong cluster token" → copy the join command again |
| Agent log says the server was removed | it was deleted in the UI; add it again there (Servers → Add Server → agent URL `http://<server-ip>:7070`) |
| Model stuck in *failed*: "not enough VRAM" | free GPUs, enable more GPUs, add a server, or use a smaller quantization |
| `could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA Container Toolkit is not installed on that server |
| Agent log: `RPC firewall unavailable (...); RPC ports are NOT restricted` | `GPUPOOL_RPC_FIREWALL=1` but iptables is missing or the container lacks root/`NET_ADMIN`: add `--cap-add NET_ADMIN` (compose `cap_add: [NET_ADMIN]`) and recreate the container, or unset the variable and firewall 9000–9999 yourself |
| Agent log: `RPC firewall: ... rule for port N failed ... port left unrestricted` | one iptables call failed and the rule was rolled back; read the message (usually the same missing capability). `ip6tables unavailable` only means IPv6 is not restricted |
| Multi-server model stuck *starting* while the firewall is on, engine on another server never answers | the peer is refused. `RPC firewall: cannot resolve peer ...` means a name does not resolve on that server: use IPs (`GPUPOOL_HOST`) or fix DNS. To confirm, unset `GPUPOOL_RPC_FIREWALL` on that server and recreate it |
| Model was *launching* when the coordinator was killed | marked *failed* ("coordinator restarted during launch") and planned again by itself within a reconcile tick |
| Windows / WSL2: every container restarts about a minute after the last terminal is closed | WSL shuts its VM down about 1 minute after the last `wsl.exe` session, even with Docker running. In `%UserProfile%\.wslconfig` add `vmIdleTimeout=-1` under `[wsl2]`, then run `wsl --shutdown` and start Docker again |
| Windows / WSL2: a bigger model fails to load or the engine is killed (out of memory) | WSL2 caps memory by default (4 GB here). Raise `memory=` under `[wsl2]` in `%UserProfile%\.wslconfig` (for example `memory=16GB`), then `wsl --shutdown` |
