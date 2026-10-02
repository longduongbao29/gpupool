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

## Optional settings (environment variables on the coordinator)

| Variable | Default | Purpose |
| --- | --- | --- |
| `GPUPOOL_API_KEYS` | empty (open) | comma-separated keys clients must send |
| `GPUPOOL_MAX_REQUEST_MB` | 32 | largest request body accepted on /v1, in MB (bigger gets HTTP 413) |
| `GPUPOOL_COLD_START_TIMEOUT_S` | 120 | how long a request waits for an unloaded (on-demand) model to load before HTTP 503 |
| `GPUPOOL_REBALANCE_S` | 600 | how often replicas with a clearly better placement are moved (one at a time, new one first); 0 = only on demand |
| `GPUPOOL_PUBLIC_URL` | detected | address servers use to reach the coordinator, if detection is wrong |
| `GPUPOOL_WEBHOOK_URL` | empty | Slack/Discord webhook for alerts (server down, GPU lost, not enough VRAM) |
| `HF_TOKEN` | empty | for gated or private Hugging Face repos |
| `GPUPOOL_ADMIN_KEY`, `GPUPOOL_CLUSTER_TOKEN` | generated | set your own instead of generated ones |

On a GPU server: `GPUPOOL_MARGIN_PCT` (default `0.10`) is the share of each GPU's memory always left
free for other users.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Server never appears in the UI | `docker logs gpupool-agent`: "connection refused" → port 8080 blocked or wrong address in the join command (use the coordinator's LAN IP); "wrong cluster token" → copy the join command again |
| Agent log says the server was removed | it was deleted in the UI; add it again there (Servers → Add Server → agent URL `http://<server-ip>:7070`) |
| Model stuck in *failed*: "not enough VRAM" | free GPUs, enable more GPUs, add a server, or use a smaller quantization |
| `could not select device driver "" with capabilities: [[gpu]]` | the NVIDIA Container Toolkit is not installed on that server |
