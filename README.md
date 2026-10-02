# gpupool

> English. Tiếng Việt: [README.vi.md](README.vi.md)

Pool scattered free GPU memory across several servers (say 3 GB + 5 GB + 10 GB on three machines)
and serve LLMs through one OpenAI-compatible endpoint. The engine is
[llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server` + `ggml-rpc-server`, GGUF models);
gpupool is the control plane on top: it measures free VRAM on every node, decides where each model
goes and how its layers are split, launches and supervises the engines, routes requests, and
re-places replicas when a node dies.

Built for shared servers: user space only (no sudo), mixed CUDA versions, VRAM that other people
use too.

## Status

Early, tested on one Windows laptop emulating three nodes against real llama.cpp. See
[docs/TEST_REPORT.en.md](docs/TEST_REPORT.en.md). Not yet run on a real multi-server cluster.

## How it works

```
client ──OpenAI API──► coordinator (router + scheduler + reconciler, SQLite)
                            ▲ heartbeats            │ start/stop engines
   server A: agent ─ llama-server (head) ──RPC──► server B: agent ─ ggml-rpc-server
                                         └─RPC──► server C: agent ─ ggml-rpc-server
```

- **agent** (one per server): reports GPUs via NVML, starts/stops llama.cpp processes, caches GGUF files.
- **scheduler**: reads the GGUF header (no full download), estimates memory per layer, prefers one GPU,
  then one server, then the fewest servers; GPUs before CPU RAM.
- **router**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming, prefix-aware
  load balancing (shared system prompts hit the replica that has them cached), retry before the first byte.
- **reconciler**: keeps the desired replica count, rolls back failed launches, fails over dead nodes, drains.

Design: [docs/DESIGN.en.md](docs/DESIGN.en.md).

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), a llama.cpp build (b11342 tested) with `llama-server`
and `ggml-rpc-server` matching each node's CUDA driver.

```bash
git clone <this repo> && cd multi-gpu-inference
uv sync
```

Coordinator (`coordinator.toml`):

```toml
host = "0.0.0.0"
port = 8080
cluster_token = "change-me"
admin_key = "change-me-too"
api_keys = ["client-key"]
models_dir = "/data/gguf"   # files here are served to agents as coordinator://<file>
```

Each server (`agent.toml`):

```toml
node_id = "server-a"
host = "10.0.0.5"            # an IP the other servers can reach (not 0.0.0.0)
port = 7070
coordinator_url = "http://10.0.0.1:8080"
cluster_token = "change-me"
llama_dir = "/opt/llama.cpp/build/bin"
margin_pct = 0.10            # leave VRAM for other users
```

```bash
uv run gpupool coordinator --config coordinator.toml
uv run gpupool agent --config agent.toml            # on every server

export GPUPOOL_URL=http://10.0.0.1:8080 GPUPOOL_ADMIN_KEY=change-me-too
uv run gpupool register qwen3b coordinator://qwen2.5-3b-instruct-q4_k_m.gguf --ctx 4096
uv run gpupool plan qwen3b                           # dry-run placement
uv run gpupool status
```

Then point any OpenAI client at `http://10.0.0.1:8080/v1` with `client-key`.

Security: llama.cpp RPC is unencrypted and unauthenticated. Run it only on a trusted internal network.

## Tests

```bash
uv run pytest                 # unit tests
uv run pytest -m real         # needs llama.cpp in .cache/llama/b11342-cuda12.4 and a GGUF in .cache/models
uv run python scripts/e2e_local.py   # 3 emulated nodes on one machine, real llama.cpp
```
