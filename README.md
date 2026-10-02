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

One command for the coordinator, one per GPU server, then point any OpenAI client at it.
Full guide: [docs/QUICKSTART.en.md](docs/QUICKSTART.en.md).

```bash
# 1. coordinator (any machine): the log prints the admin key; open http://<its IP>:8080
docker run -d --name gpupool -p 8080:8080 -v gpupool:/data ghcr.io/longduongbao29/gpupool-coordinator
docker logs gpupool

# 2. each GPU server: run the join command shown in the UI (Servers -> Add Server)
docker run -d --name gpupool-agent --gpus all --network host --pid host -v gpupool-agent:/data \
  -e GPUPOOL_JOIN="http://10.0.0.1:8080#<cluster-token>" ghcr.io/longduongbao29/gpupool-agent

# 3. in the UI: Models -> Add model (Hugging Face or path) -> New model -> Start, then:
curl http://10.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen7b", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Tests

```bash
uv run pytest                 # unit tests
uv run pytest -m real         # needs llama.cpp in .cache/llama/b11342-cuda12.4 and a GGUF in .cache/models
uv run python scripts/e2e_local.py   # 3 emulated nodes on one machine, real llama.cpp
```
