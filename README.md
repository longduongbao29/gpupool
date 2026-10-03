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

## Features

- **Multi-model, multi-server, multi-GPU.** Several models run at once; each one is placed on one GPU,
  one server, or split across servers over llama.cpp RPC.
- **Placement tiers and scoring.** Candidates are ranked by estimated decode speed (memory bandwidth,
  network hops, GPU sharing); the winning placement is stored with its reasons and shown in the UI.
- **Recommendation, capacity and what-if.** `/api/recommend` ranks GPU/server options for a model,
  `/api/capacity` shows what still fits, `/api/simulate` answers "what would happen if" without changing anything.
- **Priority and preemption.** A higher-priority model can take room from lower-priority ones.
- **Autoscaling, including scale-to-zero.** Replicas follow load; an idle model can drop to zero and
  cold-start on the next request.
- **Make-before-break rebalancing.** When a better placement exists, the new replica is ready before the old one stops.
- **KV cache quantization** (`f16`, `q8_0`, `q4_0`) to fit a model on fewer GPUs.
- **Speculative decoding** (`ngram`, or a `draft` model) to cut RPC round trips when a model is split.
- **RPC firewall.** Each agent restricts its `ggml-rpc-server` port (iptables) to the head and itself.
- **VRAM self-calibration.** Measured engine buffers correct the memory estimate; the factor is persisted.
- **Control state survives coordinator restarts** (SQLite), including recovery of launches interrupted by a crash.
- **Deployment.** Docker images on GHCR (`ghcr.io/longduongbao29/gpupool-coordinator`, `ghcr.io/longduongbao29/gpupool-agent`),
  images that build without GitHub access, HTTP proxy support, model files taken from host paths.
- **Convert Hugging Face models to GGUF.** A model published only as safetensors / PyTorch weights can be
  converted inside the coordinator (UI or `/api/convert`) with a selectable quantization type (Q8_0 ... Q2_K),
  size and VRAM estimates per type, and validation (GGUF header, tokenizer ids vs Hugging Face, a short CPU
  generation) before it enters the model library. The Docker image ships the toolchain
  (`WITH_CONVERT=0` builds the lean image without it).
- **Web UI** for servers, GPUs, models, deployments, recommendations and events.

## Status

Tested on one Windows laptop (GTX 1650 Ti, 4 GB): natively with emulated nodes against real llama.cpp,
and as a simulated 3-server Docker cluster in WSL2 (`docker-compose.sim.yml`). A CPU-only 3-server
end-to-end test also runs in CI, including a Hugging Face to GGUF conversion. Not yet run on a real multi-server LAN. See
[docs/TEST_REPORT.en.md](docs/TEST_REPORT.en.md).

## How it works

```
client ──OpenAI API──► coordinator (router + scheduler + reconciler, SQLite)
                            ▲ heartbeats            │ start/stop engines
   server A: agent ─ llama-server (head) ──RPC──► server B: agent ─ ggml-rpc-server
                                         └─RPC──► server C: agent ─ ggml-rpc-server
```

- **agent** (one per server): reports GPUs via NVML, starts/stops llama.cpp processes, caches GGUF files, applies the RPC firewall.
- **scheduler**: reads the GGUF header (no full download), estimates memory per layer, builds candidate
  placements (one GPU, one server, subsets of servers; GPUs before CPU RAM) and scores them by estimated decode speed.
- **router**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming, prefix-aware
  load balancing (shared system prompts hit the replica that has them cached), retry before the first byte.
- **reconciler**: keeps the desired replica count, rolls back failed launches, fails over dead nodes,
  drains, autoscales, preempts and rebalances.

Design: [docs/DESIGN.en.md](docs/DESIGN.en.md), [docs/PLATFORM_DESIGN.en.md](docs/PLATFORM_DESIGN.en.md).

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
uv run pytest                 # unit tests (about 768)
uv run pytest -m real         # needs llama.cpp in .cache/llama/b11342-cuda12.4 and a GGUF in .cache/models
uv run python scripts/e2e_local.py   # 3 emulated nodes on one machine, real llama.cpp

# simulated 3-server cluster on one machine (needs Docker + NVIDIA Container Toolkit)
docker compose -f docker-compose.sim.yml up -d

# CI end-to-end: coordinator + 3 CPU-only agents (docker-compose.ci.yml), tiny model split over RPC,
# then a Hugging Face model is converted to GGUF in the coordinator and served
uv run python scripts/ci_e2e.py [--project gpupool-ci] [--port 8080] [--keep] [--skip-convert]
```

`scripts/ci_e2e.py` needs the `gpupool-agent` and `gpupool-coordinator` images (override with
`GPUPOOL_AGENT_IMAGE` / `GPUPOOL_COORDINATOR_IMAGE`); in GitHub Actions it is the `e2e` job. The conversion
stage needs a coordinator image built with the toolchain (the default) and internet access to
huggingface.co; `--skip-convert` leaves it out.

## Documentation

- [Quick start and deployment](docs/QUICKSTART.en.md)
- [HTTP API](docs/API.en.md)
- [Design](docs/DESIGN.en.md)
- [Platform design (multi-model scheduling)](docs/PLATFORM_DESIGN.en.md)
- [UI design](docs/UI_DESIGN.en.md)
- [Test report](docs/TEST_REPORT.en.md)
