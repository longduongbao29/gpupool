# gpupool — Test report

> English version. Vietnamese version: [TEST_REPORT.vi.md](TEST_REPORT.vi.md). Keep both in sync.

## 2026-10-03

Machine: the same Windows 11 laptop (GTX 1650 Ti Max-Q, 4 GB), Docker in WSL2. Real llama.cpp b11342 throughout.

### Results

| Test | Result |
| --- | --- |
| Unit suite (`uv run pytest`) | 609 passed, 5 deselected |
| CI end-to-end (`scripts/ci_e2e.py`, `docker-compose.ci.yml`) | 14/14 checks passed locally in about 21 s |
| GitHub Actions run 37082433728 on master | all jobs passed, including `e2e` (3m24s) |
| Simulated 3-server cluster (`docker-compose.sim.yml`) | RPC split with firewall on, VRAM calibration persisted, crash recovery: see below |

The CI end-to-end test runs a coordinator and 3 CPU-only agents (budget 120 MB each, so the model
cannot fit one server), serves SmolLM2-135M-Instruct Q8_0 split over RPC with the RPC firewall on,
and runs in the GitHub Actions job `e2e`.

### Simulated 3-server cluster

Setup: coordinator plus server-a (the real GTX 1650 as NVML reports it, budget 1300 MB), server-b
(simulated Tesla T4 16 GB, budget 1100 MB) and server-c (simulated A100 40 GB, budget 1100 MB). All
three share the one real GPU; every agent has the RPC firewall on.

- **RPC with firewall:** Qwen2.5-3B Q4 split with the head on server-a and RPC on server-c; chat works
  through RPC. On server-c, iptables allow 127.0.0.1, the head (172.30.0.11) and itself (172.30.0.13),
  then DROP; server-b is blocked from the RPC port. Only the head holds the 2.1 GB GGUF; server-c held
  only a ~724 MB RPC tensor cache.
- **VRAM self-calibration:** measured buffers were 1.5 % above the estimate, so the factor became 1.015
  and was still in effect after a coordinator restart.
- **Crash recovery:** `docker kill` of the coordinator mid-launch. The orphaned "launching" replica was
  failed ("coordinator restarted during launch") within about 5 s and a new replica was ready within about 30 s,
  with no leftover engines.
- **Speed (Qwen2.5-3B, GTX 1650, also listed in the quick start):** KV cache at ctx 8192 saves 132 MB
  (`q8_0`) and 204 MB (`q4_0`) with decode 51.9 / 51.2 / 50.8 tok/s (f16 / q8_0 / q4_0). Split over 2
  servers: no speculation 48.9, ngram 53.6, draft 0.5B 53.9 tok/s.

### Defects found by real runs (all fixed, each with a regression test)

This round:

| Defect | Effect | Fix |
| --- | --- | --- |
| RPC firewall blocked the agent's own readiness probe (the probe connects from the agent's bind address) | engine hung in "launching" for 600 s | the agent's own address is allowed |
| Launch orphaned by a hard kill of the coordinator | replica stuck in "launching" forever and blocked relaunch | interrupted launches are failed at startup |
| Flaky simulate-purity test compared the mock's live busy ratio | random test failures | the test ignores the mock's live load |

Earlier rounds:

| Defect | Fix |
| --- | --- |
| Draft-model reservation reduced every GPU instead of only the head | reserve only on the head |
| `-md` alone loads the draft but never uses it in b11342 | `--spec-type` is set explicitly |
| Multi-node placement offered only one candidate | subsets of servers are enumerated, capped at 12 |
| Agent image used Python 3.14 because `.python-version` was not copied | file copied, Python 3.12 pinned |
| `budget_mb` did not count the model's own replicas | own replicas are counted |

### Environment notes

WSL idle shutdown restarts the containers (gpupool self-healed). Fix: `vmIdleTimeout=-1` in `.wslconfig`.

### Still to do

- Run on a real multi-server LAN.
- Measure speculative decoding over a real network.
- Models of 7B and larger need bigger GPUs than the test laptop has.
- Per-client API keys and quotas, `/v1/embeddings`, TLS.
- Zero-downtime model swap.
- Upgrade llama.cpp.

## 2026-10-02 (history)

Machine: one laptop, GTX 1650 Ti Max-Q 4 GB, 16 GB RAM, Windows 11, llama.cpp b11342 (CUDA 12.4 build).
Three "servers" emulated by three agents on 127.0.0.1 / .2 / .3 (`scripts/e2e_local.py`):
a = real CUDA0 capped at 1200 MB; b, c = CPU devices (1000 MB each) behind a real `ggml-rpc-server`.

### Results

| Test | Result |
| --- | --- |
| Unit tests (`uv run pytest`) | 90 passed |
| Real-binary tests (`uv run pytest -m real`) | 3 passed |
| Scenario 1: 0.5B on one GPU | passed |
| Scenario 2: 3B split over 3 nodes via RPC | passed in one complete run; the clean full rerun was stopped by the host for low memory |
| Scenario 3: failover | **not run yet** (stopped for low memory before reaching it) |

### Numbers

| Setup | Prefill | Decode | TTFT via router | Load time |
| --- | --- | --- | --- | --- |
| 0.5B, llama-bench, 1 GPU (baseline) | 1292 t/s (pp512) | 182 t/s | — | — |
| 0.5B through gpupool, 1 GPU | 177 t/s (short prompt) | 185 t/s (170 measured at the client) | 76 ms | 5.2 s |
| 3B, llama-bench, 1 GPU (baseline) | 250 t/s (pp512) | 51.4 t/s | — | — |
| 3B through gpupool, split GPU + 2 CPU nodes over RPC | 13 t/s | 11.6 t/s | 7.7 s | 16.2 s |

- The router adds no measurable decode cost (185 vs 182 t/s baseline).
- The 3B split is 4.4x slower than one GPU because 23 of its 36 layers run on CPU devices in this
  emulation. On real servers the remote devices are GPUs; this run validates correctness of the
  RPC path, not speed. The answer was correct ("Paris").
- Plan for the 3B: a/CUDA0 13 layers (842 MB), b/CPU 12 layers (683 MB), c/CPU 11 layers + output (899 MB).

### Memory estimate vs reality

| Model | Estimate (before) | Estimate (calibrated) | Measured VRAM |
| --- | --- | --- | --- |
| 0.5B, ctx 4096 | 722 MB (+38%) | 587 MB (+12%) | 525 MB |

Weights and KV cache match llama.cpp's load log to the MiB. The old flat 300 MB overhead was replaced by
a measured compute-buffer formula plus a 128 MB context budget.

### Defects found by real runs (all fixed, each with a regression test)

| Defect | Effect | Fix |
| --- | --- | --- |
| `--device` passed before `--rpc` | every multi-node launch exited with a usage error | `--rpc` first; a real-parser test runs the generated command |
| Flat 300 MB overhead per device | 38% overestimate, wasted pool capacity | calibrated estimate, pinned by a test |
| Hard-killed agent left `ggml-rpc-server` running | orphan holds memory, unknown to the coordinator | pid files + reaping on agent start |
| Earlier review: CPU RAM counted like VRAM | layers in host RAM while another node had free VRAM | GPU-only pass first |
| Earlier review: ready replica's VRAM released too early | the same VRAM could be planned twice | reservation held until a fresh report |

### Still to do

- Run scenario 3 (failover): `uv run python scripts/e2e_local.py --only 3`. Needs ~4 GB free RAM.
- Test on real Linux servers with GPUs on every node (speed numbers above are not representative of that).
