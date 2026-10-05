# gpupool — Test report

> English version. Vietnamese version: [TEST_REPORT.vi.md](TEST_REPORT.vi.md). Keep both in sync.

## 2026-10-05 (engine pass: RPC topology, MTP, KV layout)

Changes: one `ggml-rpc-server` per server and replica, `draft-mtp` speculative decoding, `kv_unified`, per-layer
KV estimate (SWA, MLA, hybrid, on-demand MTP blocks), MoE decode speed, calibration reset on an estimator change,
stable routing key for multi-turn chats, uncapped router pool, body forwarding, cold-start overlap, RPC cache on
`/data` with a size cap, RDMA in the agent image (pull request #5; design in [DESIGN.en.md](DESIGN.en.md)).

### Results

| Check | How | Result |
| --- | --- | --- |
| Unit and API tests | `uv run pytest -q` | 885 passed, 5 skipped |
| Flags against llama.cpp | every flag, KV-layout rule and RPC behaviour read from the b11342 source (`common/arg.cpp`, `llama-hparams.cpp`, `llama-kv-cache*.cpp`, `src/models/*.cpp`, `ggml-rpc.cpp`, `transport.cpp`) | `-kvu`, `--spec-type draft-mtp`, `-d CUDA0,CUDA1` exist; RDMA falls back to TCP; a missing cache file only resends the tensor |
| KV layout | synthetic GGUFs per family (gemma3 SWA, deepseek2 MLA, qwen35 hybrid + MTP, nemotron_h, MoE) | sizes match the llama.cpp formulas, unknown architectures keep the old rule |
| UI | Chromium against `scripts/ui_mock_server.py` | shared-context switch and MTP option render, no page errors |
| Images | CI `agent` / `coordinator` / `e2e` jobs | run on the pull request |

### Still to do

- Run a split model on real GPUs: decode speed with one RPC server for two GPUs of one server versus one per GPU;
  `draft-mtp` acceptance and speed on a Qwen3.5 or GLM GGUF; measured buffers of a SWA and a hybrid model against
  the new estimate.
- RDMA on real InfiniBand / RoCE hardware.

## 2026-10-03 (conversion, round 2: importance matrices, IQ types, early disk check)

Features: IQ1/IQ2/IQ3 quantization types with importance matrices (the *calibrating* stage), the disk check at
submit, `failed_stage` / `imatrix_used` on jobs, per-model allowed servers and GPUs (`"<node>/*"`), copy buttons on
model cards (commit 002d42e, with f3bf76b; design in [DESIGN.en.md](DESIGN.en.md#178-importance-matrices-and-the-calibrate-stage)).
Machine: the same Windows 11 laptop (CPU only for conversion), Docker in WSL2, llama.cpp b11342, coordinator image
built with the toolchain including `llama-imatrix`.

### Results

| Test | Result |
| --- | --- |
| Unit suite (`uv run pytest`) | 809 passed |
| CI end-to-end (`scripts/ci_e2e.py`) | 33/33 checks passed in 249 s |
| E2E: `HuggingFaceTB/SmolLM2-135M-Instruct` to `Q4_K_M` | 71 s, 105,453,984 bytes; served split over RPC, answered a chat |
| E2E: a **folder** source to `Q8_0` | 25 s, 144,810,912 bytes; no importance matrix (`imatrix_used` false), the file is in the library, the source folder untouched |
| E2E: Hugging Face to `IQ2_XS` **with an importance matrix** (4 chunks) | 76 s, 84,573,088 bytes; stages *converting*, *calibrating*, *quantizing*, *validating*; `imatrix_used` true, GGUF header validated, file in the library |

### Real conversions (measured, one job at a time, laptop CPU, WSL Docker)

| Model, type | Download | Convert | Calibrate | Quantize | Validate | Total | Output | Peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct, `Q4_K_M` | 107 s | 12 s | none | 4 s | 20 s | 143 s | 397,807,488 B | 1045 MiB |
| Qwen2.5-1.5B-Instruct, `IQ3_M` (100 chunks) | 314 s | 55 s | 1175 s | 123 s | 21 s | 1687 s (28 min) | 776,663,904 B | 1697 MiB |

The cgroup peaks, which include the page cache, were 3.0 GB and 3.4 GB. Calibration is about 70 % of the second job:
on a CPU the importance matrix is the cost of the IQ types, which is why the UI states it and why *Calibration chunks*
exists (fewer = faster).

### Size estimate for the new types

The estimate shown in the dialog against the real files: `IQ2_XS` +6 %, `IQ3_M` -4 %, `Q8_0` -1 %. The IQ rows use
whole-file bits per weight (the output and the most sensitive tensors stay at higher precision), not the format's
nominal figure.

### Defects found by review and real runs (all fixed)

| Defect | Effect | Fix |
| --- | --- | --- |
| `.gitignore` had the rule `data/` (meant for runtime data folders), which also matched `src/gpupool/converter/data/` | the built-in calibration text and its README were silently left out of the commit; a fresh checkout or the CI image would have no calibration text and every importance-matrix job without a `calibration_path` would be refused with 503. Local tests passed because the file existed on disk | the package data folder is excepted from the rule (commit f3bf76b) |
| The model form said "Auto placement / choose GPUs yourself" | it read as manual placement, although pins were always an *allowed set* the scheduler chooses within | now "All servers and GPUs / Only selected ones" with a server and GPU tree; `"<node>/*"` allows a whole server including later GPUs; the Servers tab is untouched |
| Only the endpoint could be copied from a model card | users had to type the model name and a curl by hand | the card copies the endpoint, the model name and a ready `curl` separately |

Not covered by these runs: a model of several GB or more (the largest was 1.5 B parameters), an importance matrix on a
GPU (calibration always runs on the CPU), a custom `calibration_path` end to end (unit tests only), the
`needs_review` path from a real model, and a real multi-server LAN.

## 2026-10-03 (Hugging Face to GGUF conversion)

Feature: the coordinator converts Hugging Face models to GGUF with a selectable quantization type
(commit 6766ba1, design in [DESIGN.en.md](DESIGN.en.md#17-hugging-face-to-gguf-conversion-converter)). Machine: the
same Windows 11 laptop, Docker in WSL2, llama.cpp b11342, coordinator image built with the conversion toolchain.

### Results

| Test | Result |
| --- | --- |
| Unit suite (`uv run pytest`) | 768 passed |
| CI end-to-end (`scripts/ci_e2e.py`) with the conversion stage | 21/21 checks passed. `HuggingFaceTB/SmolLM2-135M-Instruct` converted to `Q4_K_M` (105,453,984 bytes) in 64 s including the download, served split over RPC, and answered a chat |
| `Qwen/Qwen2.5-0.5B-Instruct` to `Q4_K_M` | 397,807,488 bytes. GGUF header ok, chat template present, tokenizer ids identical to Hugging Face on 8 of 8 cases (English, Vietnamese, code, numbers, emoji ZWJ sequence, spacing, newlines, CJK), generation "The capital of France is Paris..." |
| Folders after the jobs | `.convert` and `.hf` under the models folder empty |
| Coordinator image size | 1.77 GB with the toolchain (`WITH_CONVERT=1`), 560 MB without (`WITH_CONVERT=0`) |

### Size estimate

The estimate shown before converting was 24 % below the real file on Qwen2.5-0.5B, because it used one average
bits-per-weight per type. Costing the embedding matrices and llama-quantize's fallback for rows not divisible by 256
separately (see DESIGN section 17.5) brought it within about 2 %:

| Model, type | Estimated | Real |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct, `Q4_K_M` | 390.7 MB | 397.8 MB |
| SmolLM2-135M-Instruct, `Q4_K_M` | 103.1 MB | 105.5 MB |

### Defects found by review and real runs (all fixed)

| Defect | Effect | Fix |
| --- | --- | --- |
| Mistral repositories ship `consolidated.*` copies of the same weights | the download was doubled | `consolidated.*` files are skipped |
| File names from a repository listing were not filtered | a hostile name such as `../x` could point outside the download folder | unsafe names (absolute, `..`, backslash, drive letter) are dropped when selecting files and checked again before use |
| Size estimate from one average bpw per type | 24 % too low on Qwen2.5-0.5B, so the dialog and the disk check under-reported | embeddings and the K-quant fallback are costed separately; within about 2 % |

Not covered by these runs: models of several GB or more (only 135M and 0.5B parameters were converted), a gated
repository with a token, AWQ / pre-quantized sources, `allow_remote_code`, and a `needs_review` job from a real model
(those paths are covered by unit tests only).

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
