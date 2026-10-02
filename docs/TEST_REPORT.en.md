# gpupool — Test report (2026-10-02)

> English version. Vietnamese version: [TEST_REPORT.vi.md](TEST_REPORT.vi.md). Keep both in sync.

Machine: one laptop, GTX 1650 Ti Max-Q 4 GB, 16 GB RAM, Windows 11, llama.cpp b11342 (CUDA 12.4 build).
Three "servers" emulated by three agents on 127.0.0.1 / .2 / .3 (`scripts/e2e_local.py`):
a = real CUDA0 capped at 1200 MB; b, c = CPU devices (1000 MB each) behind a real `ggml-rpc-server`.

## Results

| Test | Result |
| --- | --- |
| Unit tests (`uv run pytest`) | 90 passed |
| Real-binary tests (`uv run pytest -m real`) | 3 passed |
| Scenario 1: 0.5B on one GPU | passed |
| Scenario 2: 3B split over 3 nodes via RPC | passed in one complete run; the clean full rerun was stopped by the host for low memory |
| Scenario 3: failover | **not run yet** (stopped for low memory before reaching it) |

## Numbers

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

## Memory estimate vs reality

| Model | Estimate (before) | Estimate (calibrated) | Measured VRAM |
| --- | --- | --- | --- |
| 0.5B, ctx 4096 | 722 MB (+38%) | 587 MB (+12%) | 525 MB |

Weights and KV cache match llama.cpp's load log to the MiB. The old flat 300 MB overhead was replaced by
a measured compute-buffer formula plus a 128 MB context budget.

## Defects found by real runs (all fixed, each with a regression test)

| Defect | Effect | Fix |
| --- | --- | --- |
| `--device` passed before `--rpc` | every multi-node launch exited with a usage error | `--rpc` first; a real-parser test runs the generated command |
| Flat 300 MB overhead per device | 38% overestimate, wasted pool capacity | calibrated estimate, pinned by a test |
| Hard-killed agent left `ggml-rpc-server` running | orphan holds memory, unknown to the coordinator | pid files + reaping on agent start |
| Earlier review: CPU RAM counted like VRAM | layers in host RAM while another node had free VRAM | GPU-only pass first |
| Earlier review: ready replica's VRAM released too early | the same VRAM could be planned twice | reservation held until a fresh report |

## Still to do

- Run scenario 3 (failover): `uv run python scripts/e2e_local.py --only 3`. Needs ~4 GB free RAM.
- Test on real Linux servers with GPUs on every node (speed numbers above are not representative of that).
