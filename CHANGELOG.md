# Changelog

> English. Tiếng Việt: [CHANGELOG.vi.md](CHANGELOG.vi.md)

All notable changes to gpupool are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (0.x: minor versions may change behaviour).

## [Unreleased]

## [0.4.2] - 2026-10-05

### Fixed

- The "In pool" switches on the Servers and GPUs pages did nothing: the model form's per-model GPU selection
  defined a second `toggleGpu` in the UI component, which replaced the one that calls the API, so a click only
  changed the (hidden) model form. The model form's handlers are now `pinToggleServer` / `pinToggleGpu`, and a test
  fails on any duplicate method name in the component.

## [0.4.1] - 2026-10-04

### Added

- Blackwell GPUs (RTX 5090/5080, RTX PRO 6000): the agent image builds kernels for compute capability 120 (llama.cpp
  turns it into 120a for the FP4 tensor cores). The image build now fails if any requested architecture is missing
  from the CUDA backend, instead of a server failing later with "no kernel image is available".

### Changed

- The agent image is built on CUDA 12.8.1 (was 12.4.1), the first toolkit that targets Blackwell. It still runs on
  drivers >= 525; RTX 50-series cards need a driver >= 570.

### Fixed

- Stopping a model split over RPC crashed its head: every engine was stopped at once, `ggml-rpc-server` exits
  immediately, and `llama-server` then failed to free its remote buffers and aborted (SIGABRT, a core dump of
  0.1-0.5 GB each time, which WSL kept in `%TEMP%\wsl-crashes`). Heads are now stopped first and awaited, the stop
  request waits past the agent's grace period, and engines run without core dumps.

## [0.4.0] - 2026-10-03

### Added

- **Hugging Face to GGUF conversion in the coordinator**: download (or a server folder), `convert_hf_to_gguf.py`,
  `llama-quantize`, validation, then the model library; from the UI or the `/api/convert*` routes. Includes an
  inspect step that reads the config without downloading weights and lists every quantization type with estimated
  size, VRAM and whether it fits one GPU or the pool, jobs with stages, logs, cancel/retry/accept/delete, cached
  downloads, and requeue of interrupted jobs after a restart.
- **Validation gate** before a converted file enters the library: GGUF header, tokenizer ids compared with Hugging
  Face on 8 texts, and a short CPU generation. A mismatch parks the job in `needs_review`.
- **Importance matrices and 10 IQ1/IQ2/IQ3 quantization types**: a "calibrating" stage runs `llama-imatrix`
  (mode auto/on/off). A built-in, original multilingual calibration text is included; a custom `.txt` (up to 20 MB)
  can be used instead.
- **Per-model server/GPU limits**: `pin_devices` accepts `<node>/*`; the UI shows an "All servers and GPUs | Only
  selected ones" tree.
- **Copy buttons** on model cards for the endpoint, the model name and a ready-to-run `curl`.
- **RPC firewall** (`GPUPOOL_RPC_FIREWALL=1`, needs root and `NET_ADMIN`): each RPC engine port accepts only its head
  node, loopback and the agent's own address.
- **VRAM self-calibration**: measured engine buffers correct the memory estimate with a persisted per-model factor.
- **Persisted control state**: autoscaler state, preemption claims and cooldowns, backoff and an in-progress
  rebalance move survive a coordinator restart.
- **CI end-to-end test** (`scripts/ci_e2e.py`): real images as a coordinator plus 3 CPU agents serving a small model
  split over RPC, plus a conversion stage.
- Documentation: conversion, importance matrices, per-model limits, test reports (EN + VI). Licensing and community
  files: `LICENSE` (MIT), third-party notices, contributing guide, code of conduct, security policy, issue and pull
  request templates.

### Changed

- The repository is renamed from `multi-gpu-inference` to `gpupool` (old URLs redirect); image names are unchanged.
- The coordinator image can ship the conversion toolchain (`WITH_CONVERT=1`, about +1.2 GB); `WITH_CONVERT=0`
  keeps the lean image.
- Submitting a conversion is refused with 507 when the disk obviously cannot hold it, instead of failing minutes
  later.
- Size estimates for quantization types treat embeddings and `llama-quantize` fallbacks separately (the plain
  average was 24 % low on a small model; now within a few percent of real files).

### Fixed

- Docker images failed to build once `pyproject.toml` declared its license files: the Dockerfiles did not copy
  `LICENSE` and `THIRD_PARTY_NOTICES.md` into the build stage.
- CI was red since the conversion feature landed: the conversion job manager did not create the coordinator's data
  folder before opening its database, so a fresh checkout or a new `db_path` failed. It is now created, with a
  regression test.
- The built-in calibration text was excluded from a commit by the `data/` ignore rule; a fresh checkout would have
  refused every importance-matrix job. The package data folder is now excepted.
- Calibration divides the planning factor out of the estimate before sampling.
- A flaky agent test that raced a fake engine's log writes now waits; the simulate purity test ignores the mock's
  live load.

### Security

- Conversion never downloads or runs a repository's Python files unless `allow_remote_code` is set, runs the
  converter offline on a staging folder of whitelisted links, and deletes files only under `models_dir/.convert` and
  `.hf`.

## [0.3.0] - 2026-10-03

### Added

- **Multi-model platform**, phases 1 to 4 of the platform design: priority, spread and scored placement with
  `/api/recommend` and `/api/capacity`; autoscaling with scale-to-zero and cold start; priority preemption with
  `/api/simulate`; make-before-break rebalancing with `/api/rebalance`.
- KV-cache quantization (`f16`, `q8_0`, `q4_0`) and speculative decoding (`ngram` or a `draft` model) per model.
- A simulated 3-server cluster (`docker-compose.sim.yml`) for trying the system on one machine.
- Docker builds that work where git or GitHub is blocked (vendored llama.cpp tarball, mirrors, proxy and path
  documentation); model files found by host path when the coordinator runs in Docker.
- GPU identity by UUID; a cap on router request bodies.
- Redesigned UI: design tokens, light theme, motion, favicon, server file picker.

### Changed

- Multi-node placements consider every feasible server combination, not only the greedy one; the scorer picks.
- Both images pin Python 3.12 (`.python-version` is now copied; uv had picked 3.14).
- Coordinator internals cleaned up: shared helpers, typed state, no string-matched errors.

### Fixed

- GPUs are kept in multi-node placements; coordinator stalls are no longer mistaken for dead nodes.
- GPU loss, split GGUF files, port and replica leaks and crash-loop relaunching.
- A literal `\n` in the coordinator Dockerfile `ENV` block that made `docker build` fail.
- Agent cache and engine API hardened; router and `/report` made cheaper; starting a model no longer blocks on a
  tick.

## [0.2.0] - 2026-10-02

### Added

- Management UI, a server registry with polling, GPU switches and an events log.
- Failure notifications.
- One-command Docker setup (generated secrets, self-joining agents, quick start), env-var configuration and CI image
  builds.
- Model library: Hugging Face downloads and registered local paths.
- GPU telemetry for the UI; a lost GPU no longer hides healthy ones.
- Proxy support: external traffic through `http_proxy`/`https_proxy`, internal traffic never.

## Earlier (before 0.2.0, not tagged)

- Control plane built in layers: shared contracts (wire models, auth, config), the node agent (device probe, engine
  supervisor, model cache, HTTP API), the scheduler (streaming GGUF header parser, memory estimate, placement), the
  router (prefix-aware balancer, OpenAI-compatible proxy) and the coordinator (store, agent client, reconciler,
  admin API).
- Memory estimate calibrated against real llama.cpp buffers; end-to-end script; multi-node launch passes `--rpc`
  before `--device`; the agent reaps orphaned engines.
- Documentation in English and Vietnamese: README, design, test report.

[Unreleased]: https://github.com/longduongbao29/gpupool/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/longduongbao29/gpupool/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/longduongbao29/gpupool/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/longduongbao29/gpupool/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/longduongbao29/gpupool/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/longduongbao29/gpupool/releases/tag/v0.2.0
