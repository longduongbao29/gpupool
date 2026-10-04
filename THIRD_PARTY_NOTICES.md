# Third-party notices

> English. Tiếng Việt: [THIRD_PARTY_NOTICES.vi.md](THIRD_PARTY_NOTICES.vi.md)

gpupool itself is released under the [MIT License](LICENSE). It ships or pulls in the components below, each under
its own license. This file is informative, not legal advice: the license text that comes with each component is the
authority.

"Verified" means the license was read from the package metadata installed in this repository's environment
(`importlib.metadata`, against `uv.lock`) or from the file itself. "Per upstream" means it is the upstream project's
published license and was not checked against a local copy.

## Models are not covered

**gpupool does not include, license or grant rights to any model.** Model weights (Llama, Qwen, Gemma, Mistral, and
all others) are separate works with their own licenses and acceptable-use policies, set by whoever published them.

- You are responsible for reading and following the license of every model you download, convert or serve.
- Gated models on Hugging Face require you to accept the publisher's terms on huggingface.co with your own account
  before gpupool (using your token) can download them.
- Converting a model to GGUF, or quantizing it, does **not** change its license. The result is a derivative of the
  original weights and stays under the original terms.
- The only model-related data gpupool ships is the built-in calibration text
  (`src/gpupool/converter/data/calibration.txt`). It is original text written for gpupool and is licensed MIT like
  the rest of the project; see `src/gpupool/converter/data/README.md`.

## Vendored in the source tree

| Component | Version | License | Where | Link |
| --- | --- | --- | --- | --- |
| Alpine.js | 3.14.9 | MIT (verified: header of the file) | `src/gpupool/ui/vendor/alpine.min.js` | <https://github.com/alpinejs/alpine> |

## llama.cpp (engine and converter)

| Component | Version | License | Link |
| --- | --- | --- | --- |
| llama.cpp (`llama-server`, `ggml-rpc-server`, `llama-quantize`, `llama-imatrix`, `llama-tokenize`, ggml) | tag `b11342` | MIT (per upstream) | <https://github.com/ggml-org/llama.cpp> |

- The **agent image** (`docker/agent.Dockerfile`) compiles llama.cpp b11342 with CUDA and includes the binaries.
- The **coordinator image** built with `WITH_CONVERT=1` (`docker/coordinator.Dockerfile`) compiles the CPU tools
  `llama-quantize`, `llama-tokenize`, `llama-simple`, `llama-imatrix`, and copies `convert_hf_to_gguf.py`,
  `conversion/` and `gguf-py/` from the same tag into `/opt/llama.cpp`, together with the upstream `LICENSE`
  (copied to `/opt/llama.cpp/LICENSE`).
- gpupool does not modify llama.cpp. The Dockerfiles download or build it unmodified from the pinned tag.

## Python runtime dependencies of gpupool

Direct dependencies (`pyproject.toml`), versions as locked in `uv.lock`. All verified from installed metadata.

| Package | Version | License |
| --- | --- | --- |
| fastapi | 0.142.2 | MIT |
| gguf | 0.19.0 | MIT |
| httpx | 0.28.1 | BSD-3-Clause |
| nvidia-ml-py | 13.615.71 | BSD (as declared by the package) |
| psutil | 7.2.2 | BSD-3-Clause |
| pydantic | 2.13.5 | MIT |
| uvicorn (`standard` extra) | 0.54.0 | BSD-3-Clause |

Notable transitive runtime packages (also verified from metadata):

| Package | Version | License |
| --- | --- | --- |
| starlette | 1.7.0 | BSD-3-Clause |
| pydantic-core | 2.46.5 | MIT |
| anyio | 4.15.1 | MIT |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httptools | 0.8.0 | MIT |
| websockets | 17.1 | BSD-3-Clause |
| watchfiles | 1.3.0 | MIT |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| PyYAML | 6.0.3 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| requests | 2.34.2 | Apache-2.0 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| idna | 3.20 | BSD-3-Clause |
| urllib3 | 2.8.0 | MIT |
| charset-normalizer | 3.5.2 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| annotated-types, typing-inspection, annotated-doc | 0.8.0, 0.4.4, 0.0.5 | MIT |
| opentelemetry-api | 1.45.0 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| colorama (Windows only) | 0.4.6 | BSD-3-Clause (metadata says "BSD License") |
| uvloop (Linux/macOS only, via `uvicorn[standard]`) | resolved at install | MIT OR Apache-2.0 (per upstream; not installed on the Windows machine used to write this file) |

The full, exact set for your platform is in `uv.lock`. Development-only packages (pytest, pytest-asyncio, respx,
pluggy, iniconfig, Pygments) are not shipped in the images or the wheel; they are MIT, Apache-2.0, BSD-3-Clause,
MIT, MIT and BSD-2-Clause respectively (verified).

## Conversion environment (coordinator image with `WITH_CONVERT=1`)

The image contains a separate virtual environment, `/opt/convert-venv`, installed from llama.cpp's
`requirements/requirements-convert_hf_to_gguf.txt` at tag b11342 (CPU wheel of PyTorch). These packages are not
installed in this repository's environment, so their licenses are given per upstream; check the exact set inside the
image with `docker run --rm --entrypoint /opt/convert-venv/bin/python <image> -m pip list` if you need to be sure.

| Package | License (per upstream) |
| --- | --- |
| torch | BSD-3-Clause |
| transformers | Apache-2.0 |
| sentencepiece | Apache-2.0 |
| numpy | BSD-3-Clause (plus bundled components, see above) |
| protobuf | BSD-3-Clause |
| gguf | MIT |

The requirements file may pull further transitive packages (for example huggingface-hub, tokenizers, safetensors,
PyYAML, requests, tqdm); they are permissive licenses (Apache-2.0, MIT, BSD) per upstream. The PyTorch wheel bundles
third-party components with their own notices; see <https://github.com/pytorch/pytorch/blob/main/LICENSE> and
`NOTICE`.

## Tools and base images

| Component | License | Notes |
| --- | --- | --- |
| uv (binary copied from `ghcr.io/astral-sh/uv`, or `pip install uv`) | MIT OR Apache-2.0 (per upstream) | <https://github.com/astral-sh/uv> |
| `python:3.12-slim` (coordinator image base) | Python is under the PSF License; the image also contains Debian packages, each with its own license | <https://docs.python.org/3/license.html>, <https://hub.docker.com/_/python>; per-package texts are in `/usr/share/doc/*/copyright` inside the image |
| `nvidia/cuda:*-devel-ubuntu*` and `nvidia/cuda:*-runtime-ubuntu*` (agent image base, default CUDA 12.8.1 on Ubuntu 22.04) | NVIDIA CUDA EULA and NVIDIA Deep Learning Container license, plus Ubuntu package licenses | see the notice below |
| Python 3.12 (agent image; CPython build downloaded by uv) | PSF License | python-build-standalone, <https://github.com/astral-sh/python-build-standalone> |
| iptables, libgomp1 and other Debian/Ubuntu packages installed in the images | GPL/LGPL and others, per package | texts in `/usr/share/doc/*/copyright` inside the image |

### NVIDIA terms for the agent image

The agent image is built on NVIDIA's CUDA images and contains NVIDIA CUDA runtime libraries. **If you build, publish
or redistribute the agent image, you accept NVIDIA's terms for those components.** They are governed by the
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html) and the
[NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license).
Read them before redistributing the image outside your organization. The MIT license of gpupool does not extend to
NVIDIA's components, and the NVIDIA GPU driver on the host is not part of gpupool.

## Updating this file

When you add or upgrade a dependency, a base image or the pinned llama.cpp tag, update both language versions of
this file in the same change.
