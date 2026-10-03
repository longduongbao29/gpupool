# gpupool coordinator: web UI + admin API + scheduler + OpenAI-compatible router. No GPU needed.
#
#   docker run -d --name gpupool -p 8080:8080 -v gpupool:/data ghcr.io/longduongbao29/gpupool-coordinator
#   docker logs gpupool    # UI URL, admin key and the join command for GPU servers
#
# The admin key and cluster token are generated on first start and kept in /data/secrets.json.
# Set GPUPOOL_ADMIN_KEY / GPUPOOL_CLUSTER_TOKEN to choose them, and GPUPOOL_PUBLIC_URL if the
# address shown in the log is not reachable from your servers. Mount existing GGUF files with
# -v /srv/gguf:/models:ro: the UI browses /models (GPUPOOL_MODEL_ROOTS). To also accept host paths
# typed in the UI, set GPUPOOL_PATH_MAP (see docker-compose.coordinator.yml).
# /data holds the database, secrets and models downloaded from Hugging Face.
#
# Behind a proxy: Docker forwards the predefined build args to every RUN step (pip, uv downloads),
# so pass them at build time:
#   docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
#     --build-arg no_proxy=$no_proxy -f docker/coordinator.Dockerfile -t gpupool-coordinator .
# They are deliberately NOT declared with ARG/ENV here: predefined proxy args are excluded from
# the image history and never baked into the final image (they are site secrets). Pulling the
# base images and the uv image is done by the Docker daemon: configure the daemon proxy for that.
# At run time pass them through from the host: -e http_proxy -e https_proxy -e no_proxy
#
# Where ghcr.io / GitHub is blocked (see vendor/README.md): the base build never touches GitHub
# (UV_PYTHON_DOWNLOADS=never, Python comes from the base image). The uv binary comes from
#   --build-arg UV_IMAGE=registry.corp/astral-sh/uv:0.10   (a registry mirror), or
#   --build-arg UV_FROM_PYPI=1                              (pip install uv; PyPI usually works via the proxy)
#
# Hugging Face -> GGUF conversion toolchain (--build-arg WITH_CONVERT=1, the default). It lets the
# UI/API convert safetensors models to GGUF and quantize them. It adds, on top of the lean image:
#   /opt/llama.cpp       convert_hf_to_gguf.py + conversion/ + gguf-py/ from llama.cpp LLAMA_CPP_REF
#   /opt/llama/bin       llama-quantize, llama-tokenize, llama-simple (static CPU build, no GPU needed)
#   /opt/convert-venv    Python 3.12 venv with CPU-only PyTorch and transformers (the bulk of the size)
# and sets GPUPOOL_CONVERT_DIR / GPUPOOL_CONVERT_PYTHON / GPUPOOL_LLAMA_TOOLS_DIR. Size impact
# (measured): 560 MB -> 1.77 GB on disk, of which 900 MB is the venv (torch) and 21 MB the llama.cpp
# tools; a cold build takes about 7 minutes (-j2) longer than the lean one.
# --build-arg WITH_CONVERT=0 builds the lean image (no conversion); BuildKit then skips the
# toolchain stages entirely. The toolchain build needs three things from the network:
#   llama.cpp source, tried in this order: vendor/llama.cpp-<LLAMA_CPP_REF>.tar.gz (no network),
#     git clone from GitHub, curl of LLAMA_CPP_URL (an internal mirror of the GitHub tag archive).
#     Where GitHub is blocked, drop the tarball into vendor/ (see vendor/README.md).
#   PyPI packages: the usual PIP_INDEX_URL / UV_INDEX_URL settings and the proxy build args apply.
#   PyTorch CPU wheels: https://download.pytorch.org/whl/cpu by default; behind a mirror use
#     --build-arg TORCH_INDEX_URL=https://mirror.corp/pytorch/whl/cpu

# uv binary source. UV_FROM_PYPI selects the stage name below: empty -> uv-src, 1 -> uv-src1.
# BuildKit builds only the stage that is used, so the unused source is never pulled.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10
ARG UV_FROM_PYPI=
ARG UV_PYPI_SPEC="uv>=0.10,<0.11"
# 1 = ship the HF -> GGUF conversion toolchain, 0 = lean image. Selects the stage `final-0` or
# `final-1` below, so BuildKit builds only the stages that are used.
ARG WITH_CONVERT=1

FROM ${UV_IMAGE} AS uv-src

FROM docker.io/library/python:3.12-slim AS uv-src1
ARG UV_PYPI_SPEC
RUN pip install --no-cache-dir "${UV_PYPI_SPEC}" && cp "$(command -v uv)" /uv

FROM uv-src${UV_FROM_PYPI} AS uv

FROM docker.io/library/python:3.12-slim AS base
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# ---- conversion toolchain (WITH_CONVERT=1 only) ----------------------------------------------------
# Built in a throw-away stage so no compiler or cache reaches the final image. Same base image as
# `base`, so the venv's python symlink (/usr/local/bin/python3.12) resolves in the final image too.
FROM docker.io/library/python:3.12-slim AS convert-build
# Pinned: gpupool was tested on b11342 (same tag as docker/agent.Dockerfile).
ARG LLAMA_CPP_REF=b11342
# Last-resort source download, used when there is no vendored tarball and git cannot reach GitHub.
# Must serve the GitHub tag archive layout (one top-level directory, stripped on extract).
ARG LLAMA_CPP_URL=https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_CPP_REF}.tar.gz
# Parallel compile jobs. Low by default: small CI runners and WSL have little RAM.
ARG LLAMA_BUILD_JOBS=2
# PyTorch CPU wheel index (the CPU build is ~200 MB; the default CUDA one is several GB).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
# Fetch the source: vendored tarball -> git clone -> curl (same logic as docker/agent.Dockerfile).
# The vendor/ bind mount is read-only and never becomes a layer. A failed clone must not leave a
# half-done /src behind.
RUN --mount=type=bind,source=vendor,target=/vendor \
    set -eu; \
    tarball="/vendor/llama.cpp-${LLAMA_CPP_REF}.tar.gz"; \
    extract() { mkdir -p /src && tar -xzf "$1" -C /src --strip-components=1; }; \
    header_commit() { gzip -dc "$1" 2>/dev/null | head -c 1024 | grep -a -o -E 'comment=[0-9a-f]{40}' | head -n1 | cut -c9-15 || true; }; \
    commit=""; \
    if [ -f "$tarball" ]; then \
        echo "llama.cpp: using vendored $tarball"; \
        extract "$tarball"; commit="$(header_commit "$tarball")"; \
    elif GIT_TERMINAL_PROMPT=0 git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 \
            clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp /src; then \
        commit="$(git -C /src rev-parse --short=7 HEAD)"; \
    else \
        echo "llama.cpp: git clone failed, downloading $LLAMA_CPP_URL" >&2; \
        rm -rf /src; \
        curl -fsSL --retry 3 --connect-timeout 20 -o /tmp/llama.tar.gz "$LLAMA_CPP_URL"; \
        extract /tmp/llama.tar.gz; commit="$(header_commit /tmp/llama.tar.gz)"; rm -f /tmp/llama.tar.gz; \
    fi; \
    test -f /src/CMakeLists.txt || { echo "llama.cpp source is empty or has the wrong layout" >&2; exit 1; }; \
    echo "$commit" > /src/.gpupool-commit
WORKDIR /src
# Portable CPU build: GGML_NATIVE=OFF (no -march=native, runs on any x86-64 / arm64 host), static
# libraries (BUILD_SHARED_LIBS=OFF: the tools are self-contained apart from libgomp), no libcurl /
# OpenSSL (models come from gpupool, not llama.cpp's downloader), no server, no tests. Tools and
# examples are switched on only because the three targets live there; only those are built.
# The build number comes from the tag because a tarball has no .git (see docker/agent.Dockerfile).
RUN num="${LLAMA_CPP_REF#b}"; \
    case "$num" in ''|*[!0-9]*) num=0 ;; esac; \
    commit="$(cat /src/.gpupool-commit)"; \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_NUMBER="$num" -DLLAMA_BUILD_COMMIT="${commit:-unknown}" \
    && cmake --build build --config Release -j"${LLAMA_BUILD_JOBS}" \
        --target llama-quantize llama-tokenize llama-simple \
    && mkdir -p /out/llama/bin /out/llama.cpp \
    && cp build/bin/llama-quantize build/bin/llama-tokenize build/bin/llama-simple /out/llama/bin/ \
    && strip /out/llama/bin/* \
    && cp -r convert_hf_to_gguf.py conversion gguf-py /out/llama.cpp/ \
    && cp LICENSE /out/llama.cpp/LICENSE \
    && find /out/llama.cpp -name __pycache__ -type d -prune -exec rm -rf {} + \
    && rm -rf /out/llama.cpp/gguf-py/tests
# Python deps of the converter. The requirements file pins torch==2.11.0 and carries its own
# --extra-index-url (download.pytorch.org); that line is dropped from a copy so TORCH_INDEX_URL
# decides where torch comes from. unsafe-best-match lets uv pick each package from whichever index
# has a matching version (torch only on the torch index, the rest on PyPI or its mirror).
# The venv is created from the image's own python (UV_PYTHON_DOWNLOADS=never: no GitHub download).
RUN mkdir /req && cp /src/requirements/requirements-convert_hf_to_gguf.txt \
        /src/requirements/requirements-convert_legacy_llama.txt /req/ \
    && sed -i '/^--extra-index-url/d' /req/requirements-convert_hf_to_gguf.txt \
    && UV_PYTHON_DOWNLOADS=never uv venv --python /usr/local/bin/python3.12 /out/convert-venv \
    && UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy uv pip install --python /out/convert-venv/bin/python \
        --no-cache --index-strategy unsafe-best-match --extra-index-url "${TORCH_INDEX_URL}" \
        -r /req/requirements-convert_hf_to_gguf.txt \
    && find /out/convert-venv -name __pycache__ -type d -prune -exec rm -rf {} +

# ---- final image variants: final-0 = lean, final-1 = with the toolchain ------------------------------
FROM base AS final-0

FROM base AS final-1
# libgomp1: OpenMP runtime of the CPU build (llama-quantize is multi-threaded).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=convert-build /out/llama /opt/llama
COPY --from=convert-build /out/llama.cpp /opt/llama.cpp
COPY --from=convert-build /out/convert-venv /opt/convert-venv
ENV GPUPOOL_CONVERT_DIR=/opt/llama.cpp \
    GPUPOOL_CONVERT_PYTHON=/opt/convert-venv/bin/python \
    GPUPOOL_LLAMA_TOOLS_DIR=/opt/llama/bin

FROM final-${WITH_CONVERT}
ENV GPUPOOL_HOST=0.0.0.0 \
    GPUPOOL_PORT=8080 \
    GPUPOOL_DB_PATH=/data/coordinator.db \
    GPUPOOL_MODELS_DIR=/data/models \
    GPUPOOL_MODEL_ROOTS=/models
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)" || exit 1
ENTRYPOINT ["gpupool", "coordinator"]
