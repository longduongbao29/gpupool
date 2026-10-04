# gpupool agent: CUDA runtime + llama.cpp (CUDA + RPC) + the gpupool agent.
#
# Run on each GPU server (needs the NVIDIA Container Toolkit):
#   docker run -d --name gpupool-agent --gpus all --network host --pid host -v gpupool-agent:/data \
#     -e GPUPOOL_JOIN=http://10.0.0.1:8080#<cluster_token> ghcr.io/longduongbao29/gpupool-agent
#
# GPUPOOL_JOIN (printed in the coordinator log) is all that is needed: the node id defaults to
# the hostname, the IP is detected, and the agent registers itself with the coordinator.
# Override with GPUPOOL_NODE_ID / GPUPOOL_HOST if the defaults are wrong.
#
# --network host: engines open RPC/HTTP ports chosen by the coordinator and must be reachable
#   on the server's own IP. --pid host: NVML reports host PIDs; without it GPU process names
#   would be looked up in the container's PID namespace and come out wrong.
#
# CUDA 12.8 is the first toolkit that builds for Blackwell (RTX 50-series, sm_120). It runs on
# drivers >= 525 through CUDA minor-version compatibility; RTX 50-series cards themselves need a
# driver >= 570. For older drivers rebuild with a lower CUDA_VERSION (and without 120 in CUDA_ARCHS).
#
# Behind a proxy: Docker forwards the predefined build args to every RUN step (apt-get, git clone,
# curl, uv downloads), so pass them at build time:
#   docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
#     --build-arg no_proxy=$no_proxy -f docker/agent.Dockerfile -t gpupool-agent .
# They are deliberately NOT declared with ARG/ENV here: predefined proxy args are excluded from
# the image history and never baked into the final image (they are site secrets). Pulling the
# base images and the uv image is done by the Docker daemon: configure the daemon proxy for that.
# At run time pass them through from the host: -e http_proxy -e https_proxy -e no_proxy
#
# Where git / GitHub is blocked (needs BuildKit, the default since Docker 23; see vendor/README.md):
#   llama.cpp source, tried in this order:
#     1. vendor/llama.cpp-${LLAMA_CPP_REF}.tar.gz in the build context (no network at all)
#     2. git clone from GitHub
#     3. curl of LLAMA_CPP_URL (an internal mirror: --build-arg LLAMA_CPP_URL=...)
#   uv image:        --build-arg UV_IMAGE=registry.corp/astral-sh/uv:0.10
#   managed Python:  --build-arg UV_PYTHON_INSTALL_MIRROR=file:///vendor/python  (or an http mirror)
#   llama.cpp's web UI is downloaded from Hugging Face at build time (gpupool does not use it; a
#   failed download only logs a warning): --build-arg LLAMA_USE_PREBUILT_UI=OFF skips the attempt.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=22.04
# Overridable so a registry mirror can be used where ghcr.io is blocked.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10

FROM ${UV_IMAGE} AS uv

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama
# Pinned: the RPC protocol must match on every server, and gpupool was tested on b11342.
ARG LLAMA_CPP_REF=b11342
# Compute capabilities to compile kernels for. 61 Pascal, 70 Volta, 75 Turing, 80/86 Ampere,
# 89 Ada, 90 Hopper, 120 Blackwell (RTX 5090/5080, RTX PRO 6000; needs CUDA >= 12.8, and
# llama.cpp turns it into 120a for the FP4 tensor cores). Fewer archs = much faster build.
ARG CUDA_ARCHS="61;70;75;80;86;89;90;120"
# Last-resort source download, used when there is no vendored tarball and git cannot reach GitHub.
# Must serve the GitHub tag archive layout (one top-level directory, stripped on extract).
ARG LLAMA_CPP_URL=https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_CPP_REF}.tar.gz
# The llama-server web UI is fetched from Hugging Face during the build; failure is only a warning.
ARG LLAMA_USE_PREBUILT_UI=ON
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Fetch the source. The vendor/ bind mount is read-only and never becomes a layer.
# The short commit goes to /src/.gpupool-commit for the build step: a tarball has no .git (GitHub
# embeds the commit id in the tar header). A failed clone must not leave a half-done /src behind.
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
# Without .git, llama.cpp's CMake cannot derive the build number (a --depth 1 clone would report 1
# too), so pass it explicitly: tag bNNNN -> build number NNNN. `llama-server --version` prints it
# and the agent reports it as llama_version. LLAMA_BUILD_NUMBER / LLAMA_BUILD_COMMIT are honoured
# by CMakeLists.txt (`if (NOT DEFINED ...)`), verified against tag b11342.
# --allow-shlib-undefined: libcuda.so comes from the host driver at run time, only a stub
# exists in the build image.
RUN num="${LLAMA_CPP_REF#b}"; \
    case "$num" in ''|*[!0-9]*) num=0 ;; esac; \
    commit="$(cat /src/.gpupool-commit)"; \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_RPC=ON \
        -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_NUMBER="$num" -DLLAMA_BUILD_COMMIT="${commit:-unknown}" \
        -DLLAMA_USE_PREBUILT_UI="${LLAMA_USE_PREBUILT_UI}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server ggml-rpc-server
# Every requested architecture must really be in the CUDA backend: a GPU whose code is missing
# fails only at run time on that server ("no kernel image is available"), far from the build.
RUN lib="$(find build -name 'libggml-cuda.so*' -type f | head -n1)"; \
    have="$(cuobjdump --list-elf "$lib" | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"; \
    echo "CUDA kernels built for: $have"; \
    for a in $(echo "$CUDA_ARCHS" | tr ';' ' '); do \
        a="${a%%-*}"; a="${a%a}"; \
        echo "$have" | grep -qw "sm_$a" || { echo "missing sm_$a in $lib" >&2; exit 1; }; \
    done
RUN mkdir -p /out \
    && cp build/bin/llama-server build/bin/ggml-rpc-server /out/ \
    && find build -name "*.so*" -exec cp -P {} /out/ \;

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 libssl3 iptables \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=llama /out /opt/llama
ENV LD_LIBRARY_PATH=/opt/llama \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

# The Ubuntu 22.04 base has no Python 3.12, so uv downloads a managed CPython (python-build-standalone,
# from GitHub). Set UV_PYTHON_INSTALL_MIRROR to an internal mirror, or to file:///vendor/python for
# archives placed in vendor/python/ (layout: vendor/README.md). Empty = uv's default.
ARG UV_PYTHON_INSTALL_MIRROR=

WORKDIR /app
# .python-version pins 3.12, the version gpupool is tested on: without it uv picks the newest CPython
# that satisfies requires-python (>=3.12), which was 3.14 at the time of writing.
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md .python-version ./
# An empty value is unset first so uv never sees an empty mirror URL.
RUN --mount=type=bind,source=vendor,target=/vendor \
    [ -n "${UV_PYTHON_INSTALL_MIRROR:-}" ] || unset UV_PYTHON_INSTALL_MIRROR; \
    uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN --mount=type=bind,source=vendor,target=/vendor \
    [ -n "${UV_PYTHON_INSTALL_MIRROR:-}" ] || unset UV_PYTHON_INSTALL_MIRROR; \
    uv sync --frozen --no-dev --no-editable

ENV GPUPOOL_LLAMA_DIR=/opt/llama \
    GPUPOOL_CACHE_DIR=/data/cache \
    GPUPOOL_LOG_DIR=/data/logs \
    GPUPOOL_PORT=7070 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
VOLUME /data
EXPOSE 7070
# The agent API binds 0.0.0.0 (engines bind the resolved GPUPOOL_HOST), so probing loopback
# works whether or not GPUPOOL_HOST is set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS "http://127.0.0.1:${GPUPOOL_PORT}/health" || exit 1
ENTRYPOINT ["gpupool", "agent"]
