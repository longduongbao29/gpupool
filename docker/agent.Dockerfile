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
# CUDA 12.4 runs on drivers >= 525 (minor-version compatibility). For older drivers rebuild
# with a lower CUDA_VERSION.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS llama
# Pinned: the RPC protocol must match on every server, and gpupool was tested on b11342.
ARG LLAMA_CPP_REF=b11342
# Compute capabilities to compile kernels for. 61 Pascal, 70 Volta, 75 Turing, 80/86 Ampere,
# 89 Ada, 90 Hopper. Fewer archs = much faster build.
ARG CUDA_ARCHS="61;70;75;80;86;89;90"
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
# --allow-shlib-undefined: libcuda.so comes from the host driver at run time, only a stub
# exists in the build image.
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_RPC=ON \
        -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server ggml-rpc-server
RUN mkdir -p /out \
    && cp build/bin/llama-server build/bin/ggml-rpc-server /out/ \
    && find build -name "*.so*" -exec cp -P {} /out/ \;

FROM docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
COPY --from=llama /out /opt/llama
ENV LD_LIBRARY_PATH=/opt/llama \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

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
