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
# Where ghcr.io / GitHub is blocked (see vendor/README.md): the build itself never touches GitHub
# (UV_PYTHON_DOWNLOADS=never, Python comes from the base image). The uv binary comes from
#   --build-arg UV_IMAGE=registry.corp/astral-sh/uv:0.10   (a registry mirror), or
#   --build-arg UV_FROM_PYPI=1                              (pip install uv; PyPI usually works via the proxy)

# uv binary source. UV_FROM_PYPI selects the stage name below: empty -> uv-src, 1 -> uv-src1.
# BuildKit builds only the stage that is used, so the unused source is never pulled.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10
ARG UV_FROM_PYPI=
ARG UV_PYPI_SPEC="uv>=0.10,<0.11"

FROM ${UV_IMAGE} AS uv-src

FROM docker.io/library/python:3.12-slim AS uv-src1
ARG UV_PYPI_SPEC
RUN pip install --no-cache-dir "${UV_PYPI_SPEC}" && cp "$(command -v uv)" /uv

FROM uv-src${UV_FROM_PYPI} AS uv

FROM docker.io/library/python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

ENV GPUPOOL_HOST=0.0.0.0 \
    GPUPOOL_PORT=8080 \
    GPUPOOL_DB_PATH=/data/coordinator.db \
    GPUPOOL_MODELS_DIR=/data/models \n    GPUPOOL_MODEL_ROOTS=/models
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)" || exit 1
ENTRYPOINT ["gpupool", "coordinator"]
