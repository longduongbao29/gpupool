# gpupool coordinator: web UI + admin API + scheduler + OpenAI-compatible router. No GPU needed.
#
#   docker run -d --name gpupool -p 8080:8080 -v gpupool:/data ghcr.io/longduongbao29/gpupool-coordinator
#   docker logs gpupool    # UI URL, admin key and the join command for GPU servers
#
# The admin key and cluster token are generated on first start and kept in /data/secrets.json.
# Set GPUPOOL_ADMIN_KEY / GPUPOOL_CLUSTER_TOKEN to choose them, and GPUPOOL_PUBLIC_URL if the
# address shown in the log is not reachable from your servers. Mount existing GGUF files with
# -v /srv/gguf:/models:ro and add them in the UI by path.
# /data holds the database, secrets and models downloaded from Hugging Face.
#
# Behind a proxy: Docker forwards the predefined build args to every RUN step (apt-get, git clone,
# uv downloads), so pass them at build time:
#   docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy \
#     --build-arg no_proxy=$no_proxy -f docker/coordinator.Dockerfile -t gpupool-coordinator .
# They are deliberately NOT declared with ARG/ENV here: predefined proxy args are excluded from
# the image history and never baked into the final image (they are site secrets). Pulling the
# base images and the uv image is done by the Docker daemon: configure the daemon proxy for that.
# At run time pass them through from the host: -e http_proxy -e https_proxy -e no_proxy

FROM docker.io/library/python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
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
    GPUPOOL_MODELS_DIR=/data/models
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)" || exit 1
ENTRYPOINT ["gpupool", "coordinator"]
