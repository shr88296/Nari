FROM ghcr.io/astral-sh/uv:bookworm-slim AS build

# See github.com/astral-sh/uv-docker-example
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/python
ENV UV_PYTHON_PREFERENCE=only-managed

RUN uv python install 3.12

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM docker.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

COPY --from=build --chown=1000:1000 /python /python
COPY --from=build --chown=1000:1000 /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"
ENV GRADIO_SHARE="False"

CMD ["python", "/app/app.py"]
