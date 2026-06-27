FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
ENV UV_SYSTEM_PYTHON=1
COPY pyproject.toml uv.lock ./
ARG DEV=false
RUN if [ "$DEV" = "true" ]; then uv sync --frozen; else uv sync --frozen --no-dev; fi
COPY app/ /app