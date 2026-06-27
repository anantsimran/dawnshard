FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
ENV UV_SYSTEM_PYTHON=1
ENV UV_ENV_FILE=".env"
COPY pyproject.toml uv.lock .env ./
ARG DEV=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$DEV" = "true" ]; then uv sync --frozen; else uv sync --frozen --no-dev; fi
COPY app/ /app