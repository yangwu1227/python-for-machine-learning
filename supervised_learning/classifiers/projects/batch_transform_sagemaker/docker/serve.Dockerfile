FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Set environment variable for the source code directory
ENV CODE_PATH="/opt/ml/code" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR $CODE_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen


FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAGEMAKER_PROGRAM="main.py" \
    CODE_PATH="/opt/ml/code"

WORKDIR $CODE_PATH
COPY --from=builder $CODE_PATH $CODE_PATH
COPY ./server/ ./server/
# Move start_server.py to the root of the code directory
RUN mv server/start_server.py start_server.py

# Place executables in the environment at the front of the path
ENV PATH="$CODE_PATH/.venv/bin:$PATH"

EXPOSE 8080

ENTRYPOINT ["python", "start_server.py"]
