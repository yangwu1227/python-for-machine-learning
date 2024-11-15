FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

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
    uv sync --frozen --no-install-project --no-dev

# Must match the builder image 
FROM python:3.12-slim-bookworm AS runtime

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=1 \
    # Disable the generation of bytecode '.pyc' files
    PYTHONDONTWRITEBYTECODE=1 \
    # Entry point for training
    SAGEMAKER_PROGRAM="xgboost_entry.py" \
    CODE_PATH="/opt/ml/code"

WORKDIR $CODE_PATH
COPY --from=builder $CODE_PATH $CODE_PATH
COPY ./src/ ./

# Place executables in the environment at the front of the path
ENV PATH="$CODE_PATH/.venv/bin:$PATH"