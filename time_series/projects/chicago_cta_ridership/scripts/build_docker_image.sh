#!/bin/bash

# Always anchor the execution to the directory this script is in, so we can run this bash script from anywhere
SCRIPT_DIR=$(python3 -c "import os; print(os.path.dirname(os.path.realpath('$0')))")

# Set BUILD_CONTEXT as the parent directory of SCRIPT_DIR, which is the root of the project
BUILD_CONTEXT=$(dirname "$SCRIPT_DIR")

# Set DOCKERFILE_PATH as the path to the Dockerfile
DOCKERFILE_PATH="$BUILD_CONTEXT/docker/Dockerfile"

DOCKER_BUILDKIT=1 docker build \
    -f "$DOCKERFILE_PATH" \
    -t latest \
    "$BUILD_CONTEXT"