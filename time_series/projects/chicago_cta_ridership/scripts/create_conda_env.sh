#!/bin/bash

ENV_NAME="ts_env"
PY_VERSION="3.10"
POETRY_VERSION="1.6.1"

echo "Installing Poetry ${POETRY_VERSION}..."
curl -sSL https://install.python-poetry.org | POETRY_VERSION="${POETRY_VERSION}" python3 -

# Add poetry to PATH
case "$(uname -s)" in
  Darwin)
    export PATH="$HOME/Library/Application Support/pypoetry/bin:$PATH"
    ;;
  Linux)
    export PATH="$HOME/.local/share/pypoetry/bin:$PATH"
    ;;
  *)
    echo "Unknown OS: $(uname -s); ensure Poetry is in PATH manually"
    ;;
esac

# https://stackoverflow.com/a/56155771/12923148
eval "$(conda shell.bash hook)"

# Create a new conda environment
echo "Creating conda environment '${ENV_NAME}' with Python ${PY_VERSION}..."
conda create -n "${ENV_NAME}" -y "python=${PY_VERSION}" || { echo "There was an issue creating the environment"; exit 2; }

echo "Environment '${ENV_NAME}' with Python ${PY_VERSION} was successfully created"

# Activate conda environment
echo "Activating '${ENV_NAME}'..."
conda activate "${ENV_NAME}" || { echo "Failed to activate ${ENV_NAME}"; exit 1; }

read -sp "Enter the 'absolute' path to your conda's activate executable: " conda_activate_executable

# Activate the newly created environment
source "$conda_activate_executable" ts_env || { echo "Failed to activate ts_env. Check the error messages above."; exit 1; }
