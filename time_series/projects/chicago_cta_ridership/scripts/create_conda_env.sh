#!/bin/bash

# Locate the conda executable
CONDA_EXEC=$(which conda)

# If empty, conda is not installed or not in the PATH
if [ -z "$CONDA_EXEC" ]; then
    echo "conda is not installed or not in the PATH"
    exit 1
fi

# Create a new conda environment with Python 3.10
yes | $CONDA_EXEC create --name ts_env python=3.10

if [ $? -eq 0 ]; then
    echo "Environment 'ts_env' with Python 3.10 was successfully created"
else
    echo "There was an issue creating the environment"
    exit 2
fi

read -sp "Enter the 'absolute' path to your conda's activate executable: " conda_activate_executable

# Activate the newly created environment
source "$conda_activate_executable" ts_env || { echo "Failed to activate ts_env. Check the error messages above."; exit 1; }

# Install poetry using pip
pip3 install poetry

if [ $? -eq 0 ]; then
    echo "Poetry was successfully installed in 'ts_env'"
else
    echo "There was an issue installing Poetry"
    exit 3
fi