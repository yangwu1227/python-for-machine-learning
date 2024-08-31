#!/bin/bash

# Install Poetry outside of the environment it manages
curl -sSL https://install.python-poetry.org | python3 -
# Add the Poetry bin directory to the PATH
export PATH="$HOME/.local/bin:$PATH"

# Create and activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n tab-tf-env python=3.10 -y
conda activate tab-tf-env

# This creates a new project with directories src and tests and a pyproject.toml file
poetry new income-classification --name src 
# Use sed to update the Python version constraint in pyproject.toml
cd income-classification
sed --in-place 's/python = "^3.10"/python = ">=3.10, <3.12"/' pyproject.toml 
# Rename the project from "src" to "income-classification"
sed --in-place 's/name = "src"/name = "income-classification"/' pyproject.toml
# Remove the lines version, description, and authors under [tool.poetry]
sed --in-place '/^version = /d' pyproject.toml
sed --in-place '/^description = /d' pyproject.toml
sed --in-place '/^authors = /d' pyproject.toml
# Add `package-mode = false` under `[tool.poetry]`
sed --in-place '/\[tool.poetry\]/a package-mode = false' pyproject.toml

# Install project, test, notebook dependencies
poetry add "polars==0.18.15" "tensorflow==2.13.0" "tensorflow-io==0.32.0" "hydra-core==1.3.2" "boto3==1.26.131" "optuna==3.1.0" "s3fs==2023.6.0" "pymysql==1.1.0"
poetry add "pytest==7.4.2" --group test
poetry add "scikit-learn==1.3.1" "ipykernel==6.25.2" "ipython==8.15.0" "kaleido==0.2.1" "matplotlib==3.8.0" --group notebook

poetry install
