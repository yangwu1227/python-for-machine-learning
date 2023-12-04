#!/bin/bash

read -sp "Enter the 'absolute' path to your conda's activate executable: " conda_activate_executable

# Activate the newly created environment
source "$conda_activate_executable" ts_env || { echo "Failed to activate ts_env. Check the error messages above."; exit 1; }

# This creates a new project with directories src and tests and a pyproject.toml file
poetry new chicago_cta_ridership --name src 
# Use sed to update the Python version constraint in pyproject.toml
sed -i '' 's/python = "^3.10"/python = ">=3.10, <3.12"/' chicago_cta_ridership/pyproject.toml

# Install project, test, notebook dependencies
cd chicago_cta_ridership

# Some initial dependencies
poetry add "tensorflow>=2.13.0" "tensorflow-io>=0.34.0" "hydra-core>=1.3.2" "boto3>=1.26.131" "optuna>=3.1.0" "s3fs>=2023.6.0" "scikit-learn>=1.3.1" \
           "sktime>=0.23.0" "pmdarima>=2.0.3" "statsmodels>=0.14.0" "pandas>=2.1.1" "pyarrow>=13.0.0" "matplotlib>=3.8.0"
poetry add "pytest>=7.4.2" --group test
poetry add "ipykernel>=6.25.2" "python-dotenv>=1.0.0" "poetry-dotenv-plugin>=0.2.0" "ipython>=8.15.0" "notebook>=7.0.5" "jupyterlab>=4.0.7" "dvc>=3.25.0" --group dev

poetry install