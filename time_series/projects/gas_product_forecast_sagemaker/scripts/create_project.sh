#!/bin/bash

# Install Poetry outside of the environment it manages
curl -sSL https://install.python-poetry.org | python3 -
# Add the Poetry bin directory to the PATH
export PATH="$HOME/.local/bin:$PATH"

# Create and activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n forecast-project-env python=3.10 -y
conda activate forecast-project-env

# Use Poetry to create a new project with directories src and tests and a pyproject.toml file
poetry new forecast-project --name src 

# Use sed to update the Python version constraint in pyproject.toml
cd forecast-project
sed --in-place 's/python = "^3.10"/python = ">=3.10, <3.12"/' pyproject.toml 
# Rename the project from "src" to "forecast-project"
sed --in-place 's/name = "src"/name = "forecast-project"/' pyproject.toml
# Remove the lines version, description, and authors under [tool.poetry]
sed --in-place '/^version = /d' pyproject.toml
sed --in-place '/^description = /d' pyproject.toml
sed --in-place '/^authors = /d' pyproject.toml
# Add `package-mode = false` under `[tool.poetry]`
sed --in-place '/\[tool.poetry\]/a package-mode = false' pyproject.toml

# Install project, test, and notebook dependencies using Poetry
poetry add "pandas[performance]==1.5.3" "hydra-core==1.3.2" "boto3==1.26.131" \
           "pmdarima==2.0.4" "sktime==0.24.0" "statsmodels==0.14.0" "statsforecast==1.4.0" \
           "xlrd==2.0.1" "fastapi==0.104.1" "joblib==1.3.2" "uvicorn==0.24.0.post1" 
poetry add "pytest==7.4.2" --group test
poetry add "ipykernel==6.25.2" "ipython==8.15.0" "kaleido==0.2.1" "matplotlib==3.8.0" --group notebook

# Install all dependencies specified in pyproject.toml
poetry install
