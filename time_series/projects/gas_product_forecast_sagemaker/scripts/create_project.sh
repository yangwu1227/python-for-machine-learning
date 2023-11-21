#!/bin/bash

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh && conda activate python3 && pip3 install poetry

# This creates a new project with directories src and tests and a pyproject.toml file
poetry new forecast-project --name src 
# Use sed to update the Python version constraint in pyproject.toml
sed --in-place 's/python = "^3.10"/python = ">=3.10, <3.12"/' pyproject.toml 

# Install project, test, notebook dependencies
cd forecast-project
poetry add "pandas[performance]==1.5.3" "hydra-core==1.3.2" "boto3==1.26.131" "pmdarima==2.0.4" "sktime==2.24.0" "statsmodels==0.14.0" "statsforecast==1.4.0" "xlrd==2.0.1" "fastapi==0.104.1" "joblib==1.3.2" "uvicorn==0.24.0.post1"
poetry add "pytest==7.4.2" --group test
poetry add "ipykernel==6.25.2" "ipython==8.15.0" "kaleido==0.2.1" "matplotlib==3.8.0" --group notebook

poetry install