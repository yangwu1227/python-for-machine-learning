#!/bin/bash

set -e

echo "Installing poetry..."
# Install Poetry outside of the environment it manages
curl -sSL https://install.python-poetry.org | python3 -

echo "Adding poetry to PATH..."
export PATH="$HOME/.local/bin:$PATH"

echo "Initializing conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n forecast-project-env -y python=3.10

echo "Activating conda environment..."
conda activate forecast-project-env

echo "Creating new poetry project..."
poetry new forecast-project --python ">=3.10, <3.12" --flat --name src

cd forecast-project

echo "Installing dependencies..."
poetry add "pandas[performance]==1.5.3" "hydra-core==1.3.2" "boto3==1.26.131" \
    "pmdarima==2.0.4" "sktime==0.24.0" "statsmodels==0.14.0" "statsforecast==1.4.0" \
    "xlrd==2.0.1" "fastapi==0.104.1" "joblib==1.3.2" "uvicorn==0.24.0.post1"

poetry add "pytest==7.4.2" --group test
poetry add "ipykernel==6.25.2" "ipython==8.15.0" "kaleido==0.2.1" "matplotlib==3.8.0" --group notebook

echo "Installing all dependencies..."
poetry install

echo "Project setup complete!"
