FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Install dependencies and Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    xgboost \
    scikit-learn \ 
    optuna \
    hydra-core \
    pymysql \
    polars \
    numpy \
    boto3 \
    sagemaker-training \
    sagemaker \
    "boto3-stubs[s3,sagemaker,secretsmanager]"

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAGEMAKER_PROGRAM="xgboost_entry.py" \
    CODE_PATH="/opt/ml/code"

COPY ./src $CODE_PATH

WORKDIR $CODE_PATH
