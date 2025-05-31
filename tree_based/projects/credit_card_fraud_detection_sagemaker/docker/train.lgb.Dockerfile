FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    vim \
    mercurial \
    subversion \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    gcc \
    g++ \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ocl-icd-opencl-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add OpenCL ICD files for LightGBM
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    scikit-learn \
    optuna \
    s3fs \
    hydra-core \
    pymysql \
    polars \
    numpy \
    boto3 \
    sagemaker-training \
    sagemaker \
    "boto3-stubs[s3,sagemaker,secretsmanager]" 

RUN pip3 install --no-cache-dir --no-binary \
    lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON \ 
    lightgbm

# Ensure python I/O is unbuffered for immediate log messages
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SAGEMAKER_PROGRAM="lightgbm_entry.py" \
    CODE_PATH="/opt/ml/code"

COPY ./src $CODE_PATH

WORKDIR $CODE_PATH
