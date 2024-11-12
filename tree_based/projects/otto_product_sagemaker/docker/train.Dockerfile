# To get the latest RAPIDS, simply exclude the RAPIDS version '23.04'
FROM rapidsai/rapidsai-core:23.04-cuda11.8-base-ubuntu22.04-py3.10

# Check if the required build argument is provided
ARG SAGEMAKER_PROGRAM
RUN if [ -z "$SAGEMAKER_PROGRAM" ]; then \
      echo "ERROR: The build argument 'SAGEMAKER_PROGRAM' is required to specify the entry point script in the src directory"; \
      exit 1; \
    fi

# Remove outdated signing keys according to https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80 && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    rm cuda-keyring_1.0-1_all.deb

# Install dependencies, activate the environment, and install Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    source activate rapids && \
    pip3 install --no-cache-dir \
        sagemaker \
        sagemaker-training \
        dask-ml \
        boto3 \
        s3fs \
        lightgbm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True \
    # Disable the generation of bytecode '.pyc' files
    PYTHONDONTWRITEBYTECODE=True \
    CODE_PATH='/opt/ml/code' \
    # Define entrypoint script using the build argument at build time
    SAGEMAKER_PROGRAM=$SAGEMAKER_PROGRAM

# Copy src directory into the container
COPY ./src $CODE_PATH

WORKDIR $CODE_PATH
