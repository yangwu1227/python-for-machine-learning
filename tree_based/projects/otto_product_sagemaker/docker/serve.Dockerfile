# To get the latest RAPIDS, simply exclude the RAPIDS version '23.04'
FROM rapidsai/rapidsai-core:23.04-cuda11.8-base-ubuntu22.04-py3.10

# Remove outdated signing keys according to https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
RUN apt-key del 7fa2af80
# Get latest keys from 'https://developer.download.nvidia.com/compute/cuda/repos/', which should match the Ubuntu version in the base image above
# Use && to chain commands together and dkpg to install the keys
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    source activate rapids && \
    pip3 install --no-cache-dir \
        sagemaker \
        sagemaker-training \
        boto3 \
        s3fs \
        flask \
        lightgbm && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=1 \
    # Disable the generation of bytecode '.pyc' files
    PYTHONDONTWRITEBYTECODE=1 \
    PROGRAM_PATH='/opt/program' \
    # Ensure PROGRAM_PATH is included in the search path for executables, so any executables (i.e., 'serve') in that directory will take precedence over other executables
    PATH="/opt/program:$PATH"

# Copy src directory into the container
COPY ./src $PROGRAM_PATH

# Change permissions for 'serve' script and set working directory
RUN chmod +x $PROGRAM_PATH/serve
WORKDIR $PROGRAM_PATH
