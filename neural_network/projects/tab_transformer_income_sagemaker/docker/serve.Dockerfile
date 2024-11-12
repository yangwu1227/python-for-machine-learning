# See https://hub.docker.com/r/tensorflow/tensorflow/
FROM tensorflow/tensorflow:2.13.0

# Install necessary system packages and Python libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir \
        sagemaker \
        sagemaker-training \
        polars \
        boto3 \
        flask

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True \
    # Disable the generation of bytecode '.pyc' files
    PYTHONDONTWRITEBYTECODE=True \
    # Set environment variable for the source code directory
    PROGRAM_PATH='/opt/program' \
    # Ensure PROGRAM_PATH is included in the search path for executables
    PATH="/opt/program:${PATH}"

# Copy src directory into the container
COPY ./src $PROGRAM_PATH

# Change permissions of the 'serve' script
RUN chmod +x $PROGRAM_PATH/serve

WORKDIR $PROGRAM_PATH
