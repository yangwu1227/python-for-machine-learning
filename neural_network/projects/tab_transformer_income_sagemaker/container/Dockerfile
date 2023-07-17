# See https://hub.docker.com/r/tensorflow/tensorflow/
FROM tensorflow/tensorflow:2.12.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx curl && \
    pip3 install \
    sagemaker \
    sagemaker-training \
    boto3 \
    flask 

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True
# Disable the generation of bytecode '.pyc' files
ENV PYTHONDONTWRITEBYTECODE=True
# Set environment variable for the source code directory
ENV PROGRAM_PATH='/opt/program'
# Ensure PROGRAM_PATH is included in the search path for executables, so any executables (i.e., 'serve') in that directory will take precedence over other executables
ENV PATH="/opt/program:${PATH}"

# Copy src directory into the container
COPY ./src $PROGRAM_PATH

# Change permissions of the 'serve' script
RUN chmod +x $PROGRAM_PATH/serve

WORKDIR $PROGRAM_PATH