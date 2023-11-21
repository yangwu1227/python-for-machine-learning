FROM python:3.10.12-slim-bullseye

WORKDIR /opt/ml/code/

# Only copy files not listed in the dockerfile specific .dockerignore file
COPY ./src/ ./

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
RUN pip install pandas[performance]==1.5.3 \ 
                sktime==0.24.0 \
                statsforecast==1.4.0 \
                statsmodels==0.14.0 \
                hydra-core==1.3.2 \
                matplotlib==3.8.0 \
                joblib==1.3.2 \
                sagemaker-training==4.7.4

# Rename train_entry.py to train.py
RUN mv train_entry.py train.py

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True
# Disable the generation of bytecode '.pyc' files
ENV PYTHONDONTWRITEBYTECODE=True
# Set entrypoint to the training script
ENV SAGEMAKER_PROGRAM train.py