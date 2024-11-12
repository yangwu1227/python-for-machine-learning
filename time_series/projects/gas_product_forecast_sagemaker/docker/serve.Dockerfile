FROM python:3.10.12-slim-bullseye

WORKDIR /opt/ml/code/

# Only copy files not listed in the dockerfile specific .dockerignore file
COPY ./src/ ./

# Install Python packages with specified versions and prevent caching
RUN pip install --no-cache-dir pandas[performance]==1.5.3 \
                              sktime==0.24.0 \
                              statsforecast==1.4.0 \
                              hydra-core==1.3.2 \
                              joblib==1.3.2 \
                              fastapi==0.104.1 \
                              uvicorn==0.24.0.post1

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True \
    # Disable the generation of bytecode '.pyc' files
    PYTHONDONTWRITEBYTECODE=True

ENTRYPOINT ["python3", "serve_entry.py"]
