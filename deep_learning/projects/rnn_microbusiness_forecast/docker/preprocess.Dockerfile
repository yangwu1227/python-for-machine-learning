FROM python:3.10.12-slim-bullseye

WORKDIR /src

# Only copy files not listed in the dockerfile-specific .dockerignore file
COPY ./src/ ./

# Check if the correct files are copied
RUN ls -la

# Install dependencies from requirements file and prevent caching
RUN pip install --no-cache-dir -r requirements.txt

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True \
    # Disable the generation of bytecode '.pyc' files
    PYTHONDONTWRITEBYTECODE=True

ENTRYPOINT ["python3", "preprocess_entry.py"]
