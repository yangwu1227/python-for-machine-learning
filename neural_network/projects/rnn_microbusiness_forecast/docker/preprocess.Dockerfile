FROM python:3.10.12-slim-bullseye

WORKDIR /src

# Only copy files not listed in the dockerfile specific .dockerignore file
COPY ./src/ ./

# Check if right files are copied
RUN ls -la

RUN pip install --no-cache-dir -r requirements.txt

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True
# Disable the generation of bytecode '.pyc' files
ENV PYTHONDONTWRITEBYTECODE=True

ENTRYPOINT ["python3", "preprocess_entry.py"]