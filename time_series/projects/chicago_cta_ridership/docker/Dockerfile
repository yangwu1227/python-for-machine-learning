FROM python:3.10.12-slim-bullseye as builder

WORKDIR /project

# Copy only the necessary files for dependency installation
COPY poetry.lock pyproject.toml ./

# Install dependencies, including build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && pip3 install poetry==1.7.1 gunicorn==21.2.0\
    && poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction \
    && apt-get purge -y --auto-remove gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary parts of the src package
COPY src/ src/
COPY app/ app/

# Start a new stage from a smaller base image
FROM python:3.10.12-slim-bullseye

WORKDIR /project

# Copy the built artifacts from the builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /project /project

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "app.app:server"]