FROM python:3.10.12-slim-bullseye

WORKDIR /src

# Only copy files not listed in the dockerfile specific .dockerignore file
COPY ./src/ ./

RUN pip install pandas[performance]==1.5.3 \ 
                sktime==0.24.0 \
                statsforecast==1.4.0 \
                hydra-core==1.3.2

# Ensure python I/O is unbuffered so log messages are immediate
ENV PYTHONUNBUFFERED=True
# Disable the generation of bytecode '.pyc' files
ENV PYTHONDONTWRITEBYTECODE=True

ENTRYPOINT ["python3", "preprocess_entry.py"]