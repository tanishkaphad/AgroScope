# Airflow runs the pipeline as one container, the simplest free setup.
# The pipeline's packages have their own environment, so they can't conflict with Airflow's.
FROM mirror.gcr.io/apache/airflow:2.11.0-python3.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/pipeline/
RUN python -m venv /opt/pipeline-venv \
    && /opt/pipeline-venv/bin/pip install --no-cache-dir -r /tmp/pipeline/requirements-dev.txt \
    && rm -rf /tmp/pipeline

# DVC keeps its record of file hashes here, on a volume that takes this ownership.
RUN mkdir -p /var/tmp/dvc && chown 50000:0 /var/tmp/dvc

USER airflow
