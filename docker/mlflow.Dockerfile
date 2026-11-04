# MLflow server at the same version as the client in requirements-dev.txt.
FROM mirror.gcr.io/library/python:3.10-slim
RUN pip install --no-cache-dir mlflow==3.18.0
