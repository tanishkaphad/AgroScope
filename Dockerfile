# Multi-stage lightweight Dockerfile for AgroScope FastAPI Backend
FROM mirror.gcr.io/library/python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Stage
FROM mirror.gcr.io/library/python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

COPY configs/ ./configs/
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY data/raw/study_area/ ./data/raw/study_area/
COPY data/processed/ ./data/processed/
COPY reports/ ./reports/
COPY frontend/dist/ ./frontend/dist/

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
