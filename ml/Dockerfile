# ML Worker Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# ML 모듈 복사
COPY ml/ ./ml/

# app 모듈 복사 (DB 모델, 설정 등)
COPY app/ ./app/

# Alembic 설정 복사
COPY alembic/ ./alembic/
COPY alembic.ini .

RUN mkdir -p /app/models /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

#CMD ["python", "scripts/ml/run_pipeline.py"]
