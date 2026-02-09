# 백엔드 Dockerfile
FROM python:3.11-slim AS builder

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app

# 애플리케이션 코드 복사
COPY --from=builder /install /usr/local
COPY . .

# 모델 및 데이터 디렉토리 생성
RUN mkdir -p /app/models /app/data

# 포트 노출
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "30", "--no-access-log"]
