# StockSense

FastAPI 기반 주식 예측 및 분석 시스템

## 프로젝트 구조

```
StockSense/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 애플리케이션
│   ├── config.py            # 설정 관리
│   ├── api/                 # API 엔드포인트
│   ├── models/              # 데이터베이스 모델
│   ├── schemas/             # Pydantic 스키마
│   └── services/            # 비즈니스 로직
├── data/                    # 데이터 저장소
├── models/                  # ML 모델 저장소
├── tests/                   # 테스트
├── requirements.txt         # 의존성 패키지
└── .env.example            # 환경변수 예시
```

## 환경 설정

### 1. 가상환경 생성 및 활성화

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경변수 설정

```bash
# .env.example을 .env로 복사
copy .env.example .env

# .env 파일을 수정하여 필요한 설정 입력
```

## 실행 방법

### 개발 서버 실행

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

또는

```bash
python -m uvicorn app.main:app --reload
```

### API 문서 확인

서버 실행 후 브라우저에서:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 주요 기능

- 주식 데이터 수집 (yfinance)
- 머신러닝 기반 주가 예측
- RESTful API 제공
- 데이터 시각화

## 개발 중인 기능

- [ ] 주식 데이터 API
- [ ] 예측 모델 학습
- [ ] 실시간 데이터 업데이트
- [ ] 사용자 인증

## 기술 스택

- **Backend**: FastAPI
- **Database**: SQLite (개발), PostgreSQL (프로덕션 권장)
- **ML**: Scikit-learn, TensorFlow, Prophet
- **Data**: yfinance, pandas, numpy
