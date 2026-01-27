# StockSense

한국투자증권 API를 활용한 주식 포트폴리오 관리 및 AI 주가 예측 서비스

## 주요 기능

- 실시간 포트폴리오 조회 (잔고, 보유종목, 수익률)
- 종목 상세 정보 및 차트
- AI 기반 주가 예측
- 주식 매수/매도 주문
- 거래량 상위 종목 조회

## 기술 스택

### Backend
- FastAPI
- PostgreSQL
- Redis (캐싱)
- 한국투자증권 Open API

### Frontend
- React + Vite
- Chart.js
- Axios

### Infrastructure
- Docker / Docker Compose
- Kubernetes

## 시작하기

### 환경 변수 설정

`.env.example`을 `.env`로 복사 후 설정:

```bash
cp .env.example .env
```

필수 설정:
- `KIS_APP_KEY`: 한국투자증권 앱 키
- `KIS_APP_SECRET`: 한국투자증권 앱 시크릿
- `KIS_ACCOUNT_NUMBER`: 계좌번호

### 로컬 실행

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Docker Compose 실행

```bash
docker-compose up -d
```

### Kubernetes 배포

```bash
kubectl apply -f k8s/
```

## 프로젝트 구조

```
StockSense/
├── app/                    # Backend (FastAPI)
│   ├── api/               # API 라우터
│   ├── models/            # DB 모델
│   ├── schemas/           # Pydantic 스키마
│   └── services/          # 비즈니스 로직
├── frontend/              # Frontend (React)
│   └── src/
│       ├── components/
│       ├── pages/
│       └── services/
├── k8s/                   # Kubernetes 매니페스트
├── scripts/               # 유틸리티 스크립트
└── docker-compose.yml
```

## API 문서

서버 실행 후 접속:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 라이선스

MIT License
