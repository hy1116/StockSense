# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StockSense is a Korean stock portfolio management and AI price prediction service using the Korea Investment & Securities (KIS) Open API. It consists of a FastAPI backend, React frontend, ML pipeline, and Kafka-based real-time alert system.

## Development Commands

### Backend (FastAPI)
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (from project root)
uvicorn app.main:app --reload

# API docs available at: http://localhost:8000/docs
```

### Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev        # Dev server on port 3000 (proxies /api to localhost:8000)
npm run build
npm run lint
```

### Docker Compose (local full-stack)
```bash
docker-compose up -d                    # Start all services (postgres, redis, kafka, backend, frontend)
docker-compose --profile ml up argo    # Also start ML pipeline container
docker-compose down
```

### Database Migrations (Alembic)
```bash
# Apply all migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Note: In DEBUG=True mode, tables are auto-created on startup. Use Alembic for production.
```

### ML Pipeline
```bash
# Run full pipeline (collect → evaluate → crawl news → preprocess → train)
python ml/run_pipeline.py

# Individual steps
python ml/collect_daily_data.py
python ml/preprocess_data.py
python ml/train_model.py                          # XGBoost
python ml/daily_train_batch.py                    # XGBoost + LSTM
python ml/train_model.py --activate-anyway        # Force activate new model
python ml/train_model.py --no-db                  # Train without saving to DB
```

### Kubernetes
```bash
kubectl apply -f k8s/           # Deploy all manifests to `stocksense` namespace
kubectl apply -f k8s/kafka-deployment.yaml
```

## Architecture

### Services Structure
```
app/
├── main.py          # FastAPI app, lifespan (starts Kafka background tasks)
├── config.py        # Settings via pydantic-settings (.env file)
├── database.py      # SQLAlchemy 2.0 async engine (PostgreSQL/asyncpg)
├── api/             # Route handlers
├── models/          # SQLAlchemy ORM models
├── schemas/         # Pydantic request/response schemas
└── services/        # Business logic
```

### Backend API Routers
All routes are prefixed with `/api/`:
- `portfolio` – stock search, balance, OHLCV charts, buy/sell orders, market rankings
- `auth` – JWT-based registration and login
- `prediction` – price prediction results
- `watchlist`, `price_alert`, `comment`, `news`, `ml_model`

### Kafka Real-Time Alert Flow
Two background asyncio tasks start with the app (see `main.py` lifespan):

1. **`price_producer`** (`services/price_producer.py`): Polls active alert stock prices from **Naver Finance** (unauthenticated API) every `KAFKA_POLL_INTERVAL_SECONDS` seconds and publishes to topic `stock-price`.
2. **`alert_consumer`** (`services/alert_consumer.py`): Consumes `stock-price` topic, checks alert conditions (above/below target price), updates DB, and publishes triggered events to `price-alert-triggered`.

Kafka topics: `stock-price`, `price-alert-triggered`

### External APIs
- **KIS API** (`services/kis_api.py`): Synchronous client for stock data, portfolio balance, and order placement. Wrapped in `asyncio.run_in_executor` for FastAPI compatibility (`run_sync()` helper in `api/portfolio.py`).
- **Naver Finance** (`services/naver_finance.py`): Unauthenticated API for real-time price polling in the alert system.

### Caching (Redis)
Redis caches stock detail, intraday charts, and market rankings for 10 minutes. The `redis_client.py` service gracefully degrades if Redis is unavailable.

### ML Prediction Service (`services/prediction.py`)
- **Ensemble**: XGBoost (60%) + LSTM (40%)
- Models are loaded from PostgreSQL `model_training_history` table (active flag), with fallback to `./models/stock_prediction_v1.pkl`
- Technical indicators computed: MA5/10/20, RSI, Bollinger Bands, MACD
- News sentiment features fetched from `stock_news` table
- `PredictionService` is instantiated per-request (loads models from DB each time)

### ML Pipeline (`ml/`)
Sequential steps orchestrated by `run_pipeline.py`, run as a Kubernetes CronJob via Argo Workflows:
1. `collect_daily_data.py` – OHLCV data collection
2. `evaluate_predictions.py` – Evaluate previous day predictions
3. `crawl_news.py` – News crawling + sentiment analysis
4. `preprocess_data.py` – Feature engineering (outputs to `./data/datasets/`)
5. `daily_train_batch.py` – Train XGBoost + LSTM, save to DB with auto-activation if better than current

### Database
PostgreSQL with SQLAlchemy 2.0 async. Key tables: `stocks`, `users`, `watchlist`, `price_alerts`, `stock_news`, `model_training_history`, `comments`.

### Infrastructure
- **Local**: Docker Compose with KRaft-mode Kafka (no Zookeeper), Kafka UI at `localhost:8989`
- **Production (K8s)**: Kubernetes in `stocksense` namespace, EFK stack (Elasticsearch + Fluentd + Kibana) for log aggregation, Argo Workflows for ML CronJobs

### Environment Configuration
Key `.env` variables:
- `KIS_APP_KEY`, `KIS_APP_SECRET`, `KIS_ACCOUNT_NUMBER` – required for trading features
- `KIS_USE_MOCK=True` – use KIS mock/paper trading environment
- `DATABASE_URL` – `postgresql+asyncpg://...` (async driver required)
- `KAFKA_BOOTSTRAP_SERVERS` – `localhost:9092` for local dev, `kafka:29092` inside Docker
- `REDIS_HOST`, `REDIS_PORT`
- `JWT_SECRET_KEY`

Logging uses **loguru** with all standard `logging` module output redirected via `InterceptHandler`. `/health` endpoint logs are filtered out globally.
