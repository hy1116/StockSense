import asyncio
import sys
import time
from loguru import logger
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import portfolio, prediction, auth, comment, ml_model, news, watchlist
from app.api import price_alert
from app.database import init_db, close_db
from app.config import get_settings
from app.services.kafka_client import close_kafka_producer
from app.services.price_producer import run_price_producer
from app.services.alert_consumer import run_alert_consumer
from app.services.naver_finance import close_naver_finance_client

logger.remove()
logger.add(
    sys.stdout, 
    colorize=True, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}:{line}</cyan> - <level>{message}</level>",
    filter=lambda record: "/health" not in record["message"] # /health 포함 로그 전역 필터링
)

_REQ_BODY_MAX = 300  # 요청 바디 로그 최대 길이 (bytes)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # WebSocket 및 Health 체크 스킵
        if request.url.path.startswith("/api/portfolio/ws/") or request.url.path == "/health":
            return await call_next(request)

        start_time = time.time()

        # 요청 바디: POST/PUT/PATCH 에만, 최대 300자로 잘라서 기록
        request_body = None
        if request.method in ("POST", "PUT", "PATCH"):
            raw = await request.body()
            if raw:
                decoded = raw.decode(errors="replace")
                request_body = decoded[:_REQ_BODY_MAX] + ("…" if len(decoded) > _REQ_BODY_MAX else "")

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        status_code = response.status_code

        # 응답 바디는 읽지 않음 — 전체를 버퍼링하면 대용량 응답에서 심각한 지연 발생
        logger.bind(
            url=request.url.path,
            method=request.method,
            status_code=status_code,
        ).info(
            f"{request.method} {request.url.path} | "
            f"{status_code} | {process_time:.0f}ms"
            + (f" | body={request_body}" if request_body else "")
        )

        return response

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# 기본 logging을 loguru로 리다이렉트
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# uvicorn 로거도 loguru로 리다이렉트 (자체 핸들러 제거)
for _logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    _uv_logger = logging.getLogger(_logger_name)
    _uv_logger.handlers = [InterceptHandler()]
    _uv_logger.propagate = False

# 내부 라이브러리 로그 레벨 조정 (DEBUG/INFO 과다 출력 방지)
logging.getLogger("aiokafka").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy").setLevel(logging.WARNING)          # 모든 sqlalchemy 서브로거 포함
logging.getLogger("sqlalchemy.engine.Engine").setLevel(logging.WARNING)  # echo 직접 억제

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # ── 시작 ──────────────────────────────────────
    logger.info("Starting StockSense API...")
    await init_db()
    logger.info("Database initialized")

    # Kafka 백그라운드 태스크 시작
    producer_task = asyncio.create_task(run_price_producer(), name="price-producer")
    consumer_task = asyncio.create_task(run_alert_consumer(), name="alert-consumer")
    logger.info("Kafka background tasks started (price-producer, alert-consumer)")

    yield

    # ── 종료 ──────────────────────────────────────
    logger.info("Shutting down StockSense API...")

    # Kafka 태스크 취소
    producer_task.cancel()
    consumer_task.cancel()
    await asyncio.gather(producer_task, consumer_task, return_exceptions=True)
    logger.info("Kafka background tasks stopped")

    # Kafka Producer 종료
    await close_kafka_producer()

    # 네이버 금융 HTTP 클라이언트 종료
    await close_naver_finance_client()

    await close_db()
    logger.info("Database connection closed")


app = FastAPI(
    title="StockSense API",
    description="주식 예측 및 분석 API",
    version="1.0.0",
    lifespan=lifespan
)

settings = get_settings()
cors_origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 미들웨어 추가
app.add_middleware(LoggingMiddleware)

app.include_router(auth.router)
app.include_router(portfolio.router)
app.include_router(prediction.router)
app.include_router(comment.router)
app.include_router(ml_model.router)
app.include_router(news.router)
app.include_router(watchlist.router)
app.include_router(price_alert.router)  # 주가 알림


@app.get("/")
async def root():
    return {
        "message": "StockSense API에 오신 것을 환영합니다",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
