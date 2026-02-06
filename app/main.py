from time import time
from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import iterate_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from h11 import Request

from app.api import portfolio, prediction, auth, comment, ml_model, news
from app.database import init_db, close_db
from app.config import get_settings

LOGGING_FORMAT = "%(asctime)s  %(levelname)5s %(process)d --- [%(threadName)15s] %(name)-40s : %(message)s"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    format="%(message)s"
)
logger = logging.getLogger("api_logger")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # WebSocket 요청은 로깅 미들웨어 스킵
        if request.url.path.startswith("/api/portfolio/ws/"):
            return await call_next(request)
        # 1. Request 정보 추출
        start_time = time.time()
        url = request.url.path
        method = request.method
        headers = dict(request.headers)

        # Request Body 처리 (Body를 읽으면 스트림이 소비되므로 복사본 생성 필요)
        body = await request.body()
        request_body = body.decode() if body else None

        # 2. 다음 핸들러(API 로직) 실행
        response = await call_next(request)

        # 3. Response 정보 추출
        process_time = (time.time() - start_time) * 1000 # ms 단위
        response_body = [section async for section in response.body_iterator]
        response.body_iterator = iterate_in_threadpool(iter(response_body))

        status_code = response.status_code

        # 4. 로그 출력 (JSON 형태로 찍으면 Kibana에서 검색하기 매우 편함)
        log_dict = {
            "url": url,
            "method": method,
            "status_code": status_code,
            "process_time_ms": f"{process_time:.2f}",
            "request": {
                "headers": headers,
                "body": request_body
            },
            "response": {
                "body": response_body[0].decode() if response_body else None
            }
        }

        logger.info(log_dict)

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    logger.info("Starting StockSense API...")
    await init_db()
    logger.info("Database initialized")
    yield
    # 종료 시
    logger.info("Shutting down StockSense API...")
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
