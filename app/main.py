import sys
import time
from loguru import logger
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

logger.remove()
logger.add(
    sys.stdout, 
    colorize=True, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}:{line}</cyan> - <level>{message}</level>",
    filter=lambda record: "/health" not in record["message"] # /health 포함 로그 전역 필터링
)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # WebSocket 및 Health 체크 스킵
        if request.url.path.startswith("/api/portfolio/ws/") or request.url.path == "/health":
            return await call_next(request)
        
        start_time = time.time()
        
        # Request Body 처리 (주의: 큰 데이터의 경우 메모리 이슈가 있을 수 있음)
        body = await request.body()
        request_body = body.decode() if body else None

        response = await call_next(request)

        # Response Body 복구 및 추출
        process_time = (time.time() - start_time) * 1000
        response_body_bytes = [section async for section in response.body_iterator]
        response.body_iterator = iterate_in_threadpool(iter(response_body_bytes))

        status_code = response.status_code
        
        # 로그 데이터 구성
        # .opt(depth=1)을 사용하면 미들웨어가 아닌 실제 호출 지점을 찍을 수도 있지만, 
        # 여기서는 미들웨어 라인이 찍히는 것이 정상입니다.
        logger.bind(
            url=request.url.path,
            method=request.method,
            status_code=status_code
        ).info(
            f"Request: {request.method} {request.url.path} | "
            f"Status: {status_code} | "
            f"Time: {process_time:.2f}ms | "
            f"ReqBody: {request_body} | "
            f"ResBody: {response_body_bytes[0].decode() if response_body_bytes else None}"
        )

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
