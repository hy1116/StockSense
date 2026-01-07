from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api import portfolio, prediction
from app.database import init_db, close_db

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio.router)
app.include_router(prediction.router)


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
