"""
데이터베이스 연결 설정
SQLAlchemy 2.0 비동기 엔진 사용
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# SQLAlchemy Base
Base = declarative_base()

# 비동기 엔진 생성
# SQLite는 aiosqlite 드라이버 필요: sqlite+aiosqlite:///
# PostgreSQL은 asyncpg 드라이버 필요: postgresql+asyncpg://
is_sqlite = "sqlite" in settings.database_url

engine_kwargs = {
    "echo": settings.debug,  # SQL 쿼리 로깅
}

# SQLite는 NullPool 사용 (연결 풀 비활성화)
if is_sqlite:
    engine_kwargs["poolclass"] = NullPool
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # PostgreSQL 등은 연결 풀 사용
    engine_kwargs["pool_pre_ping"] = True
    engine_kwargs["pool_size"] = 5
    engine_kwargs["max_overflow"] = 10
    engine_kwargs["pool_timeout"] = 30
    engine_kwargs["pool_recycle"] = 1800  # 30분마다 커넥션 재생성

engine = create_async_engine(settings.database_url, **engine_kwargs)

# 세션 팩토리
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """데이터베이스 세션 의존성

    FastAPI 의존성 주입에 사용
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """데이터베이스 초기화

    애플리케이션 시작 시 테이블 생성
    """
    async with engine.begin() as conn:
        # 개발 환경에서만 테이블 자동 생성
        if settings.debug:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created")
        else:
            logger.info("Database initialized (use Alembic for migrations in production)")


async def close_db():
    """데이터베이스 연결 종료

    애플리케이션 종료 시 호출
    """
    await engine.dispose()
    logger.info("Database connection closed")