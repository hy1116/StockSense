"""Redis 클라이언트"""
import redis
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis 캐시 클라이언트"""

    def __init__(self, host: str, port: int, db: int = 0, password: str = ""):
        logger.info(f"Attempting to connect to Redis at {host}:{port}/{db}")
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password if password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 연결 테스트
            ping_result = self.client.ping()

            # MISCONF 에러 방지: RDB 저장 실패 시에도 쓰기 명령 허용
            try:
                self.client.config_set('stop-writes-on-bgsave-error', 'no')
                logger.info("Redis config: stop-writes-on-bgsave-error set to no")
            except Exception as config_error:
                logger.warning(f"Could not set Redis config: {config_error}")

            logger.info(f"Redis connected successfully: {host}:{port}/{db} (ping={ping_result})")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed ({host}:{port}): {e}. Running without cache.")
            self.client = None
        except redis.exceptions.ResponseError as e:
            if "MISCONF" in str(e):
                logger.warning(f"Redis MISCONF error ({host}:{port}): {e}. Attempting workaround...")
                try:
                    # MISCONF 에러 시 재시도
                    self.client = redis.Redis(
                        host=host,
                        port=port,
                        db=db,
                        password=password if password else None,
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5
                    )
                    self.client.config_set('stop-writes-on-bgsave-error', 'no')
                    self.client.ping()
                    logger.info(f"Redis MISCONF resolved: {host}:{port}/{db}")
                except Exception:
                    logger.warning("Failed to resolve MISCONF error. Running without cache.")
                    self.client = None
            else:
                logger.error(f"Redis ResponseError ({host}:{port}): {e}")
                self.client = None
        except Exception as e:
            logger.error(f"Redis initialization error ({host}:{port}): {type(e).__name__} - {e}")
            self.client = None

    def is_available(self) -> bool:
        """Redis 사용 가능 여부 확인"""
        return self.client is not None

    def get(self, key: str) -> Optional[str]:
        """값 조회"""
        if not self.is_available():
            return None

        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    def set(self, key: str, value: str, expire: int = None) -> bool:
        """값 저장

        Args:
            key: 키
            value: 값
            expire: 만료 시간 (초), None이면 무제한

        Returns:
            성공 여부
        """
        if not self.is_available():
            return False

        try:
            if expire:
                return self.client.setex(key, expire, value)
            else:
                return self.client.set(key, value)
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """값 삭제"""
        if not self.is_available():
            return False

        try:
            return self.client.delete(key) > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """키 존재 여부 확인"""
        if not self.is_available():
            return False

        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False

    def ttl(self, key: str) -> int:
        """남은 만료 시간 조회 (초)

        Returns:
            남은 시간 (초), -1: 만료시간 없음, -2: 키 없음
        """
        if not self.is_available():
            return -2

        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            return -2


@lru_cache()
def get_redis_client() -> RedisClient:
    """Redis 클라이언트 싱글톤"""
    from app.config import get_settings

    settings = get_settings()

    return RedisClient(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        password=settings.redis_password
    )
