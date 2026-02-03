"""인증 서비스 - JWT 및 세션 관리"""
import json
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
from cryptography.fernet import Fernet
import base64
import hashlib

from app.config import get_settings
from app.services.redis_client import get_redis_client

logger = logging.getLogger(__name__)

settings = get_settings()


def _get_encryption_key() -> bytes:
    """JWT secret key를 Fernet 키로 변환"""
    key = hashlib.sha256(settings.jwt_secret_key.encode()).digest()
    return base64.urlsafe_b64encode(key)


def encrypt_credentials(credentials: dict) -> str:
    """민감한 credentials 암호화"""
    fernet = Fernet(_get_encryption_key())
    json_data = json.dumps(credentials)
    encrypted = fernet.encrypt(json_data.encode())
    return encrypted.decode()


def decrypt_credentials(encrypted_data: str) -> dict:
    """암호화된 credentials 복호화"""
    fernet = Fernet(_get_encryption_key())
    decrypted = fernet.decrypt(encrypted_data.encode())
    return json.loads(decrypted.decode())


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """JWT 액세스 토큰 생성"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """JWT 토큰 검증"""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


class SessionManager:
    """Redis 기반 세션 관리"""

    SESSION_PREFIX = "session:"

    def __init__(self):
        self.redis = get_redis_client()

    def create_session(self, account_no: str, credentials: dict) -> str:
        """새 세션 생성

        Args:
            account_no: 계좌번호
            credentials: API 인증 정보 (apiKey, apiSecret, accountNo)

        Returns:
            session_id
        """
        session_id = secrets.token_urlsafe(32)

        # credentials 암호화하여 Redis에 저장
        session_data = {
            "account_no": account_no,
            "credentials": encrypt_credentials(credentials),
            "created_at": datetime.utcnow().isoformat()
        }

        key = f"{self.SESSION_PREFIX}{session_id}"

        if self.redis.is_available():
            self.redis.set(
                key,
                json.dumps(session_data),
                expire=settings.session_expire_seconds
            )
            logger.info(f"Session created for account {account_no[:4]}****")
        else:
            logger.warning("Redis not available - session will not persist")

        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """세션 조회

        Returns:
            {account_no, credentials(복호화됨), created_at} 또는 None
        """
        if not self.redis.is_available():
            return None

        key = f"{self.SESSION_PREFIX}{session_id}"
        data = self.redis.get(key)

        if not data:
            return None

        try:
            session_data = json.loads(data)
            # credentials 복호화
            session_data["credentials"] = decrypt_credentials(session_data["credentials"])
            return session_data
        except Exception as e:
            logger.error(f"Failed to parse session data: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """세션 삭제 (로그아웃)"""
        if not self.redis.is_available():
            return False

        key = f"{self.SESSION_PREFIX}{session_id}"
        result = self.redis.delete(key)

        if result:
            logger.info(f"Session {session_id[:8]}... deleted")

        return result

    def refresh_session(self, session_id: str) -> bool:
        """세션 만료 시간 갱신"""
        if not self.redis.is_available():
            return False

        key = f"{self.SESSION_PREFIX}{session_id}"

        # Redis TTL 갱신
        try:
            return self.redis.client.expire(key, settings.session_expire_seconds)
        except Exception as e:
            logger.error(f"Failed to refresh session: {e}")
            return False


def get_session_manager() -> SessionManager:
    """SessionManager 인스턴스 반환"""
    return SessionManager()
