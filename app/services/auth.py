"""인증 서비스 - JWT, 세션, 비밀번호 관리"""
import json
import secrets
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import jwt, JWTError
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import base64
import hashlib

from app.config import get_settings
from app.services.redis_client import get_redis_client

logger = logging.getLogger(__name__)

settings = get_settings()

KST = timezone(timedelta(hours=9))

# 비밀번호 해싱
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _get_encryption_key() -> bytes:
    """JWT secret key를 Fernet 키로 변환"""
    key = hashlib.sha256(settings.jwt_secret_key.encode()).digest()
    return base64.urlsafe_b64encode(key)


def encrypt_data(data: str) -> str:
    """민감한 데이터 암호화"""
    fernet = Fernet(_get_encryption_key())
    encrypted = fernet.encrypt(data.encode())
    return encrypted.decode()


def decrypt_data(encrypted_data: str) -> str:
    """암호화된 데이터 복호화"""
    fernet = Fernet(_get_encryption_key())
    decrypted = fernet.decrypt(encrypted_data.encode())
    return decrypted.decode()


def _prepare_password(password: str) -> str:
    """bcrypt 72바이트 제한을 위해 패스워드 전처리 (SHA256 + base64)"""
    # SHA256 해시 후 base64 인코딩 (44자로 고정)
    password_hash = hashlib.sha256(password.encode('utf-8')).digest()
    return base64.b64encode(password_hash).decode('utf-8')


def hash_password(password: str) -> str:
    """비밀번호 해싱"""
    prepared = _prepare_password(password)
    return pwd_context.hash(prepared)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호 검증"""
    prepared = _prepare_password(plain_password)
    return pwd_context.verify(prepared, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """JWT 액세스 토큰 생성"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=settings.jwt_expire_minutes)

    print(f"expire: {expire}")

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

    def create_session(self, user_id: int, username: str, nickname: str = None, account_no: str = None) -> str:
        """새 세션 생성"""
        session_id = secrets.token_urlsafe(32)

        session_data = {
            "user_id": user_id,
            "username": username,
            "nickname": nickname,
            "account_no": account_no,
            "created_at": datetime.now().isoformat()
        }
        print(f"created_at: {datetime.now().isoformat()}")

        key = f"{self.SESSION_PREFIX}{session_id}"

        if self.redis.is_available():
            self.redis.set(
                key,
                json.dumps(session_data),
                expire=settings.session_expire_seconds
            )
            logger.info(f"Session created for user {username}")
        else:
            logger.warning("Redis not available - session will not persist")

        return session_id

    def get_session(self, session_id: str) -> Optional[dict]:
        """세션 조회"""
        if not self.redis.is_available():
            return None

        key = f"{self.SESSION_PREFIX}{session_id}"
        data = self.redis.get(key)

        if not data:
            return None

        try:
            return json.loads(data)
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

        try:
            return self.redis.client.expire(key, settings.session_expire_seconds)
        except Exception as e:
            logger.error(f"Failed to refresh session: {e}")
            return False


def get_session_manager() -> SessionManager:
    """SessionManager 인스턴스 반환"""
    return SessionManager()
