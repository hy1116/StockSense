"""User 모델"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.sql import func

from app.database import Base


class User(Base):
    """사용자 모델 - KIS API 인증정보 포함"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    nickname = Column(String(30), unique=True, nullable=False, index=True)  # 닉네임
    password_hash = Column(String(255), nullable=False)

    # KIS API 정보 (암호화하여 저장)
    kis_api_key = Column(Text, nullable=True)  # 암호화된 API Key
    kis_api_secret = Column(Text, nullable=True)  # 암호화된 API Secret
    kis_account_no = Column(String(20), nullable=True)  # 계좌번호
    kis_account_product_code = Column(String(10), default="01")

    # 상태
    is_active = Column(Boolean, default=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username})>"
