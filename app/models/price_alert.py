"""주가 알림 모델"""
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class PriceAlert(Base):
    """사용자 주가 알림 설정

    condition:
        - "above": 현재가가 target_price 이상이면 알림
        - "below": 현재가가 target_price 이하이면 알림
    """
    __tablename__ = "price_alerts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)   # 종목코드 (예: 005930)
    stock_name = Column(String(100), nullable=True)               # 종목명 (예: 삼성전자)

    # 알림 조건
    condition = Column(String(10), nullable=False)    # "above" | "below"
    target_price = Column(Float, nullable=False)      # 목표 가격

    # 상태
    is_active = Column(Boolean, default=True, nullable=False)       # 알림 활성화 여부
    is_triggered = Column(Boolean, default=False, nullable=False)   # 이미 발동됐는지 여부

    # 발동 정보
    triggered_price = Column(Float, nullable=True)                          # 실제 발동된 가격
    triggered_at = Column(DateTime(timezone=True), nullable=True)           # 발동 시각

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    user = relationship("User", backref="price_alerts")

    __table_args__ = (
        # 활성 알림을 종목 코드로 빠르게 조회하기 위한 복합 인덱스
        Index("idx_alert_stock_active", "stock_code", "is_active", "is_triggered"),
        Index("idx_alert_user", "user_id", "is_active"),
    )

    def __repr__(self):
        return (
            f"<PriceAlert(id={self.id}, user={self.user_id}, "
            f"stock={self.stock_code}, {self.condition} {self.target_price:,})>"
        )
