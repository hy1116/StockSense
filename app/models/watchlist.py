"""관심종목 모델"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class Watchlist(Base):
    """사용자별 관심종목"""
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    stock_code = Column(String(10), nullable=False, index=True)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    user = relationship("User", backref="watchlist_items")

    __table_args__ = (
        UniqueConstraint('user_id', 'stock_code', name='uq_watchlist_user_stock'),
    )

    def __repr__(self):
        return f"<Watchlist(id={self.id}, user_id={self.user_id}, stock_code={self.stock_code})>"
