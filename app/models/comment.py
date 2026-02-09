"""댓글 모델"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class Comment(Base):
    """주식 종목별 댓글"""
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    content = Column(Text, nullable=False)

    # 타임스탬프
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    user = relationship("User", backref="comments")

    __table_args__ = (
        Index('idx_comment_stock_created', 'stock_code', 'created_at'),
    )

    def __repr__(self):
        return f"<Comment(id={self.id}, stock_code={self.stock_code}, user_id={self.user_id})>"
