"""주식 뉴스 모델"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index
from sqlalchemy.sql import func

from app.database import Base


class StockNews(Base):
    """종목별 뉴스 테이블"""
    __tablename__ = "stock_news"

    id = Column(Integer, primary_key=True, index=True)

    # 종목 정보
    stock_code = Column(String(20), nullable=False, index=True)
    stock_name = Column(String(100), nullable=True)

    # 뉴스 정보
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)  # 뉴스 본문 (요약 또는 전문)
    summary = Column(Text, nullable=True)  # AI 요약 (나중에 사용)

    # 메타 정보
    source = Column(String(100), nullable=True)  # 뉴스 출처 (예: 한경, 매경 등)
    author = Column(String(100), nullable=True)  # 기자명
    url = Column(String(1000), nullable=False, unique=True)  # 뉴스 URL (중복 방지용)
    image_url = Column(String(1000), nullable=True)  # 썸네일 이미지

    # 감성 분석 (나중에 ML 학습용)
    sentiment_score = Column(Integer, nullable=True)  # -100 ~ 100 (부정 ~ 긍정)
    sentiment_label = Column(String(20), nullable=True)  # positive, negative, neutral

    # 시간 정보
    published_at = Column(DateTime(timezone=True), nullable=True)  # 뉴스 발행일
    crawled_at = Column(DateTime(timezone=True), server_default=func.now())  # 크롤링 시간

    # 상태
    is_processed = Column(Boolean, default=False)  # 감성분석 완료 여부
    is_used_for_training = Column(Boolean, default=False)  # 학습에 사용됨 여부

    __table_args__ = (
        Index('idx_stock_news_code_date', 'stock_code', 'published_at'),
        Index('idx_stock_news_sentiment', 'sentiment_label'),
        Index('idx_stock_news_crawled', 'crawled_at'),
    )

    def __repr__(self):
        return f"<StockNews(id={self.id}, stock_code={self.stock_code}, title={self.title[:30]}...)>"
