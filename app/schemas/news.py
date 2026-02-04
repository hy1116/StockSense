"""뉴스 스키마"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class NewsBase(BaseModel):
    """뉴스 기본 스키마"""
    stock_code: str
    stock_name: Optional[str] = None
    title: str
    url: str
    source: Optional[str] = None
    author: Optional[str] = None
    content: Optional[str] = None
    summary: Optional[str] = None
    image_url: Optional[str] = None
    published_at: Optional[datetime] = None


class NewsCreate(NewsBase):
    """뉴스 생성 스키마"""
    pass


class NewsResponse(NewsBase):
    """뉴스 응답 스키마"""
    id: int
    sentiment_score: Optional[int] = None
    sentiment_label: Optional[str] = None
    crawled_at: Optional[datetime] = None
    is_processed: bool = False

    class Config:
        from_attributes = True


class NewsListResponse(BaseModel):
    """뉴스 목록 응답"""
    news: List[NewsResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class NewsCrawlRequest(BaseModel):
    """뉴스 크롤링 요청"""
    stock_code: str
    stock_name: Optional[str] = None
    max_pages: int = 3


class NewsCrawlResponse(BaseModel):
    """뉴스 크롤링 응답"""
    stock_code: str
    crawled_count: int
    saved_count: int
    duplicate_count: int
    message: str


class NewsSentimentStats(BaseModel):
    """뉴스 감성 분석 통계"""
    stock_code: str
    total_news: int
    positive_count: int
    negative_count: int
    neutral_count: int
    average_sentiment: Optional[float] = None
    period_days: int
