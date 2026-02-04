"""뉴스 API 라우터"""
import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.stock_news import StockNews
from app.schemas.news import (
    NewsResponse,
    NewsListResponse,
    NewsCrawlRequest,
    NewsCrawlResponse,
    NewsSentimentStats
)
from app.services.news_crawler import crawl_stock_news, search_stock_news

router = APIRouter(prefix="/api/news", tags=["news"])
logger = logging.getLogger(__name__)


@router.get("/{stock_code}", response_model=NewsListResponse)
async def get_stock_news(
    stock_code: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    days: Optional[int] = Query(None, ge=1, le=365, description="최근 N일 이내 뉴스만"),
    db: AsyncSession = Depends(get_db)
):
    """종목별 뉴스 목록 조회"""
    try:
        # 기본 쿼리
        query = select(StockNews).where(StockNews.stock_code == stock_code)

        # 날짜 필터
        if days:
            start_date = datetime.now() - timedelta(days=days)
            query = query.where(StockNews.published_at >= start_date)

        # 정렬 및 페이징
        query = query.order_by(StockNews.published_at.desc())
        offset = (page - 1) * page_size

        # 전체 개수
        count_query = select(func.count(StockNews.id)).where(StockNews.stock_code == stock_code)
        if days:
            count_query = count_query.where(StockNews.published_at >= start_date)

        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # 뉴스 조회
        result = await db.execute(query.offset(offset).limit(page_size))
        news_list = result.scalars().all()

        return NewsListResponse(
            news=[NewsResponse.model_validate(n) for n in news_list],
            total=total,
            page=page,
            page_size=page_size,
            has_more=(offset + len(news_list)) < total
        )

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{stock_code}/stats", response_model=NewsSentimentStats)
async def get_news_sentiment_stats(
    stock_code: str,
    days: int = Query(7, ge=1, le=365),
    db: AsyncSession = Depends(get_db)
):
    """종목별 뉴스 감성 통계"""
    try:
        start_date = datetime.now() - timedelta(days=days)

        # 전체 뉴스 수
        total_query = select(func.count(StockNews.id)).where(
            and_(
                StockNews.stock_code == stock_code,
                StockNews.published_at >= start_date
            )
        )
        total_result = await db.execute(total_query)
        total_news = total_result.scalar() or 0

        # 감성별 통계
        positive_query = select(func.count(StockNews.id)).where(
            and_(
                StockNews.stock_code == stock_code,
                StockNews.published_at >= start_date,
                StockNews.sentiment_label == 'positive'
            )
        )
        negative_query = select(func.count(StockNews.id)).where(
            and_(
                StockNews.stock_code == stock_code,
                StockNews.published_at >= start_date,
                StockNews.sentiment_label == 'negative'
            )
        )
        neutral_query = select(func.count(StockNews.id)).where(
            and_(
                StockNews.stock_code == stock_code,
                StockNews.published_at >= start_date,
                StockNews.sentiment_label == 'neutral'
            )
        )

        positive_result = await db.execute(positive_query)
        negative_result = await db.execute(negative_query)
        neutral_result = await db.execute(neutral_query)

        positive_count = positive_result.scalar() or 0
        negative_count = negative_result.scalar() or 0
        neutral_count = neutral_result.scalar() or 0

        # 평균 감성 점수
        avg_query = select(func.avg(StockNews.sentiment_score)).where(
            and_(
                StockNews.stock_code == stock_code,
                StockNews.published_at >= start_date,
                StockNews.sentiment_score.isnot(None)
            )
        )
        avg_result = await db.execute(avg_query)
        average_sentiment = avg_result.scalar()

        return NewsSentimentStats(
            stock_code=stock_code,
            total_news=total_news,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            average_sentiment=float(average_sentiment) if average_sentiment else None,
            period_days=days
        )

    except Exception as e:
        logger.error(f"Error fetching sentiment stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/crawl", response_model=NewsCrawlResponse)
async def crawl_news(
    request: NewsCrawlRequest,
    db: AsyncSession = Depends(get_db)
):
    """종목 뉴스 크롤링 (수동 실행)"""
    try:
        logger.info(f"📰 Starting news crawl for {request.stock_code}")

        # 뉴스 크롤링
        news_list = await crawl_stock_news(
            stock_code=request.stock_code,
            stock_name=request.stock_name,
            max_pages=request.max_pages
        )

        crawled_count = len(news_list)
        saved_count = 0
        duplicate_count = 0

        # DB에 저장
        for news_item in news_list:
            try:
                # 중복 체크
                existing = await db.execute(
                    select(StockNews).where(StockNews.url == news_item['url'])
                )
                if existing.scalar_one_or_none():
                    duplicate_count += 1
                    continue

                # 새 뉴스 저장
                news = StockNews(
                    stock_code=news_item['stock_code'],
                    stock_name=news_item.get('stock_name'),
                    title=news_item['title'],
                    url=news_item['url'],
                    source=news_item.get('source'),
                    content=news_item.get('content'),
                    image_url=news_item.get('image_url'),
                    published_at=news_item.get('published_at'),
                )
                db.add(news)
                saved_count += 1

            except Exception as e:
                logger.warning(f"⚠️ Error saving news: {e}")
                continue

        await db.commit()

        logger.info(f"✅ Crawl completed: {crawled_count} crawled, {saved_count} saved, {duplicate_count} duplicates")

        return NewsCrawlResponse(
            stock_code=request.stock_code,
            crawled_count=crawled_count,
            saved_count=saved_count,
            duplicate_count=duplicate_count,
            message=f"Successfully crawled {crawled_count} news, saved {saved_count} new items"
        )

    except Exception as e:
        logger.error(f"❌ Error crawling news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-crawl", response_model=NewsCrawlResponse)
async def search_and_crawl_news(
    stock_code: str,
    stock_name: str,
    days: int = Query(7, ge=1, le=30),
    db: AsyncSession = Depends(get_db)
):
    """네이버 검색을 통한 뉴스 크롤링"""
    try:
        logger.info(f"🔍 Starting search crawl for {stock_name} ({stock_code})")

        # 뉴스 검색
        news_list = await search_stock_news(
            stock_name=stock_name,
            stock_code=stock_code,
            days=days
        )

        crawled_count = len(news_list)
        saved_count = 0
        duplicate_count = 0

        # DB에 저장
        for news_item in news_list:
            try:
                # 중복 체크
                existing = await db.execute(
                    select(StockNews).where(StockNews.url == news_item['url'])
                )
                if existing.scalar_one_or_none():
                    duplicate_count += 1
                    continue

                # 새 뉴스 저장
                news = StockNews(
                    stock_code=news_item['stock_code'],
                    stock_name=news_item.get('stock_name'),
                    title=news_item['title'],
                    url=news_item['url'],
                    source=news_item.get('source'),
                    content=news_item.get('content'),
                    image_url=news_item.get('image_url'),
                    published_at=news_item.get('published_at'),
                )
                db.add(news)
                saved_count += 1

            except Exception as e:
                logger.warning(f"⚠️ Error saving news: {e}")
                continue

        await db.commit()

        return NewsCrawlResponse(
            stock_code=stock_code,
            crawled_count=crawled_count,
            saved_count=saved_count,
            duplicate_count=duplicate_count,
            message=f"Successfully searched and saved {saved_count} new items"
        )

    except Exception as e:
        logger.error(f"❌ Error search crawling news: {e}")
        raise HTTPException(status_code=500, detail=str(e))
