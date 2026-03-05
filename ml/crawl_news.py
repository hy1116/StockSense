"""뉴스 크롤링 배치 스크립트

1시간마다 실행하여 종목별 뉴스를 수집합니다.

Usage:
    # 기본 실행 (관심 종목, 최근 1시간 뉴스만)
    python -m ml.crawl_news

    # 모든 수집 대상 종목
    python -m ml.crawl_news --all

    # 특정 종목만
    python -m ml.crawl_news --stock-code 005930 --stock-name 삼성전자

    # 최근 N시간 뉴스
    python -m ml.crawl_news --hours 2

    # 최근 N일 뉴스 (네이버 검색 사용)
    python -m ml.crawl_news --days 7 --use-search

    # Transformer 감성분석 사용
    python -m ml.crawl_news --use-transformer
"""
import sys
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

# 프로젝트 루트 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Windows 콘솔 인코딩
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


async def crawl_news_for_stocks(
    stocks: list,
    max_pages: int = 3,
    use_search: bool = False,
    days: int = 7,
    hours: int = None,
    use_transformer: bool = False
):
    """여러 종목의 뉴스 크롤링

    Args:
        stocks: 종목 리스트 [{'code': '005930', 'name': '삼성전자'}, ...]
        max_pages: 크롤링할 최대 페이지 수
        use_search: 네이버 검색 사용 여부
        days: 최근 N일 (use_search=True일 때 사용)
        hours: 최근 N시간 내 뉴스만 필터링 (None이면 필터링 안함)
    """
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from app.models.stock_news import StockNews
    from app.services.news_crawler import crawl_stock_news, search_stock_news
    from app.services.news_sentiment import NewsSentimentAnalyzer

    load_dotenv()

    # 감성 분석기 초기화
    sentiment_analyzer = NewsSentimentAnalyzer(use_transformer=use_transformer)

    # DB 연결
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocksense")
    db_url = db_url.replace("+asyncpg", "")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    # 시간 필터 기준
    time_threshold = None
    if hours:
        KST = timezone(timedelta(hours=9))
        time_threshold = datetime.now(KST) - timedelta(hours=hours)

    total_crawled = 0
    total_saved = 0
    total_duplicate = 0
    total_filtered = 0

    print(f"\n{'='*60}")
    print(f"📰 News Crawling Batch Started")
    print(f"   Stocks: {len(stocks)}")
    print(f"   Method: {'Search' if use_search else 'Finance News'}")
    print(f"   Max Pages: {max_pages}")
    if hours:
        print(f"   Time Filter: Last {hours} hour(s)")
    print(f"{'='*60}\n")

    for stock in stocks:
        stock_code = stock['code']
        stock_name = stock['name']

        print(f"\n🔍 Crawling news for {stock_name} ({stock_code})...")

        try:
            # 크롤링 방식 선택
            if use_search:
                news_list = await search_stock_news(
                    stock_name=stock_name,
                    stock_code=stock_code,
                    days=days
                )
            else:
                news_list = await crawl_stock_news(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    max_pages=max_pages
                )

            crawled_count = len(news_list)
            saved_count = 0
            duplicate_count = 0
            filtered_count = 0

            # DB에 저장
            with Session() as session:
                for news_item in news_list:
                    try:
                        # 시간 필터링 (hours 옵션이 있을 때)
                        if time_threshold and news_item.get('published_at'):
                            if news_item['published_at'] < time_threshold:
                                filtered_count += 1
                                continue

                        # 중복 체크
                        existing = session.execute(
                            select(StockNews).where(StockNews.url == news_item['url'])
                        ).scalar_one_or_none()

                        if existing:
                            duplicate_count += 1
                            continue

                        # 감성 분석 수행
                        analysis_text = news_item['title']
                        if news_item.get('content'):
                            analysis_text += " " + news_item['content']
                        sentiment_score, sentiment_label = sentiment_analyzer.analyze(analysis_text)

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
                            sentiment_score=sentiment_score,
                            sentiment_label=sentiment_label,
                            is_processed=True,
                        )
                        session.add(news)
                        saved_count += 1

                    except Exception as e:
                        print(f"   ⚠️ Error saving: {e}")
                        continue

                session.commit()

            total_crawled += crawled_count
            total_saved += saved_count
            total_duplicate += duplicate_count
            total_filtered += filtered_count

            print(f"   ✅ Crawled: {crawled_count}, Saved: {saved_count}, Duplicates: {duplicate_count}, Filtered(old): {filtered_count}")

            # 요청 간 딜레이
            await asyncio.sleep(1)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"📊 Crawling Summary")
    print(f"   Total Crawled: {total_crawled}")
    print(f"   Total Saved: {total_saved}")
    print(f"   Total Duplicates: {total_duplicate}")
    print(f"   Total Filtered (old): {total_filtered}")
    print(f"{'='*60}\n")

    return {
        'total_crawled': total_crawled,
        'total_saved': total_saved,
        'total_duplicate': total_duplicate,
        'total_filtered': total_filtered
    }


def get_collection_stocks():
    """수집 대상 종목 목록 조회"""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, select, text
    from sqlalchemy.orm import sessionmaker

    load_dotenv()

    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocksense")
    db_url = db_url.replace("+asyncpg", "")
    engine = create_engine(db_url)

    stocks = []

    with engine.connect() as conn:
        # collection_stocks 테이블에서 조회
        result = conn.execute(text("""
            SELECT stock_code, stock_name
            FROM stocks
            WHERE is_active = true
            ORDER BY stock_name
        """))

        for row in result:
            stocks.append({
                'code': row[0],
                'name': row[1]
            })

    return stocks


def get_default_stocks():
    """기본 주요 종목 목록"""
    return [
        {'code': '005930', 'name': '삼성전자'},
        {'code': '000660', 'name': 'SK하이닉스'},
        {'code': '035420', 'name': 'NAVER'},
        {'code': '035720', 'name': '카카오'},
        {'code': '005380', 'name': '현대차'},
        {'code': '006400', 'name': '삼성SDI'},
        {'code': '051910', 'name': 'LG화학'},
        {'code': '003670', 'name': '포스코퓨처엠'},
        {'code': '005490', 'name': 'POSCO홀딩스'},
        {'code': '055550', 'name': '신한지주'},
    ]


async def main():
    parser = argparse.ArgumentParser(description='Stock News Crawler')
    parser.add_argument('--all', action='store_true',
                        help='Crawl all collection stocks')
    parser.add_argument('--stock-code', '-c', type=str,
                        help='Specific stock code')
    parser.add_argument('--stock-name', '-n', type=str,
                        help='Specific stock name')
    parser.add_argument('--max-pages', '-p', type=int, default=2,
                        help='Max pages to crawl (default: 2)')
    parser.add_argument('--use-search', '-s', action='store_true',
                        help='Use Naver search instead of finance news')
    parser.add_argument('--days', '-d', type=int, default=7,
                        help='Days to search (only with --use-search)')
    parser.add_argument('--hours', '-H', type=int, default=1,
                        help='Filter news within last N hours (default: 1, 0=no filter)')
    parser.add_argument('--use-transformer', action='store_true',
                        help='Use Transformer model for sentiment analysis (default: keyword-based)')

    args = parser.parse_args()

    # 종목 목록 결정
    if args.stock_code:
        if not args.stock_name:
            print("Error: --stock-name is required with --stock-code")
            return
        stocks = [{'code': args.stock_code, 'name': args.stock_name}]
    elif args.all:
        print("📋 Loading collection stocks from DB...")
        stocks = get_collection_stocks()
        if not stocks:
            print("⚠️ No collection stocks found, using defaults")
            stocks = get_default_stocks()
    else:
        stocks = get_default_stocks()

    print(f"📌 Target stocks: {len(stocks)}")
    for s in stocks[:5]:
        print(f"   - {s['name']} ({s['code']})")
    if len(stocks) > 5:
        print(f"   ... and {len(stocks) - 5} more")

    # 시간 필터 (0이면 필터링 안함)
    hours_filter = args.hours if args.hours > 0 else None

    # 크롤링 실행
    result = await crawl_news_for_stocks(
        stocks=stocks,
        max_pages=args.max_pages,
        use_search=args.use_search,
        days=args.days,
        hours=hours_filter,
        use_transformer=args.use_transformer
    )

    print(f"\n✅ News crawling completed at {datetime.now().isoformat()}")


if __name__ == '__main__':
    from ml.logger import TeeStdout
    with TeeStdout("crawl_news"):
        asyncio.run(main())
