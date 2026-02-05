"""뉴스 크롤링 서비스"""
import logging
import re
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from urllib.parse import quote, urljoin

KST = timezone(timedelta(hours=9))

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NewsCrawler:
    """네이버 금융 뉴스 크롤러"""

    # 네이버 금융 뉴스 URL
    NAVER_FINANCE_NEWS_URL = "https://finance.naver.com/item/news_news.naver"
    NAVER_NEWS_DETAIL_URL = "https://finance.naver.com"

    # 요청 헤더
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://finance.naver.com/',
    }

    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(headers=self.HEADERS, timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def crawl_stock_news(
        self,
        stock_code: str,
        stock_name: str = None,
        page: int = 1,
        max_pages: int = 3
    ) -> List[Dict]:
        """종목별 뉴스 크롤링

        Args:
            stock_code: 종목코드 (예: 005930)
            stock_name: 종목명 (예: 삼성전자)
            page: 시작 페이지
            max_pages: 크롤링할 최대 페이지 수

        Returns:
            뉴스 리스트
        """
        all_news = []

        for p in range(page, page + max_pages):
            try:
                news_list = await self._crawl_news_page(stock_code, stock_name, p)
                if not news_list:
                    break  # 더 이상 뉴스가 없으면 중단

                all_news.extend(news_list)
                logger.info(f"📰 Crawled page {p} for {stock_code}: {len(news_list)} news")

                # 요청 간 딜레이 (서버 부하 방지)
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Error crawling page {p} for {stock_code}: {e}")
                break

        return all_news

    async def _crawl_news_page(
        self,
        stock_code: str,
        stock_name: str,
        page: int
    ) -> List[Dict]:
        """단일 페이지 뉴스 크롤링"""
        params = {
            'code': stock_code,
            'page': page,
            'sm': 'title_entity_id.basic',
            'clusterId': ''
        }

        try:
            response = await self.client.get(self.NAVER_FINANCE_NEWS_URL, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            news_list = []

            # 뉴스 테이블에서 뉴스 추출
            news_table = soup.select('table.type5 tbody tr')

            for row in news_table:
                try:
                    # 제목이 있는 행만 처리
                    title_elem = row.select_one('td.title a')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')

                    # 상대 URL을 절대 URL로 변환
                    if url and not url.startswith('http'):
                        url = urljoin(self.NAVER_NEWS_DETAIL_URL, url)

                    # 출처
                    source_elem = row.select_one('td.info')
                    source = source_elem.get_text(strip=True) if source_elem else None

                    # 날짜
                    date_elem = row.select_one('td.date')
                    date_str = date_elem.get_text(strip=True) if date_elem else None
                    published_at = self._parse_date(date_str)

                    news_item = {
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'title': title,
                        'url': url,
                        'source': source,
                        'published_at': published_at,
                        'content': None,  # 상세 페이지에서 가져올 수 있음
                        'image_url': None,
                    }

                    news_list.append(news_item)

                except Exception as e:
                    logger.warning(f"⚠️ Error parsing news row: {e}")
                    continue

            return news_list

        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP error: {e}")
            return []

    async def crawl_news_content(self, url: str) -> Optional[Dict]:
        """뉴스 상세 내용 크롤링

        Args:
            url: 뉴스 URL

        Returns:
            뉴스 상세 내용 딕셔너리
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # 본문 추출 (네이버 금융 뉴스 구조)
            content = None
            content_elem = soup.select_one('div#news_read')
            if content_elem:
                # 불필요한 태그 제거
                for tag in content_elem.select('script, style, iframe'):
                    tag.decompose()
                content = content_elem.get_text(strip=True)

            # 이미지 URL 추출
            image_url = None
            img_elem = soup.select_one('div#news_read img')
            if img_elem:
                image_url = img_elem.get('src')

            # 기자명 추출
            author = None
            author_elem = soup.select_one('span.byline')
            if author_elem:
                author = author_elem.get_text(strip=True)

            return {
                'content': content,
                'image_url': image_url,
                'author': author
            }

        except Exception as e:
            logger.warning(f"⚠️ Error crawling news content: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """날짜 문자열 파싱 (KST timezone-aware)"""
        if not date_str:
            return None

        now = datetime.now(KST)

        try:
            # "2024.01.15 10:30" 형식
            if '.' in date_str:
                naive = datetime.strptime(date_str, "%Y.%m.%d %H:%M")
                return naive.replace(tzinfo=KST)

            # "1시간 전", "2일 전" 등
            if '분 전' in date_str:
                minutes = int(re.search(r'(\d+)', date_str).group(1))
                return now - timedelta(minutes=minutes)

            if '시간 전' in date_str:
                hours = int(re.search(r'(\d+)', date_str).group(1))
                return now - timedelta(hours=hours)

            if '일 전' in date_str:
                days = int(re.search(r'(\d+)', date_str).group(1))
                return now - timedelta(days=days)

            return None

        except Exception:
            return None


class NaverSearchNewsCrawler:
    """네이버 검색 뉴스 크롤러 (더 많은 뉴스 수집용)"""

    NAVER_SEARCH_URL = "https://search.naver.com/search.naver"

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    }

    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(headers=self.HEADERS, timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def search_news(
        self,
        stock_name: str,
        stock_code: str,
        days: int = 7,
        max_results: int = 50
    ) -> List[Dict]:
        """네이버 뉴스 검색

        Args:
            stock_name: 종목명
            stock_code: 종목코드
            days: 최근 N일 이내 뉴스
            max_results: 최대 결과 수

        Returns:
            뉴스 리스트
        """
        all_news = []
        start = 1

        while len(all_news) < max_results:
            try:
                params = {
                    'where': 'news',
                    'query': f'{stock_name} 주식',
                    'sort': 1,  # 최신순
                    'start': start,
                    'nso': f'so:dd,p:from{self._get_date_str(days)},a:all'
                }

                response = await self.client.get(self.NAVER_SEARCH_URL, params=params)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                news_items = soup.select('div.news_area')

                if not news_items:
                    break

                for item in news_items:
                    try:
                        # 제목
                        title_elem = item.select_one('a.news_tit')
                        if not title_elem:
                            continue

                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')

                        # 출처
                        source_elem = item.select_one('a.info.press')
                        source = source_elem.get_text(strip=True) if source_elem else None

                        # 요약
                        desc_elem = item.select_one('div.news_dsc')
                        content = desc_elem.get_text(strip=True) if desc_elem else None

                        # 날짜
                        date_elem = item.select_one('span.info')
                        date_str = date_elem.get_text(strip=True) if date_elem else None
                        published_at = self._parse_relative_date(date_str)

                        # 이미지
                        img_elem = item.select_one('img.thumb')
                        image_url = img_elem.get('src') if img_elem else None

                        news_item = {
                            'stock_code': stock_code,
                            'stock_name': stock_name,
                            'title': title,
                            'url': url,
                            'source': source,
                            'content': content,
                            'published_at': published_at,
                            'image_url': image_url,
                        }

                        all_news.append(news_item)

                    except Exception as e:
                        logger.warning(f"⚠️ Error parsing search result: {e}")
                        continue

                start += 10
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"❌ Error searching news: {e}")
                break

        return all_news[:max_results]

    def _get_date_str(self, days: int) -> str:
        """검색 날짜 문자열 생성"""
        date = datetime.now(KST) - timedelta(days=days)
        return date.strftime("%Y%m%d")

    def _parse_relative_date(self, date_str: str) -> Optional[datetime]:
        """상대 날짜 파싱 (KST timezone-aware)"""
        if not date_str:
            return None

        now = datetime.now(KST)

        try:
            if '분 전' in date_str:
                minutes = int(re.search(r'(\d+)', date_str).group(1))
                return now - timedelta(minutes=minutes)

            if '시간 전' in date_str:
                hours = int(re.search(r'(\d+)', date_str).group(1))
                return now - timedelta(hours=hours)

            if '일 전' in date_str:
                days = int(re.search(r'(\d+)', date_str).group(1))
                return now - timedelta(days=days)

            # "2024.01.15." 형식
            match = re.search(r'(\d{4})\.(\d{2})\.(\d{2})', date_str)
            if match:
                return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=KST)

            return None

        except Exception:
            return None


# 편의 함수
async def crawl_stock_news(stock_code: str, stock_name: str = None, max_pages: int = 3) -> List[Dict]:
    """종목 뉴스 크롤링 헬퍼 함수"""
    async with NewsCrawler() as crawler:
        return await crawler.crawl_stock_news(stock_code, stock_name, max_pages=max_pages)


async def search_stock_news(stock_name: str, stock_code: str, days: int = 7) -> List[Dict]:
    """네이버 검색 뉴스 헬퍼 함수"""
    async with NaverSearchNewsCrawler() as crawler:
        return await crawler.search_news(stock_name, stock_code, days=days)
