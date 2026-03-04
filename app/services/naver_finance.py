"""
네이버 금융 비공개 API 클라이언트

⚠️  비공개(undocumented) API입니다.
    - 공식 SLA 없음, 언제든 스펙이 바뀔 수 있음
    - 인증 불필요, 요청 제한 느슨함 → 개발/테스트 환경 적합
    - 실시간 데이터 (장중 기준)

사용 엔드포인트:
    https://m.stock.naver.com/api/stock/{code}/basic
"""
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# 네이버 금융 모바일 API
_BASE_URL = "https://m.stock.naver.com/api/stock"

# 브라우저처럼 보이는 헤더 (User-Agent 없으면 차단될 수 있음)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://m.stock.naver.com/",
    "Accept": "application/json",
}


class NaverFinanceClient:
    """네이버 금융 현재가 조회 클라이언트 (비동기)"""

    def __init__(self, timeout: float = 5.0):
        self._client = httpx.AsyncClient(
            headers=_HEADERS,
            timeout=timeout,
            follow_redirects=True,
        )

    async def get_current_price(self, stock_code: str) -> Optional[dict]:
        """종목 현재가 조회

        Args:
            stock_code: 6자리 종목코드 (예: "005930")

        Returns:
            {
                "stock_code": "005930",
                "stock_name": "삼성전자",
                "price": 75000,             # 현재가 (원)
                "change": 500,              # 전일 대비
                "change_rate": 0.67,        # 등락률 (%)
                "market_status": "CLOSE",   # OPEN | CLOSE | HALT
            }
            조회 실패 시 None 반환
        """
        url = f"{_BASE_URL}/{stock_code}/basic"
        try:
            resp = await self._client.get(url)
            resp.raise_for_status()
            data = resp.json()

            price_str = data.get("closePrice", "0").replace(",", "")
            change_str = data.get("compareToPreviousClosePrice", "0").replace(",", "")
            rate_str = data.get("fluctuationsRatio", "0").replace(",", "")

            return {
                "stock_code": stock_code,
                "stock_name": data.get("stockName") or data.get("itemName", ""),
                "price": int(price_str) if price_str else 0,
                "change": int(change_str) if change_str else 0,
                "change_rate": float(rate_str) if rate_str else 0.0,
                "market_status": data.get("marketStatus", "UNKNOWN"),
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"[Naver] HTTP {e.response.status_code} for {stock_code}: {e}")
        except httpx.RequestError as e:
            logger.error(f"[Naver] Request failed for {stock_code}: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"[Naver] Parse error for {stock_code}: {e}")

        return None

    async def get_prices_bulk(self, stock_codes: list[str]) -> dict[str, dict]:
        """여러 종목 현재가 일괄 조회

        Args:
            stock_codes: 종목코드 리스트

        Returns:
            { "005930": {...}, "035720": {...} }
            실패한 종목은 결과에서 제외
        """
        import asyncio

        tasks = {
            code: asyncio.create_task(self.get_current_price(code))
            for code in stock_codes
        }
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        return {
            code: result
            for code, result in zip(tasks.keys(), results)
            if isinstance(result, dict) and result is not None
        }

    async def get_chart(self, stock_code: str, period: str) -> list[dict]:
        """네이버 fchart API (1W/1M/3M/1Y 전용 — 일봉/주봉)

        1D는 KIS API intraday를 사용하므로 여기선 처리하지 않음.

        Args:
            stock_code: 6자리 종목코드
            period: '1W' | '1M' | '3M' | '1Y'

        Returns:
            [{"dt": "YYYYMMDD", "open": int, "high": int, "low": int, "close": int, "volume": int}, ...]
        """
        import ast as _ast
        from datetime import datetime, timedelta
        now = datetime.now()

        # fchart는 일봉(day)/주봉(week)만 OHLCV 완전 지원
        configs = {
            '1W': ('day',  (now - timedelta(days=14)).strftime("%Y%m%d"),  now.strftime("%Y%m%d")),
            '1M': ('day',  (now - timedelta(days=40)).strftime("%Y%m%d"),  now.strftime("%Y%m%d")),
            '3M': ('day',  (now - timedelta(days=100)).strftime("%Y%m%d"), now.strftime("%Y%m%d")),
            '1Y': ('week', (now - timedelta(days=375)).strftime("%Y%m%d"), now.strftime("%Y%m%d")),
        }
        if period not in configs:
            return []

        timeframe, start_dt, end_dt = configs[period]
        url = "https://fchart.stock.naver.com/siseJson.nhn"
        params = {
            "symbol": stock_code,
            "requestType": "1",
            "startTime": start_dt,
            "endTime": end_dt,
            "timeframe": timeframe,
        }
        # fchart는 finance.naver.com Referer 필요
        headers = {"Referer": "https://finance.naver.com/"}

        try:
            resp = await self._client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            # 응답은 single-quote 혼재 Python 리터럴 → ast로 파싱
            raw = resp.text.replace('null', 'None')
            rows = _ast.literal_eval(raw.strip())

            result = []
            for row in rows[1:]:  # 첫 행은 한글 헤더 스킵
                if not row or row[1] is None:
                    continue
                result.append({
                    "dt":     str(row[0]),  # YYYYMMDD
                    "open":   int(row[1]),
                    "high":   int(row[2]),
                    "low":    int(row[3]),
                    "close":  int(row[4]),
                    "volume": int(row[5]),
                })
            return result
        except Exception as e:
            logger.error(f"[fchart] {stock_code}/{period}: {e}")
            return []

    async def close(self):
        await self._client.aclose()


# ─── 싱글톤 ──────────────────────────────────────────────────────
_naver_client: Optional[NaverFinanceClient] = None


def get_naver_finance_client() -> NaverFinanceClient:
    """싱글톤 클라이언트 반환"""
    global _naver_client
    if _naver_client is None:
        _naver_client = NaverFinanceClient()
        logger.info("NaverFinanceClient initialized")
    return _naver_client


async def close_naver_finance_client():
    """클라이언트 종료 (앱 shutdown 시 호출)"""
    global _naver_client
    if _naver_client is not None:
        await _naver_client.close()
        _naver_client = None
        logger.info("NaverFinanceClient closed")
