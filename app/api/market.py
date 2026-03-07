"""글로벌 매크로 지표 API (KOSPI, 환율, 유가 등)"""
import ast
import json
import logging
import asyncio
from datetime import datetime, timedelta

import requests

from fastapi import APIRouter

from app.services.redis_client import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/market", tags=["market"])

CACHE_KEY = "market:macro"
CACHE_TTL = 900  # 15분

_NAVER_FCHART = "https://fchart.stock.naver.com/siseJson.nhn"
_NAVER_HEADERS = {"Referer": "https://finance.naver.com/"}
_YAHOO_V8 = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d"
_YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0"}

INDICATORS = [
    {"symbol": "KOSPI",  "label": "KOSPI",   "type": "index",    "unit": "",  "source": "naver"},
    {"symbol": "KOSDAQ", "label": "KOSDAQ",  "type": "index",    "unit": "",  "source": "naver"},
    {"symbol": "USDKRW", "label": "USD/KRW", "type": "exchange", "unit": "원", "source": "frankfurter"},
    {"symbol": "CL=F",   "label": "WTI",     "type": "commodity","unit": "$", "source": "yahoo"},
    {"symbol": "GC=F",   "label": "금",       "type": "commodity","unit": "$", "source": "yahoo"},
    {"symbol": "^GSPC",  "label": "S&P500",  "type": "index",    "unit": "",  "source": "yahoo"},
    {"symbol": "^IXIC",  "label": "NASDAQ",  "type": "index",    "unit": "",  "source": "yahoo"},
    {"symbol": "^VIX",   "label": "VIX",     "type": "index",    "unit": "",  "source": "yahoo"},
]


def _naver_close_prices(symbol: str) -> tuple[float | None, float | None]:
    """Naver fchart에서 종가 2개 (최신, 전일) 반환"""
    end = datetime.today()
    start = end - timedelta(days=10)
    params = {
        "symbol": symbol,
        "requestType": "1",
        "startTime": start.strftime("%Y%m%d"),
        "endTime": end.strftime("%Y%m%d"),
        "timeframe": "day",
    }
    r = requests.get(_NAVER_FCHART, params=params, headers=_NAVER_HEADERS, timeout=8)
    r.raise_for_status()
    rows = ast.literal_eval(r.text.replace("null", "None").strip())
    closes = [float(row[4]) for row in rows[1:] if row and row[4] is not None and row[4] > 0]
    if len(closes) >= 2:
        return closes[-1], closes[-2]
    if len(closes) == 1:
        return closes[-1], None
    return None, None


def _yahoo_close_prices(symbol: str) -> tuple[float | None, float | None]:
    """Yahoo Finance v8 API에서 종가 2개 반환"""
    r = requests.get(
        _YAHOO_V8.format(symbol=symbol),
        headers=_YAHOO_HEADERS,
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()
    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    closes = [float(c) for c in closes if c is not None]
    if len(closes) >= 2:
        return closes[-1], closes[-2]
    if len(closes) == 1:
        return closes[-1], None
    return None, None


def _frankfurter_usdkrw() -> tuple[float | None, float | None]:
    """frankfurter.app에서 USD/KRW 현재가 및 전일가 반환"""
    r = requests.get(
        "https://api.frankfurter.app/latest?from=USD&to=KRW",
        timeout=8,
    )
    r.raise_for_status()
    price = float(r.json()["rates"]["KRW"])
    # 전일가: 날짜별 조회
    yesterday = (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d")
    r2 = requests.get(
        f"https://api.frankfurter.app/{yesterday}?from=USD&to=KRW",
        timeout=8,
    )
    prev = float(r2.json()["rates"]["KRW"]) if r2.ok else None
    return price, prev


def _fetch_macro() -> list:
    """멀티소스 매크로 지표 조회 (Naver / Yahoo v8 / Frankfurter)"""
    result = []
    for meta in INDICATORS:
        sym = meta["symbol"]
        price, prev_close = None, None
        try:
            if meta["source"] == "naver":
                price, prev_close = _naver_close_prices(sym)
            elif meta["source"] == "yahoo":
                price, prev_close = _yahoo_close_prices(sym)
            elif meta["source"] == "frankfurter":
                price, prev_close = _frankfurter_usdkrw()

            if price is None or prev_close is None or prev_close == 0:
                change, change_pct = 0.0, 0.0
            else:
                change = round(price - prev_close, 4)
                change_pct = round(change / prev_close * 100, 2)

            result.append({
                "symbol": sym,
                "label": meta["label"],
                "type": meta["type"],
                "unit": meta["unit"],
                "price": round(price, 2) if price else None,
                "change": change,
                "change_pct": change_pct,
            })
        except Exception as e:
            logger.warning(f"매크로 지표 조회 실패 ({sym}): {e}")
            result.append({
                "symbol": sym,
                "label": meta["label"],
                "type": meta["type"],
                "unit": meta["unit"],
                "price": None,
                "change": 0.0,
                "change_pct": 0.0,
            })

    return result


@router.get("/macro")
async def get_macro_indicators():
    """글로벌 매크로 지표 조회 (KOSPI, KOSDAQ, USD/KRW, WTI, 금, S&P500, NASDAQ, VIX)"""
    redis = get_redis_client()

    cached = redis.get(CACHE_KEY)
    if cached:
        return json.loads(cached)

    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, _fetch_macro)

    redis.set(CACHE_KEY, json.dumps(data), expire=CACHE_TTL)
    return data
