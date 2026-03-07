"""글로벌 매크로 지표 API (KOSPI, 환율, 유가 등)"""
import json
import logging
import asyncio
from typing import Optional

from fastapi import APIRouter

from app.services.redis_client import get_redis_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/market", tags=["market"])

CACHE_KEY = "market:macro"
CACHE_TTL = 900  # 15분

INDICATORS = [
    {"symbol": "^KS11",  "label": "KOSPI",   "type": "index",    "unit": ""},
    {"symbol": "^KQ11",  "label": "KOSDAQ",  "type": "index",    "unit": ""},
    {"symbol": "KRW=X",  "label": "USD/KRW", "type": "exchange", "unit": "원"},
    {"symbol": "CL=F",   "label": "WTI",     "type": "commodity","unit": "$"},
    {"symbol": "GC=F",   "label": "금",       "type": "commodity","unit": "$"},
    {"symbol": "^GSPC",  "label": "S&P500",  "type": "index",    "unit": ""},
    {"symbol": "^IXIC",  "label": "NASDAQ",  "type": "index",    "unit": ""},
    {"symbol": "^VIX",   "label": "VIX",     "type": "index",    "unit": ""},
]


def _fetch_macro() -> list:
    """yfinance로 매크로 지표 조회 (동기)"""
    import yfinance as yf

    symbols = [i["symbol"] for i in INDICATORS]
    tickers = yf.Tickers(" ".join(symbols))

    result = []
    for meta in INDICATORS:
        sym = meta["symbol"]
        try:
            info = tickers.tickers[sym].fast_info
            price = info.last_price
            prev_close = info.previous_close

            if price is None or prev_close is None or prev_close == 0:
                change = 0.0
                change_pct = 0.0
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
