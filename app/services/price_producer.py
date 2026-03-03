"""
주가 폴링 Producer

동작 흐름:
    1. DB에서 활성화된 알림의 종목 코드를 중복 없이 조회
    2. 네이버 금융 API로 현재가 일괄 조회 (비동기 병렬)
    3. Kafka topic `stock-price`에 발행
    4. POLL_INTERVAL 초마다 반복

가격 조회 소스: 네이버 금융 비공개 API
    - 인증 불필요, 요청 제한 느슨함
    - 실시간 데이터 (장중 기준)
    - KIS API는 기존 거래 기능(매수/매도 등)에서 그대로 사용
"""
import asyncio
import logging
from datetime import datetime

from sqlalchemy import select

from app.config import get_settings
from app.database import AsyncSessionLocal
from app.models.price_alert import PriceAlert
from app.services.kafka_client import get_kafka_producer
from app.services.naver_finance import get_naver_finance_client

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_price_producer():
    """주가 폴링 루프 (앱 시작 시 백그라운드 태스크로 실행)

    Kafka 연결 실패 시 지수 백오프로 재시도 (5s → 10s → 20s … 최대 120s).
    정상 동작 중에는 POLL_INTERVAL마다 실행.
    """
    logger.info(
        f"Price producer started (Naver Finance). "
        f"Polling every {settings.kafka_poll_interval_seconds}s"
    )
    retry_delay = 5
    max_retry_delay = 120

    while True:
        try:
            await _fetch_and_publish()
            retry_delay = 5  # 성공하면 백오프 리셋
            await asyncio.sleep(settings.kafka_poll_interval_seconds)

        except asyncio.CancelledError:
            logger.info("Price producer cancelled")
            break

        except Exception as e:
            logger.warning(
                f"Price producer error (retry in {retry_delay}s): {e}"
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)


async def _fetch_and_publish():
    """활성 알림 종목의 현재가를 네이버 금융에서 조회해 Kafka에 발행"""

    # 1. 활성화된 알림의 고유 종목 코드 목록 조회
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(PriceAlert.stock_code)
            .where(
                PriceAlert.is_active == True,
                PriceAlert.is_triggered == False,
            )
            .distinct()
        )
        stock_codes: list[str] = [row[0] for row in result.fetchall()]

    if not stock_codes:
        logger.debug("No active alerts – skipping price poll")
        return

    # 2. 네이버 금융에서 현재가 일괄 조회 (병렬)
    naver = get_naver_finance_client()
    price_map = await naver.get_prices_bulk(stock_codes)

    if not price_map:
        logger.warning("Naver Finance returned no data")
        return

    # 3. 조회된 종목을 Kafka에 발행
    producer = await get_kafka_producer()
    published = 0

    for stock_code, price_data in price_map.items():
        try:
            price = price_data.get("price", 0)
            if price <= 0:
                logger.warning(f"[Producer] {stock_code}: invalid price {price}, skip")
                continue

            message = {
                "stock_code": stock_code,
                "stock_name": price_data.get("stock_name", ""),
                "price": price,
                "change": price_data.get("change", 0),
                "change_rate": price_data.get("change_rate", 0.0),
                "market_status": price_data.get("market_status", "UNKNOWN"),
                "timestamp": datetime.now().isoformat(),
            }

            await producer.send(
                settings.kafka_stock_price_topic,
                key=stock_code,
                value=message,
            )
            published += 1
            logger.info(
                f"[Producer] {stock_code}({price_data.get('stock_name', '')}) "
                f"→ {price:,}원 ({price_data.get('change_rate', 0):+.2f}%) published"
            )

        except Exception as e:
            logger.error(f"[Producer] Failed to publish {stock_code}: {e}")

    logger.info(
        f"[Producer] Poll complete: {published}/{len(stock_codes)} stocks published"
    )
