"""
주가 알림 Consumer

동작 흐름:
    1. Kafka topic `stock-price` 구독
    2. 메시지 수신 시 해당 종목의 활성 알림 전체를 DB에서 조회
    3. 조건 체크 (above / below)
    4. 조건 충족 시:
        - DB 업데이트 (is_triggered=True, triggered_price, triggered_at)
        - topic `price-alert-triggered`에 이벤트 발행
"""
import asyncio
import logging
from datetime import datetime

from sqlalchemy import select

from app.config import get_settings
from app.database import AsyncSessionLocal
from app.models.price_alert import PriceAlert
from app.services.kafka_client import get_kafka_producer, make_consumer

logger = logging.getLogger(__name__)
settings = get_settings()


async def run_alert_consumer():
    """알림 컨슈머 루프 (앱 시작 시 백그라운드 태스크로 실행)

    Kafka 연결 실패 시 지수 백오프로 재시도 (5s → 10s → 20s … 최대 120s).
    연결 성공 후 메시지 루프 중 장애 발생 시 재연결 시도.
    """
    retry_delay = 5
    max_retry_delay = 120

    while True:
        consumer = make_consumer(
            topics=[settings.kafka_stock_price_topic],
            group_id="price-alert-group",
        )
        try:
            await consumer.start()
            logger.info(
                f"Alert consumer started → topic: {settings.kafka_stock_price_topic}"
            )
            retry_delay = 5  # 연결 성공 시 백오프 리셋

            try:
                async for msg in consumer:
                    try:
                        await _check_and_trigger(msg.value)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Alert check error: {e}", exc_info=True)

            except asyncio.CancelledError:
                logger.info("Alert consumer cancelled")
                return  # 앱 종료 시 완전히 종료

            finally:
                await consumer.stop()
                logger.info("Alert consumer stopped")

        except asyncio.CancelledError:
            return  # 앱 종료 시 완전히 종료

        except Exception as e:
            logger.warning(
                f"Alert consumer failed to connect (retry in {retry_delay}s): {e}"
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)


async def _check_and_trigger(price_data: dict):
    """수신된 가격 데이터로 알림 조건을 체크하고 발동 처리"""
    stock_code: str = price_data.get("stock_code", "")
    current_price: int = price_data.get("price", 0)

    if not stock_code or current_price <= 0:
        return

    async with AsyncSessionLocal() as session:
        # 해당 종목의 활성 알림 목록 조회
        result = await session.execute(
            select(PriceAlert).where(
                PriceAlert.stock_code == stock_code,
                PriceAlert.is_active == True,
                PriceAlert.is_triggered == False,
            )
        )
        alerts: list[PriceAlert] = result.scalars().all()

        if not alerts:
            return

        producer = await get_kafka_producer()
        triggered_count = 0

        for alert in alerts:
            # 조건 체크
            is_triggered = (
                (alert.condition == "above" and current_price >= alert.target_price)
                or (alert.condition == "below" and current_price <= alert.target_price)
            )

            if not is_triggered:
                continue

            # DB 업데이트
            alert.is_triggered = True
            alert.triggered_price = current_price
            alert.triggered_at = datetime.now()
            triggered_count += 1

            # 알림 발동 이벤트 발행
            await producer.send(
                settings.kafka_alert_triggered_topic,
                key=str(alert.user_id),
                value={
                    "alert_id": alert.id,
                    "user_id": alert.user_id,
                    "stock_code": stock_code,
                    "stock_name": alert.stock_name or price_data.get("stock_name", ""),
                    "condition": alert.condition,
                    "target_price": alert.target_price,
                    "triggered_price": current_price,
                    "triggered_at": datetime.now().isoformat(),
                },
            )
            logger.info(
                f"[Consumer] 알림 발동! "
                f"user={alert.user_id}, {stock_code} "
                f"{alert.condition} {alert.target_price:,}원 "
                f"(현재가: {current_price:,}원)"
            )

        if triggered_count > 0:
            await session.commit()
            logger.info(f"[Consumer] {stock_code}: {triggered_count}건 알림 발동 처리 완료")
