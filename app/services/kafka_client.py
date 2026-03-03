"""
Kafka 클라이언트 - Producer / Consumer 싱글톤 관리

토픽:
    stock-price          : KIS API에서 폴링한 실시간 주가 데이터
    price-alert-triggered: 알림 조건이 발동된 이벤트
"""
import json
import logging
from typing import Optional

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ─── 글로벌 Producer 인스턴스 ─────────────────────────────────────
_producer: Optional[AIOKafkaProducer] = None


async def get_kafka_producer() -> AIOKafkaProducer:
    """싱글톤 Producer 반환 (없으면 시작)

    start() 실패 시 _producer를 None으로 되돌려 다음 호출에서 재시도 가능하게 함.
    """
    global _producer
    if _producer is None:
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            acks="all",
            retry_backoff_ms=500,
            request_timeout_ms=30_000,
        )
        await producer.start()  # 실패 시 예외 전파, _producer는 None 유지
        _producer = producer
        logger.info(f"Kafka producer started → {settings.kafka_bootstrap_servers}")
    return _producer


async def close_kafka_producer():
    """Producer 종료 (앱 shutdown 시 호출)"""
    global _producer
    if _producer is not None:
        await _producer.stop()
        _producer = None
        logger.info("Kafka producer stopped")


# ─── Consumer 팩토리 ──────────────────────────────────────────────
def make_consumer(topics: list[str], group_id: str) -> AIOKafkaConsumer:
    """Consumer 인스턴스 생성

    Args:
        topics  : 구독할 토픽 리스트
        group_id: Consumer Group ID (같은 그룹끼리 파티션 분배)
    """
    return AIOKafkaConsumer(
        *topics,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",   # 새 메시지부터 읽기 (과거 무시)
        enable_auto_commit=True,
        session_timeout_ms=30_000,
        heartbeat_interval_ms=10_000,
    )
