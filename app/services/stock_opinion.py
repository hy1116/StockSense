"""Gemini AI를 이용한 종목 전망 의견 생성"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def generate_stock_opinion(stock_name: str, **kwargs) -> Optional[str]:
    """Gemini에게 종목 전망을 물어보고 마크다운 답변 반환"""
    try:
        import asyncio
        import httpx
        from app.config import get_settings

        settings = get_settings()
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY 미설정 — AI 의견 생성 불가")
            return None

        prompt = (
            f"{stock_name} 주식에 대해 향후 주가 전망이 어떤지 10줄 이내로 분석해줘.\n\n"
        )

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-flash:generateContent?key={settings.gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 1200,
                "temperature": 0.7,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(3):
                response = await client.post(url, json=payload)
                if response.status_code != 429:
                    break
                if attempt < 2:
                    logger.warning(f"Gemini 429 Rate Limit — {attempt+1}번째 재시도 대기 3초")
                    await asyncio.sleep(3)

        if response.status_code == 429:
            logger.warning("Gemini 일일 쿼터 초과 — 내일 자정 리셋")
            return "__QUOTA_EXCEEDED__"

        response.raise_for_status()

        data = response.json()
        candidate = data["candidates"][0]
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        opinion = candidate["content"]["parts"][0]["text"].strip()

        if finish_reason == "MAX_TOKENS":
            logger.warning(f"AI 의견 토큰 한도 초과: {stock_name}")
            for sep in (".", "!", "?"):
                idx = opinion.rfind(sep)
                if idx > len(opinion) // 2:
                    opinion = opinion[: idx + 1]
                    break

        logger.info(f"AI 의견 생성 완료: {stock_name} ({len(opinion)}자, finish={finish_reason})")
        return opinion

    except Exception as e:
        logger.error(f"AI 의견 생성 실패: {e}")
        return None
