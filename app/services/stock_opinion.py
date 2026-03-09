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


async def generate_portfolio_opinion(holdings: list) -> str | None:
    """보유 포트폴리오 전체에 대한 AI 종합 의견"""
    try:
        import asyncio
        import httpx
        from app.config import get_settings

        settings = get_settings()
        if not settings.gemini_api_key:
            return None

        lines = []
        for h in holdings:
            name = h.get("stock_name", "")
            profit = h.get("profit_rate", 0)
            qty = h.get("quantity", 0)
            avg = h.get("avg_price", 0)
            lines.append(f"- {name}: {qty}주 보유, 평균단가 {avg:,}원, 수익률 {profit:+.2f}%")
        holdings_text = "\n".join(lines)

        prompt = (
            f"다음은 현재 보유 중인 주식 포트폴리오입니다:\n\n{holdings_text}\n\n"
            f"이 포트폴리오에 대해 마크다운 형식으로 분석해줘.\n\n"
            f"아래 형식을 따라줘:\n\n"
            f"### 포트폴리오 구성 분석\n(섹터 분산, 집중도 등 2~3문장)\n\n"
            f"### 리스크 요인\n- 항목1\n- 항목2\n\n"
            f"### 관리 제안\n(비중 조정, 주의할 종목 등 2~3문장)\n\n"
            f"> ※ 이 내용은 참고용이며 투자 권유가 아닙니다."
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
                    await asyncio.sleep(3)

        if response.status_code == 429:
            return "__QUOTA_EXCEEDED__"

        response.raise_for_status()
        data = response.json()
        opinion = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        logger.info(f"포트폴리오 AI 의견 생성 완료 ({len(opinion)}자)")
        return opinion

    except Exception as e:
        logger.error(f"포트폴리오 AI 의견 생성 실패: {e}")
        return None
