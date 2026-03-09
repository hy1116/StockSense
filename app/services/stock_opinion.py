"""Gemini AI를 이용한 종목 투자의견 자연어 생성"""
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def _build_prompt(stock_name: str, prediction: Dict, news_list: List[Dict]) -> str:
    current = prediction.get("current_price", 0)
    predicted = prediction.get("predicted_price", 0)
    trend = prediction.get("trend", "")
    rec = prediction.get("recommendation", "")
    conf = round(prediction.get("confidence", 0) * 100)

    details = prediction.get("details", {}) or {}
    ti = details.get("technical_indicators", {}) or {}
    news_sentiment = details.get("news_sentiment", {}) or {}
    fin = details.get("financial_data", {}) or {}
    model_used = details.get("model_used", "rule_based")

    change_pct = (predicted - current) / current * 100 if current > 0 else 0

    rsi = ti.get("rsi", 50)
    macd_diff = ti.get("macd_diff", 0)
    bb_pos = ti.get("bb_position")
    volume_ratio = ti.get("volume_ratio", 1.0)
    ma5 = ti.get("ma5")
    ma20 = ti.get("ma20")

    model_desc = {
        "ensemble": "XGBoost+LSTM 앙상블",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
    }.get(model_used, "규칙 기반")

    # 종목 기본 정보
    stock_block = f"""## 종목 기본 정보
- 종목명: {stock_name}
- 현재가: {current:,}원
- AI 예측가 (1거래일): {round(predicted):,}원 ({change_pct:+.1f}%)
- 분석 모델: {model_desc} | 신뢰도: {conf}%
- 추세: {trend} | 투자의견: {rec}"""

    # 기술적 지표
    ti_lines = [
        f"- RSI: {rsi:.0f} ({'과매수' if rsi > 70 else '과매도' if rsi < 30 else '중립'})",
        f"- MACD: {'상승 모멘텀' if macd_diff > 0 else '하락 모멘텀'}",
    ]
    if bb_pos is not None:
        ti_lines.append(f"- 볼린저밴드 위치: {bb_pos * 100:.0f}% (0%=하단, 100%=상단)")
    if volume_ratio:
        ti_lines.append(f"- 거래량: 20일 평균 대비 {volume_ratio:.1f}배")
    if ma5 and ma20:
        ti_lines.append(f"- MA5: {ma5:,.0f}원 / MA20: {ma20:,.0f}원")
    ti_block = "## 기술적 지표\n" + "\n".join(ti_lines)

    # 재무 지표
    fin_lines = []
    if fin.get("per"):
        fin_lines.append(f"- PER: {fin['per']:.1f}")
    if fin.get("pbr"):
        fin_lines.append(f"- PBR: {fin['pbr']:.2f}")
    if fin.get("roe"):
        fin_lines.append(f"- ROE: {fin['roe']:.1f}%")
    if fin.get("div_yield"):
        fin_lines.append(f"- 배당수익률: {fin['div_yield']:.2f}%")
    fin_block = ("## 재무 지표\n" + "\n".join(fin_lines)) if fin_lines else ""

    # 뉴스 감성 요약
    news_count = news_sentiment.get("count", 0)
    pos_ratio = news_sentiment.get("positive_ratio", 0)
    neg_ratio = news_sentiment.get("negative_ratio", 0)
    sent_block = ""
    if news_count > 0:
        sent_block = f"## 뉴스 감성 통계\n- 분석 뉴스: {news_count}건 | 긍정 {pos_ratio*100:.0f}% / 부정 {neg_ratio*100:.0f}%"

    # 실제 뉴스 목록
    news_block = ""
    if news_list:
        news_items = []
        for i, n in enumerate(news_list[:10], 1):
            summary = n.get("summary") or n.get("title", "")
            sentiment = n.get("sentiment", "")
            sentiment_label = {"positive": "긍정", "negative": "부정", "neutral": "중립"}.get(sentiment, "")
            label = f" [{sentiment_label}]" if sentiment_label else ""
            news_items.append(f"{i}. {summary}{label}")
        news_block = "## 최근 뉴스\n" + "\n".join(news_items)

    sections = [stock_block, ti_block]
    if fin_block:
        sections.append(fin_block)
    if sent_block:
        sections.append(sent_block)
    if news_block:
        sections.append(news_block)

    data_block = "\n\n".join(sections)

    return f"""당신은 한국 주식 투자 전문 어시스턴트입니다.
아래 종목 데이터와 최근 뉴스를 종합 분석하여 투자 의견을 마크다운 형식으로 작성해주세요.

**출력 형식 (마크다운):**
### 📊 현황 요약
(현재가, 예측가, 추세를 2~3문장으로)

### 🔍 기술적 분석
(RSI, MACD, 볼린저밴드, 이동평균 등 주요 시그널 해석)

### 📰 뉴스 동향
(최근 뉴스 흐름과 시장 분위기 해석)

### 💡 종합 의견
(위 내용을 종합한 투자 관점 의견, 2~3문장)

> ⚠️ 이 분석은 참고용이며 투자 권유가 아닙니다.

**조건:**
- 각 섹션은 간결하게 3~5문장 이내
- 숫자는 자연스럽게 표현 (예: "RSI 72" → "RSI가 과매수 구간 진입")
- 뉴스 내용을 구체적으로 언급하여 현실감 있는 분석 제공
- 긍정/부정 어느 한쪽으로 단정짓지 말고 근거 있는 균형 잡힌 의견

---
{data_block}"""


async def generate_stock_opinion(
    stock_name: str,
    prediction: Dict,
    news_list: Optional[List[Dict]] = None
) -> Optional[str]:
    """Gemini REST API로 종목 투자의견 생성 (async)"""
    try:
        import asyncio
        import httpx
        from app.config import get_settings

        settings = get_settings()
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY 미설정 — AI 의견 생성 불가")
            return None

        prompt = _build_prompt(stock_name, prediction, news_list or [])
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
            # 429 Rate Limit 시 최대 2회 재시도 (3초 간격)
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
