"""Claude AI를 이용한 종목 투자의견 자연어 생성"""
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def _build_prompt(stock_name: str, prediction: Dict) -> str:
    current = prediction.get("current_price", 0)
    predicted = prediction.get("predicted_price", 0)
    trend = prediction.get("trend", "")
    rec = prediction.get("recommendation", "")
    conf = round(prediction.get("confidence", 0) * 100)

    details = prediction.get("details", {}) or {}
    ti = details.get("technical_indicators", {}) or {}
    news = details.get("news_sentiment", {}) or {}
    fin = details.get("financial_data", {}) or {}
    model_used = details.get("model_used", "rule_based")

    change_pct = (predicted - current) / current * 100 if current > 0 else 0

    rsi = ti.get("rsi", 50)
    macd_diff = ti.get("macd_diff", 0)
    bb_pos = ti.get("bb_position")
    volume_ratio = ti.get("volume_ratio", 1.0)

    news_count = news.get("count", 0)
    pos_ratio = news.get("positive_ratio", 0)
    neg_ratio = news.get("negative_ratio", 0)

    per = fin.get("per")
    pbr = fin.get("pbr")

    # 모델 설명
    model_desc = {
        "ensemble": "XGBoost+LSTM 앙상블",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
    }.get(model_used, "규칙 기반")

    lines = [
        f"종목명: {stock_name}",
        f"현재가: {current:,}원 / AI 예측가: {round(predicted):,}원 ({change_pct:+.1f}%)",
        f"분석 모델: {model_desc} | 신뢰도: {conf}%",
        f"추세: {trend} | 투자의견: {rec}",
        f"RSI: {rsi:.0f} | MACD 방향: {'상승' if macd_diff > 0 else '하락'}",
    ]
    if bb_pos is not None:
        lines.append(f"볼린저밴드 위치: {bb_pos * 100:.0f}% (0%=하단, 100%=상단)")
    if volume_ratio:
        lines.append(f"거래량: 20일 평균 대비 {volume_ratio:.1f}배")
    if news_count > 0:
        lines.append(f"최근 뉴스 {news_count}건: 긍정 {pos_ratio*100:.0f}% / 부정 {neg_ratio*100:.0f}%")
    if per:
        lines.append(f"PER: {per:.1f} | PBR: {pbr:.2f}" if pbr else f"PER: {per:.1f}")

    data_block = "\n".join(lines)

    return f"""당신은 친근하고 전문적인 한국 주식 투자 어시스턴트입니다.
아래 분석 데이터를 바탕으로 이 종목에 대한 투자 의견을 2~3문장으로 자연스럽게 말해주세요.

조건:
- 마치 전문가 친구가 편하게 조언해주는 말투로 작성
- 숫자는 자연스럽게 표현 (예: "RSI 72" → "RSI가 과매수 구간")
- 긍정/부정 어느 한쪽으로 단정짓지 말고 근거 있는 의견 제시
- 마지막에 한 줄 핵심 요약 포함
- 면책조항, 부연설명 없이 의견만 간결하게

분석 데이터:
{data_block}"""


async def generate_stock_opinion(stock_name: str, prediction: Dict) -> Optional[str]:
    """Gemini API로 종목 투자의견 생성 (async)"""
    try:
        from app.config import get_settings
        settings = get_settings()

        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY 미설정 — AI 의견 생성 불가")
            return None

        import asyncio
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = _build_prompt(stock_name, prediction)

        # google-generativeai는 sync라서 executor로 실행
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(max_output_tokens=300, temperature=0.7),
            )
        )

        opinion = response.text.strip()
        logger.info(f"AI 의견 생성 완료: {stock_name} ({len(opinion)}자)")
        return opinion

    except Exception as e:
        logger.error(f"AI 의견 생성 실패: {e}")
        return None
