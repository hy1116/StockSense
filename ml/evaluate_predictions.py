"""전일 예측 평가 배치 스크립트

미평가 예측 레코드를 찾아 실제 종가와 비교하여 적중률을 계산합니다.
run_pipeline.py 에서 데이터 수집 직후에 실행됩니다.

실제 종가 조회: Naver fchart API (인증 불필요, KIS API 의존성 제거)
"""
import sys
import os
import ast
import logging
import requests
from datetime import datetime, date, timedelta

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ml.logger import get_logger
logger = get_logger("evaluate_predictions")

_NAVER_FCHART_URL = "https://fchart.stock.naver.com/siseJson.nhn"
_NAVER_HEADERS = {"Referer": "https://finance.naver.com/"}


def get_db_session():
    """동기 DB 세션 생성"""
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/stocksense"
    )
    if "+asyncpg" in database_url:
        database_url = database_url.replace("+asyncpg", "")

    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()


def get_actual_price(stock_code: str, target_date: str) -> float | None:
    """Naver fchart API로 특정 날짜의 실제 종가 조회 (인증 불필요)

    Args:
        stock_code: 종목 코드
        target_date: 조회 날짜 (YYYY-MM-DD)

    Returns:
        실제 종가 (없으면 None — 휴장일 포함)
    """
    try:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        target_yyyymmdd = target_dt.strftime("%Y%m%d")
        # 넉넉하게 30일 전부터 조회 (휴장일 연속 구간 대비)
        start_yyyymmdd = (target_dt - timedelta(days=30)).strftime("%Y%m%d")

        params = {
            "symbol": stock_code,
            "requestType": "1",
            "startTime": start_yyyymmdd,
            "endTime": target_yyyymmdd,
            "timeframe": "day",
        }
        resp = requests.get(
            _NAVER_FCHART_URL, params=params,
            headers=_NAVER_HEADERS, timeout=10
        )
        resp.raise_for_status()

        # 응답이 Python 리터럴(single-quote 혼재) → ast로 파싱
        raw = resp.text.replace("null", "None")
        rows = ast.literal_eval(raw.strip())

        for row in rows[1:]:  # 첫 행은 한글 헤더
            if not row or row[0] is None:
                continue
            if str(row[0]) == target_yyyymmdd:
                close = row[4]  # [날짜, 시가, 고가, 저가, 종가, 거래량]
                if close is not None and close > 0:
                    return float(close)
                return None  # 해당 날짜 데이터 있으나 종가 0

        return None  # 해당 날짜 없음 (휴장일)

    except Exception as e:
        logger.warning(f"Naver 종가 조회 실패 ({stock_code} / {target_date}): {e}")
        return None


def evaluate():
    """미평가 예측을 찾아 실제 종가와 비교"""
    session = get_db_session()
    today_str = date.today().strftime("%Y-%m-%d")

    try:
        # 미평가이고 prediction_date가 오늘 이전인 레코드 조회
        rows = session.execute(
            text("""
                SELECT id, stock_code, prediction_date, predicted_price, current_price
                FROM predictions
                WHERE is_evaluated = false
                  AND prediction_date <= :today
                ORDER BY prediction_date
            """),
            {"today": today_str}
        ).fetchall()

        if not rows:
            logger.info("평가할 예측 레코드가 없습니다.")
            return

        logger.info(f"평가 대상: {len(rows)}건")

        evaluated = 0
        failed = 0

        # 종목+날짜 조합별로 캐싱 (동일 조합 중복 API 호출 방지)
        price_cache: dict[str, float | None] = {}

        for row in rows:
            pred_id = row.id
            stock_code = row.stock_code
            prediction_date = row.prediction_date  # YYYY-MM-DD
            predicted_price = row.predicted_price
            current_price = row.current_price

            cache_key = f"{stock_code}_{prediction_date}"
            if cache_key not in price_cache:
                price_cache[cache_key] = get_actual_price(stock_code, prediction_date)

            actual_price = price_cache[cache_key]

            if actual_price is None:
                logger.warning(f"실제 종가를 찾을 수 없음 (휴장일?): {stock_code} / {prediction_date}")
                failed += 1
                continue

            # 오차율: (예측가 - 실제가) / 실제가 * 100
            error_rate = (predicted_price - actual_price) / actual_price * 100

            # 방향 적중: 예측 방향(상승/하락)이 실제와 일치하는지
            if current_price and current_price > 0:
                direction_correct = (predicted_price > current_price) == (actual_price > current_price)
            else:
                direction_correct = None

            session.execute(
                text("""
                    UPDATE predictions
                    SET actual_price      = :actual_price,
                        error_rate        = :error_rate,
                        direction_correct = :direction_correct,
                        is_evaluated      = true
                    WHERE id = :id
                """),
                {
                    "actual_price": actual_price,
                    "error_rate": round(error_rate, 4),
                    "direction_correct": direction_correct,
                    "id": pred_id,
                }
            )
            evaluated += 1
            logger.info(
                f"  [{stock_code}] {prediction_date}: "
                f"예측={predicted_price:.0f}, 실제={actual_price:.0f}, "
                f"오차={error_rate:.2f}%, 방향={'O' if direction_correct else 'X'}"
            )

        session.commit()
        logger.info(f"평가 완료: {evaluated}건 성공, {failed}건 실패")

    except Exception as e:
        session.rollback()
        logger.error(f"평가 중 오류: {e}", exc_info=True)
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    evaluate()
