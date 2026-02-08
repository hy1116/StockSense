"""전일 예측 평가 배치 스크립트

미평가 예측 레코드를 찾아 실제 종가와 비교하여 적중률을 계산합니다.
run_pipeline.py 에서 데이터 수집 직후에 실행됩니다.
"""
import sys
import os
import logging
from datetime import datetime, date

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluate-predictions")


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


def get_kis_client():
    """KIS API 클라이언트 생성"""
    from app.services.kis_api import KISAPIClient

    app_key = os.getenv("KIS_APP_KEY")
    app_secret = os.getenv("KIS_APP_SECRET")
    account_number = os.getenv("KIS_ACCOUNT_NUMBER")
    account_product_code = os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "01")
    base_url = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
    use_mock = os.getenv("KIS_USE_MOCK", "True").lower() == "true"
    cust_type = os.getenv("KIS_CUST_TYPE", "P")

    if not app_key or not app_secret or not account_number:
        raise ValueError("KIS API credentials not found in environment variables.")

    return KISAPIClient(
        app_key=app_key,
        app_secret=app_secret,
        account_number=account_number,
        account_product_code=account_product_code,
        base_url=base_url,
        use_mock=use_mock,
        cust_type=cust_type,
    )


def get_actual_price(kis_client, stock_code: str, target_date: str) -> float | None:
    """KIS API에서 특정 날짜의 실제 종가를 조회

    Args:
        kis_client: KIS API 클라이언트
        stock_code: 종목 코드
        target_date: 조회 날짜 (YYYY-MM-DD)

    Returns:
        실제 종가 (없으면 None)
    """
    try:
        chart_data = kis_client.get_daily_chart(stock_code, period="D", count=10)
        if not chart_data or "output2" not in chart_data:
            return None

        # target_date를 YYYYMMDD 형식으로 변환
        target_yyyymmdd = target_date.replace("-", "")

        for item in chart_data["output2"]:
            item_date = item.get("stck_bsop_date", "")
            if item_date == target_yyyymmdd:
                close_price = int(item.get("stck_clpr", 0))
                if close_price > 0:
                    return float(close_price)

        return None
    except Exception as e:
        logger.warning(f"종가 조회 실패 ({stock_code} / {target_date}): {e}")
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
                  AND prediction_date < :today
                ORDER BY prediction_date
            """),
            {"today": today_str}
        ).fetchall()

        if not rows:
            logger.info("평가할 예측 레코드가 없습니다.")
            return

        logger.info(f"평가 대상: {len(rows)}건")

        kis_client = get_kis_client()
        evaluated = 0
        failed = 0

        # 종목별로 그룹핑하여 API 호출 최소화
        stock_prices = {}

        for row in rows:
            pred_id = row.id
            stock_code = row.stock_code
            prediction_date = row.prediction_date
            predicted_price = row.predicted_price
            current_price = row.current_price  # 예측 시점의 현재가

            cache_key = f"{stock_code}_{prediction_date}"
            if cache_key not in stock_prices:
                actual = get_actual_price(kis_client, stock_code, prediction_date)
                stock_prices[cache_key] = actual

            actual_price = stock_prices[cache_key]

            if actual_price is None:
                logger.warning(f"실제 종가를 찾을 수 없음: {stock_code} / {prediction_date}")
                failed += 1
                continue

            # 오차율: (예측가 - 실제가) / 실제가 * 100
            error_rate = (predicted_price - actual_price) / actual_price * 100

            # 방향 적중: (예측가 > 현재가) == (실제가 > 현재가)
            if current_price and current_price > 0:
                predicted_up = predicted_price > current_price
                actual_up = actual_price > current_price
                direction_correct = predicted_up == actual_up
            else:
                direction_correct = None

            session.execute(
                text("""
                    UPDATE predictions
                    SET actual_price = :actual_price,
                        error_rate = :error_rate,
                        direction_correct = :direction_correct,
                        is_evaluated = true
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
