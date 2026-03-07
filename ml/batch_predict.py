"""배치 예측 생성 스크립트

ML 파이프라인 학습 완료 후 모든 모니터링 종목에 대해
예측을 생성하여 predictions 테이블에 저장합니다.
매일 자동으로 예측 레코드를 쌓아 적중률 추적을 가능하게 합니다.
"""
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from ml.logger import get_logger
logger = get_logger("batch_predict")


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


def get_active_stocks(session) -> list:
    """활성 종목 조회 (priority 높은 순)"""
    rows = session.execute(
        text("""
            SELECT stock_code, stock_name
            FROM stocks
            WHERE is_active = true
            ORDER BY priority DESC, stock_code
        """)
    ).fetchall()
    return [{"stock_code": r.stock_code, "stock_name": r.stock_name} for r in rows]


def get_chart_data_from_csv(stock_code: str, count: int = 100) -> list:
    """CSV 파일에서 일봉 데이터를 KIS API 형식으로 반환 (최신순)"""
    import pandas as pd

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    csv_path = data_dir / "raw" / "historical" / f"{stock_code}_historical.csv"

    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df.sort_values("date", ascending=False).head(count)

        return [
            {
                "stck_bsop_date": str(row["date"]).replace("-", ""),
                "stck_oprc": str(int(row.get("open", 0))),
                "stck_hgpr": str(int(row.get("high", 0))),
                "stck_lwpr": str(int(row.get("low", 0))),
                "stck_clpr": str(int(row.get("close", 0))),
                "acml_vol": str(int(row.get("volume", 0))),
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        logger.warning(f"CSV 로드 실패 ({stock_code}): {e}")
        return []


def save_prediction(session, result: dict):
    """예측 결과를 DB에 저장 (동일 종목+날짜 있으면 UPDATE)"""
    existing = session.execute(
        text("""
            SELECT id FROM predictions
            WHERE stock_code = :code AND prediction_date = :date
        """),
        {"code": result["stock_code"], "date": result["prediction_date"]}
    ).fetchone()

    if existing:
        session.execute(
            text("""
                UPDATE predictions
                SET current_price = :current_price,
                    predicted_price = :predicted_price,
                    confidence = :confidence,
                    trend = :trend,
                    recommendation = :recommendation,
                    stock_name = :stock_name
                WHERE id = :id
            """),
            {
                "current_price": result["current_price"],
                "predicted_price": result["predicted_price"],
                "confidence": result["confidence"],
                "trend": result["trend"],
                "recommendation": result["recommendation"],
                "stock_name": result["stock_name"],
                "id": existing.id,
            }
        )
    else:
        session.execute(
            text("""
                INSERT INTO predictions
                    (stock_code, stock_name, current_price, predicted_price,
                     prediction_date, confidence, trend, recommendation)
                VALUES
                    (:stock_code, :stock_name, :current_price, :predicted_price,
                     :prediction_date, :confidence, :trend, :recommendation)
            """),
            {
                "stock_code": result["stock_code"],
                "stock_name": result["stock_name"],
                "current_price": result["current_price"],
                "predicted_price": result["predicted_price"],
                "prediction_date": result["prediction_date"],
                "confidence": result["confidence"],
                "trend": result["trend"],
                "recommendation": result["recommendation"],
            }
        )


def run():
    """모든 활성 종목 배치 예측 실행"""
    from app.services.prediction import PredictionService

    session = get_db_session()
    predictor = PredictionService(use_ml=True, use_db=True)

    try:
        stocks = get_active_stocks(session)
        logger.info(f"배치 예측 대상: {len(stocks)}종목")

        success, failed, skipped = 0, 0, 0

        for stock in stocks:
            code = stock["stock_code"]
            name = stock["stock_name"]

            try:
                chart_data = get_chart_data_from_csv(code)

                if not chart_data or len(chart_data) < 5:
                    logger.warning(f"  [{code}] 차트 데이터 부족 — 스킵")
                    skipped += 1
                    continue

                result = predictor.predict_price(
                    stock_code=code,
                    stock_name=name,
                    chart_data=chart_data,
                )

                save_prediction(session, result)
                session.commit()

                change_pct = (result["predicted_price"] - result["current_price"]) / result["current_price"] * 100
                logger.info(
                    f"  [{code}] {name}: {result['current_price']:,}원 → "
                    f"{round(result['predicted_price']):,}원 ({change_pct:+.1f}%) "
                    f"| {result['prediction_date']} | {result['recommendation']}"
                )
                success += 1

            except Exception as e:
                session.rollback()
                logger.error(f"  [{code}] 예측 실패: {e}")
                failed += 1

        logger.info(f"배치 예측 완료: 성공 {success}건, 실패 {failed}건, 스킵 {skipped}건")

    except Exception as e:
        session.rollback()
        logger.error(f"배치 예측 중 치명적 오류: {e}", exc_info=True)
        sys.exit(1)
    finally:
        session.close()


if __name__ == "__main__":
    run()
