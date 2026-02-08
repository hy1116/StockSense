"""주가 예측 API"""
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
import logging
import sqlalchemy as sa

from sqlalchemy import select, func as sa_func, text

from app.services.kis_api import get_kis_client, KISAPIClient
from app.services.prediction import get_prediction_service, PredictionService, _get_sync_db
from app.models.prediction import Prediction
from app.schemas.portfolio import PredictionAccuracyResponse, PredictionHistoryItem

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/prediction", tags=["prediction"])


class PredictionRequest(BaseModel):
    """예측 요청"""
    stock_code: str


class PredictionResponse(BaseModel):
    """예측 응답"""
    stock_code: str
    stock_name: str
    current_price: int
    predicted_price: float
    prediction_date: str
    confidence: float
    trend: str
    recommendation: str


def _save_prediction_to_db(result: dict):
    """예측 결과를 DB에 저장 (동일 종목+날짜 있으면 UPDATE)"""
    try:
        engine, Session = _get_sync_db()

        with Session() as session:
            existing = session.execute(
                select(Prediction).where(
                    Prediction.stock_code == result["stock_code"],
                    Prediction.prediction_date == result["prediction_date"],
                )
            ).scalar_one_or_none()

            if existing:
                existing.current_price = result["current_price"]
                existing.predicted_price = result["predicted_price"]
                existing.confidence = result["confidence"]
                existing.trend = result["trend"]
                existing.recommendation = result["recommendation"]
                existing.stock_name = result["stock_name"]
            else:
                pred = Prediction(
                    stock_code=result["stock_code"],
                    stock_name=result["stock_name"],
                    current_price=result["current_price"],
                    predicted_price=result["predicted_price"],
                    prediction_date=result["prediction_date"],
                    confidence=result["confidence"],
                    trend=result["trend"],
                    recommendation=result["recommendation"],
                )
                session.add(pred)

            session.commit()
            logger.info(f"예측 저장 완료: {result['stock_code']} / {result['prediction_date']}")
    except Exception as e:
        logger.warning(f"예측 저장 실패 (무시): {e}")


@router.post("/", response_model=PredictionResponse)
async def predict_stock(
    request: PredictionRequest,
    kis_client: KISAPIClient = Depends(get_kis_client),
    predictor: PredictionService = Depends(get_prediction_service)
):
    """주가 예측

    Args:
        request: 예측 요청 (종목 코드)

    Returns:
        예측 결과
    """
    try:
        stock_code = request.stock_code.upper()

        # KIS API로 주식 정보 조회
        stock_data = kis_client.get_stock_price(stock_code)

        if not stock_data or "output" not in stock_data:
            raise HTTPException(status_code=404, detail=f"종목 {stock_code}를 찾을 수 없습니다")

        # 종목명
        stock_name = stock_data["output"].get("hts_kor_isnm", stock_code)

        # 차트 데이터 조회 (최근 100일)
        chart_data = kis_client.get_daily_chart(stock_code, period="D", count=100)

        if not chart_data or "output2" not in chart_data:
            raise HTTPException(status_code=500, detail="차트 데이터를 가져올 수 없습니다")

        # 예측 서비스
        result = predictor.predict_price(
            stock_code=stock_code,
            stock_name=stock_name,
            chart_data=chart_data["output2"]
        )

        logger.info(f"예측 완료: {stock_code} - {result}")

        # 예측 결과 DB 저장
        _save_prediction_to_db(result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {str(e)}")


@router.get("/{stock_code}/accuracy", response_model=PredictionAccuracyResponse)
async def get_prediction_accuracy(
    stock_code: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """예측 적중률 조회

    Args:
        stock_code: 종목 코드
        days: 조회 기간 (일)

    Returns:
        적중률 통계 + 최근 예측 기록
    """
    try:
        stock_code = stock_code.upper()
        engine, Session = _get_sync_db()

        with Session() as session:
            # 전체 예측 수 (최근 N일)
            base_filter = [
                Prediction.stock_code == stock_code,
                Prediction.created_at >= sa_func.now() - text(f"interval '{days} days'"),
            ]

            total_q = session.execute(
                select(sa_func.count(Prediction.id)).where(*base_filter)
            )
            total_predictions = total_q.scalar() or 0

            # 평가 완료 수
            eval_q = session.execute(
                select(sa_func.count(Prediction.id)).where(
                    *base_filter,
                    Prediction.is_evaluated == True,
                )
            )
            evaluated_count = eval_q.scalar() or 0

            # 방향 적중률 & 평균 오차율
            direction_accuracy = None
            avg_error_rate = None

            if evaluated_count > 0:
                stats_q = session.execute(
                    select(
                        sa_func.avg(
                            sa.cast(Prediction.direction_correct, sa.Integer)
                        ) * 100,
                        sa_func.avg(sa_func.abs(Prediction.error_rate)),
                    ).where(
                        *base_filter,
                        Prediction.is_evaluated == True,
                    )
                )
                row = stats_q.one()
                if row[0] is not None:
                    direction_accuracy = round(float(row[0]), 1)
                if row[1] is not None:
                    avg_error_rate = round(float(row[1]), 2)

            # 최근 5건
            recent_q = session.execute(
                select(Prediction)
                .where(Prediction.stock_code == stock_code)
                .order_by(Prediction.prediction_date.desc())
                .limit(5)
            )
            recent_rows = recent_q.scalars().all()

            recent_predictions = [
                PredictionHistoryItem(
                    prediction_date=r.prediction_date or "",
                    predicted_price=r.predicted_price,
                    actual_price=r.actual_price,
                    error_rate=round(r.error_rate, 2) if r.error_rate is not None else None,
                    direction_correct=r.direction_correct,
                )
                for r in recent_rows
            ]

        return PredictionAccuracyResponse(
            stock_code=stock_code,
            total_predictions=total_predictions,
            evaluated_count=evaluated_count,
            direction_accuracy=direction_accuracy,
            avg_error_rate=avg_error_rate,
            recent_predictions=recent_predictions,
        )

    except Exception as e:
        logger.error(f"적중률 조회 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"적중률 조회 중 오류: {str(e)}")


@router.get("/{stock_code}", response_model=PredictionResponse)
async def get_prediction(
    stock_code: str,
    kis_client: KISAPIClient = Depends(get_kis_client),
    predictor: PredictionService = Depends(get_prediction_service)
):
    """주가 예측 조회 (GET)

    Args:
        stock_code: 종목 코드

    Returns:
        예측 결과
    """
    return await predict_stock(
        PredictionRequest(stock_code=stock_code),
        kis_client=kis_client,
        predictor=predictor
    )
