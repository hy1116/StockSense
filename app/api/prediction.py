"""주가 예측 API"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from app.services.kis_api import get_kis_client, KISAPIClient
from app.services.prediction import get_prediction_service, PredictionService

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

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"예측 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"예측 중 오류가 발생했습니다: {str(e)}")


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
