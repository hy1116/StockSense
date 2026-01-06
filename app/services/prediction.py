"""주가 예측 서비스"""
import logging
from typing import Dict, List
from datetime import datetime, timedelta
import statistics

logger = logging.getLogger(__name__)


class PredictionService:
    """주가 예측 서비스 (간단한 이동평균 기반)"""

    def predict_price(self, stock_code: str, stock_name: str, chart_data: List[Dict]) -> Dict:
        """주가 예측

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            chart_data: 차트 데이터 (KIS API에서 받은 output2)

        Returns:
            예측 결과
        """
        try:
            if not chart_data or len(chart_data) < 5:
                return self._get_default_prediction(stock_code, stock_name)

            # 최근 데이터 (KIS API는 최신 데이터가 앞에 있음)
            recent_data = chart_data[:30]  # 최근 30일

            # 종가 데이터 추출
            close_prices = [int(item.get("stck_clpr", 0)) for item in recent_data]
            volumes = [int(item.get("acml_vol", 0)) for item in recent_data]

            if not close_prices or close_prices[0] == 0:
                return self._get_default_prediction(stock_code, stock_name)

            current_price = close_prices[0]

            # 단순 이동평균 계산
            ma5 = statistics.mean(close_prices[:5]) if len(close_prices) >= 5 else current_price
            ma10 = statistics.mean(close_prices[:10]) if len(close_prices) >= 10 else current_price
            ma20 = statistics.mean(close_prices[:20]) if len(close_prices) >= 20 else current_price

            # 추세 분석
            trend = self._analyze_trend(current_price, ma5, ma10, ma20)

            # 예측가 계산 (단순 가중 이동평균)
            predicted_price = (ma5 * 0.5 + ma10 * 0.3 + ma20 * 0.2)

            # 신뢰도 계산 (거래량 기반)
            confidence = self._calculate_confidence(volumes)

            # 투자의견 생성
            recommendation = self._get_recommendation(current_price, predicted_price, trend)

            # 예측 날짜 (다음 거래일)
            prediction_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

            return {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "current_price": current_price,
                "predicted_price": round(predicted_price, 2),
                "prediction_date": prediction_date,
                "confidence": round(confidence, 2),
                "trend": trend,
                "recommendation": recommendation
            }

        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            return self._get_default_prediction(stock_code, stock_name)

    def _analyze_trend(self, current: float, ma5: float, ma10: float, ma20: float) -> str:
        """추세 분석"""
        if current > ma5 > ma10 > ma20:
            return "강한 상승"
        elif current > ma5 > ma10:
            return "상승"
        elif current < ma5 < ma10 < ma20:
            return "강한 하락"
        elif current < ma5 < ma10:
            return "하락"
        else:
            return "보합"

    def _calculate_confidence(self, volumes: List[int]) -> float:
        """신뢰도 계산 (거래량 기반)"""
        if not volumes or len(volumes) < 5:
            return 0.5

        recent_volume = statistics.mean(volumes[:5])
        avg_volume = statistics.mean(volumes)

        if avg_volume == 0:
            return 0.5

        # 거래량 비율 기반 신뢰도
        volume_ratio = recent_volume / avg_volume

        if volume_ratio > 1.5:
            return min(0.9, 0.6 + (volume_ratio - 1.5) * 0.2)
        elif volume_ratio < 0.5:
            return max(0.3, 0.6 - (0.5 - volume_ratio) * 0.2)
        else:
            return 0.6

    def _get_recommendation(self, current: float, predicted: float, trend: str) -> str:
        """투자의견 생성"""
        change_rate = ((predicted - current) / current) * 100

        if trend in ["강한 상승", "상승"] and change_rate > 2:
            return "매수"
        elif trend in ["강한 하락", "하락"] and change_rate < -2:
            return "매도"
        else:
            return "보유"

    def _get_default_prediction(self, stock_code: str, stock_name: str) -> Dict:
        """기본 예측 결과 (데이터 부족 시)"""
        return {
            "stock_code": stock_code,
            "stock_name": stock_name,
            "current_price": 0,
            "predicted_price": 0.0,
            "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "confidence": 0.3,
            "trend": "보합",
            "recommendation": "보유"
        }


def get_prediction_service() -> PredictionService:
    """예측 서비스 인스턴스 반환"""
    return PredictionService()
