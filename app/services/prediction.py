"""주가 예측 서비스"""
import logging
import pickle
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import statistics
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class PredictionService:
    """주가 예측 서비스 (ML 모델 + 기술적 지표 기반)"""

    def __init__(self, use_ml: bool = True, model_dir: str = "./models", use_db: bool = True):
        """
        Args:
            use_ml: ML 모델 사용 여부 (True: ML 모델, False: 룰 기반)
            model_dir: 모델 파일 디렉토리 (fallback용)
            use_db: DB에서 모델 로드 여부 (True: DB 우선, False: 파일만)
        """
        self.use_ml = use_ml
        self.use_db = use_db
        self.model_dir = Path(model_dir)
        self.ml_model = None
        self.ml_scaler = None
        self.feature_columns = []
        self.model_version = None
        self.model_info = {}

        # ML 모델 로드 시도
        if self.use_ml:
            try:
                # 1. DB에서 활성 모델 로드 시도
                if self.use_db:
                    self._load_ml_model_from_db()

                # 2. DB 로드 실패 시 파일에서 로드
                if self.ml_model is None:
                    self._load_ml_model_from_file()

                if self.ml_model is not None:
                    logger.info(f"✅ ML model loaded successfully (version: {self.model_version})")
                else:
                    raise Exception("No model available")

            except Exception as e:
                logger.warning(f"⚠️ Failed to load ML model: {e}. Falling back to rule-based prediction.")
                self.use_ml = False

        # 룰 기반 예측용 스케일러
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _load_ml_model_from_db(self):
        """DB에서 활성화된 ML 모델 로드"""
        try:
            from app.config import get_settings
            from app.models.ml_model import ModelTrainingHistory

            settings = get_settings()
            # async URL을 sync URL로 변환
            db_url = settings.database_url.replace("+asyncpg", "")
            engine = create_engine(db_url)
            Session = sessionmaker(bind=engine)

            with Session() as session:
                # 활성 모델 조회
                result = session.execute(
                    select(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .order_by(ModelTrainingHistory.trained_at.desc())
                    .limit(1)
                )
                active_model = result.scalar_one_or_none()

                if active_model is None:
                    logger.info("📭 No active model found in DB")
                    return

                # 모델 바이너리 역직렬화
                if active_model.model_binary is None or active_model.scaler_binary is None:
                    logger.warning("⚠️ Active model has no binary data")
                    return

                self.ml_model = pickle.loads(active_model.model_binary)
                self.ml_scaler = pickle.loads(active_model.scaler_binary)

                # 피처 컬럼 로드
                if active_model.feature_columns:
                    self.feature_columns = json.loads(active_model.feature_columns)
                else:
                    self.feature_columns = self._get_default_feature_columns()

                # 모델 정보 저장
                self.model_version = active_model.model_version
                self.model_info = {
                    'id': active_model.id,
                    'model_name': active_model.model_name,
                    'model_type': active_model.model_type,
                    'version': active_model.model_version,
                    'trained_at': active_model.trained_at.isoformat() if active_model.trained_at else None,
                    'test_score': active_model.test_score,
                    'mae': active_model.mae,
                    'rmse': active_model.rmse,
                    'train_samples': active_model.train_samples
                }

                logger.info(f"📦 Model loaded from DB: {active_model.model_name} "
                          f"(version={active_model.model_version}, test_score={active_model.test_score:.4f})")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load model from DB: {e}")
            self.ml_model = None

    def _get_default_feature_columns(self) -> list:
        """기본 피처 컬럼 반환"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_diff',
            'price_change_1d', 'volume_change'
        ]

    def _load_ml_model_from_file(self):
        """파일에서 ML 모델 로드 (DB 실패 시 fallback)"""
        model_path = self.model_dir / "stock_prediction_v1.pkl"
        scaler_path = self.model_dir / "scaler.pkl"
        metadata_path = self.model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        # 모델 로드
        self.ml_model = joblib.load(model_path)
        self.ml_scaler = joblib.load(scaler_path)

        # 메타데이터에서 피처 컬럼 정보 로드
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.feature_columns = metadata.get('feature_columns', [])
                self.model_version = metadata.get('version', 'file')
                self.model_info = {
                    'model_name': metadata.get('model_name'),
                    'model_type': metadata.get('model_type'),
                    'version': metadata.get('version'),
                    'trained_at': metadata.get('trained_at'),
                    'test_score': metadata.get('test_score'),
                    'n_samples': metadata.get('n_samples')
                }
                logger.info(f"📁 Model loaded from file: {model_path.name} "
                          f"(version={self.model_version}, test_score={metadata.get('test_score', 'N/A')})")
        else:
            # 기본 피처 컬럼
            self.feature_columns = self._get_default_feature_columns()
            self.model_version = 'file'
            self.model_info = {'source': 'file'}
            logger.info(f"📁 Model loaded from file (no metadata): {model_path.name}")

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
            logger.info(f"🔮 Prediction start for {stock_code}: chart_data length={len(chart_data) if chart_data else 0}, use_ml={self.use_ml}")

            if not chart_data or len(chart_data) < 5:
                logger.warning(f"Insufficient chart data for {stock_code}: len={len(chart_data) if chart_data else 0}")
                return self._get_default_prediction(stock_code, stock_name)

            # 최근 데이터 (KIS API는 최신 데이터가 앞에 있음)
            recent_data = chart_data[:30]  # 최근 30일

            logger.info(f"First chart item type: {type(recent_data[0])}, keys: {list(recent_data[0].keys()) if isinstance(recent_data[0], dict) else 'not a dict'}")

            # OHLCV 데이터 추출
            close_prices = np.array([int(item.get("stck_clpr", 0)) for item in recent_data])
            open_prices = np.array([int(item.get("stck_oprc", 0)) for item in recent_data])
            high_prices = np.array([int(item.get("stck_hgpr", 0)) for item in recent_data])
            low_prices = np.array([int(item.get("stck_lwpr", 0)) for item in recent_data])
            volumes = np.array([int(item.get("acml_vol", 0)) for item in recent_data])

            logger.info(f"Parsed prices for {stock_code}: close_prices[0]={close_prices[0] if len(close_prices) > 0 else 'empty'}, len={len(close_prices)}")

            if len(close_prices) == 0 or close_prices[0] == 0:
                logger.warning(f"Invalid close prices for {stock_code}: len={len(close_prices)}, first={close_prices[0] if len(close_prices) > 0 else 'N/A'}")
                return self._get_default_prediction(stock_code, stock_name)

            current_price = int(close_prices[0])

            # 기술적 지표 계산
            ma5 = np.mean(close_prices[:5]) if len(close_prices) >= 5 else current_price
            ma10 = np.mean(close_prices[:10]) if len(close_prices) >= 10 else current_price
            ma20 = np.mean(close_prices[:20]) if len(close_prices) >= 20 else current_price

            # RSI 계산
            rsi = self._calculate_rsi(close_prices)

            # 볼린저 밴드
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)

            # MACD
            macd, signal = self._calculate_macd(close_prices)

            # 추세 분석
            trend = self._analyze_trend(current_price, ma5, ma10, ma20, rsi)

            # 예측 가격 계산 (ML 또는 룰 기반)
            if self.use_ml and self.ml_model is not None:
                predicted_price = self._predict_with_ml(
                    current_price, open_prices[0], high_prices[0], low_prices[0], volumes[0],
                    ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                    close_prices
                )
                logger.info(f"🤖 ML prediction for {stock_code}: {predicted_price:.2f}")
            else:
                # 룰 기반 예측 (기존 로직)
                predicted_price = self._predict_advanced(
                    current_price, ma5, ma10, ma20, rsi,
                    bb_upper, bb_middle, bb_lower, macd, signal
                )
                logger.info(f"📊 Rule-based prediction for {stock_code}: {predicted_price:.2f}")

            # 신뢰도 계산 (거래량 + 변동성 기반)
            confidence = self._calculate_confidence_advanced(
                volumes, close_prices, current_price, bb_upper, bb_lower
            )

            # ML 모델 사용 시 신뢰도 상향 조정
            if self.use_ml and self.ml_model is not None:
                confidence = min(0.95, confidence + 0.1)

            # 투자의견 생성
            recommendation = self._get_recommendation(
                current_price, predicted_price, trend, rsi, macd, signal
            )

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

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI(Relative Strength Index) 계산"""
        if len(prices) < period + 1:
            return 50.0

        # 가격 변화량 계산 (최신 데이터가 앞에 있으므로 역순으로)
        deltas = np.diff(prices[::-1])

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period]) if len(gains) >= period else 0
        avg_loss = np.mean(losses[:period]) if len(losses) >= period else 0

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2):
        """볼린저 밴드 계산"""
        if len(prices) < period:
            current = float(prices[0]) if len(prices) > 0 else 0
            return current, current, current

        # 최근 period 일 데이터
        recent_prices = prices[:period]

        middle = np.mean(recent_prices)
        std = np.std(recent_prices)

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        return float(upper), float(middle), float(lower)

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9):
        """MACD 계산"""
        if len(prices) < slow:
            return 0.0, 0.0

        # 지수이동평균 계산
        prices_reversed = prices[::-1]  # 시계열 순서로 변환

        # EMA 계산
        ema_fast = self._calculate_ema(prices_reversed, fast)
        ema_slow = self._calculate_ema(prices_reversed, slow)

        macd_line = ema_fast - ema_slow

        # Signal line (MACD의 EMA)
        signal_line = macd_line  # 단순화

        return float(macd_line), float(signal_line)

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """지수이동평균(EMA) 계산"""
        if len(prices) < period:
            return float(np.mean(prices))

        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return float(ema)

    def _predict_advanced(self, current: float, ma5: float, ma10: float, ma20: float,
                         rsi: float, bb_upper: float, bb_middle: float, bb_lower: float,
                         macd: float, signal: float) -> float:
        """고급 예측 (여러 지표 결합)"""
        predicted = current

        # 1. 이동평균 기반 예측 (가중치 적용)
        ma_prediction = (ma5 * 0.5 + ma10 * 0.3 + ma20 * 0.2)

        # 2. RSI 기반 조정
        if rsi > 70:  # 과매수
            rsi_factor = -0.02
        elif rsi < 30:  # 과매도
            rsi_factor = 0.02
        else:
            rsi_factor = (50 - rsi) / 5000  # 중립 영역

        # 3. 볼린저 밴드 기반 조정
        bb_position = (current - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        if bb_position > 0.8:  # 상단 근처
            bb_factor = -0.01
        elif bb_position < 0.2:  # 하단 근처
            bb_factor = 0.01
        else:
            bb_factor = 0

        # 4. MACD 기반 조정
        if macd > signal:
            macd_factor = 0.01
        else:
            macd_factor = -0.01

        # 종합 예측
        total_factor = 1 + rsi_factor + bb_factor + macd_factor
        predicted = ma_prediction * 0.3 + current * 0.7 * total_factor

        return float(predicted)

    def _calculate_confidence_advanced(self, volumes: np.ndarray, prices: np.ndarray,
                                      current: float, bb_upper: float, bb_lower: float) -> float:
        """고급 신뢰도 계산 (거래량 + 변동성)"""
        confidence = 0.5

        # 1. 거래량 기반 신뢰도
        if len(volumes) >= 5:
            recent_volume = np.mean(volumes[:5])
            avg_volume = np.mean(volumes)

            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                if volume_ratio > 1.5:
                    confidence += 0.2
                elif volume_ratio > 1.0:
                    confidence += 0.1

        # 2. 변동성 기반 신뢰도
        if len(prices) >= 5:
            volatility = np.std(prices[:5]) / np.mean(prices[:5]) if np.mean(prices[:5]) > 0 else 0
            if volatility < 0.02:  # 낮은 변동성
                confidence += 0.2
            elif volatility < 0.05:
                confidence += 0.1
            else:
                confidence -= 0.1

        # 3. 볼린저 밴드 기반 신뢰도
        if bb_upper != bb_lower:
            bb_width = (bb_upper - bb_lower) / current
            if bb_width < 0.1:  # 좁은 밴드
                confidence += 0.1

        return min(0.95, max(0.3, confidence))

    def _analyze_trend(self, current: float, ma5: float, ma10: float, ma20: float, rsi: float) -> str:
        """추세 분석"""
        if current > ma5 > ma10 > ma20 and rsi > 50:
            return "강한 상승"
        elif current > ma5 > ma10:
            return "상승"
        elif current < ma5 < ma10 < ma20 and rsi < 50:
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

    def _get_recommendation(self, current: float, predicted: float, trend: str,
                           rsi: float, macd: float, signal: float) -> str:
        """투자의견 생성 (고급)"""
        change_rate = ((predicted - current) / current) * 100

        # 기본 점수
        score = 0

        # 1. 가격 변화율 기반
        if change_rate > 3:
            score += 2
        elif change_rate > 1:
            score += 1
        elif change_rate < -3:
            score -= 2
        elif change_rate < -1:
            score -= 1

        # 2. 추세 기반
        if trend == "강한 상승":
            score += 2
        elif trend == "상승":
            score += 1
        elif trend == "강한 하락":
            score -= 2
        elif trend == "하락":
            score -= 1

        # 3. RSI 기반
        if rsi < 30:  # 과매도
            score += 1
        elif rsi > 70:  # 과매수
            score -= 1

        # 4. MACD 기반
        if macd > signal:
            score += 1
        else:
            score -= 1

        # 최종 의견
        if score >= 3:
            return "적극 매수"
        elif score >= 1:
            return "매수"
        elif score <= -3:
            return "적극 매도"
        elif score <= -1:
            return "매도"
        else:
            return "보유"

    def _predict_with_ml(self, current_price: float, open_price: float, high_price: float,
                        low_price: float, volume: float, ma5: float, ma10: float, ma20: float,
                        rsi: float, bb_upper: float, bb_middle: float, bb_lower: float,
                        macd: float, signal: float, close_prices: np.ndarray) -> float:
        """ML 모델을 사용한 예측

        Args:
            current_price: 현재가
            open_price: 시가
            high_price: 고가
            low_price: 저가
            volume: 거래량
            ma5, ma10, ma20: 이동평균
            rsi: RSI
            bb_upper, bb_middle, bb_lower: 볼린저 밴드
            macd, signal: MACD
            close_prices: 종가 배열 (변화율 계산용)

        Returns:
            예측 가격
        """
        try:
            # 거래량 변화율 계산 (간단히 0으로 설정)
            volume_change = 0.0
            price_change_1d = 0.0

            # 종가 배열이 있으면 변화율 계산
            if len(close_prices) >= 2:
                price_change_1d = (close_prices[0] - close_prices[1]) / close_prices[1]

            # 피처 준비 (모델 학습 시 사용한 순서와 동일)
            features_dict = {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'volume': volume,
                'ma5': ma5,
                'ma10': ma10,
                'ma20': ma20,
                'rsi': rsi,
                'bb_upper': bb_upper,
                'bb_middle': bb_middle,
                'bb_lower': bb_lower,
                'macd': macd,
                'macd_signal': signal,
                'macd_diff': macd - signal,
                'price_change_1d': price_change_1d,
                'volume_change': volume_change
            }

            # 모델이 요구하는 피처만 선택
            features = [features_dict.get(col, 0) for col in self.feature_columns]

            # DataFrame으로 변환 (sklearn은 2D 배열을 요구)
            X = pd.DataFrame([features], columns=self.feature_columns)

            # 스케일링
            X_scaled = self.ml_scaler.transform(X)

            # 예측
            predicted = self.ml_model.predict(X_scaled)[0]

            # 예측값이 비정상적이면 현재가 기준으로 조정
            if predicted <= 0 or predicted > current_price * 2:
                logger.warning(f"⚠️ Abnormal ML prediction: {predicted:.2f}, adjusting to current price range")
                predicted = current_price * 1.01  # 1% 상승으로 조정

            return float(predicted)

        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}, falling back to current price")
            return float(current_price)

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


    def get_model_info(self) -> Dict:
        """현재 로드된 모델 정보 반환"""
        return {
            'use_ml': self.use_ml,
            'model_loaded': self.ml_model is not None,
            'model_version': self.model_version,
            'feature_count': len(self.feature_columns),
            **self.model_info
        }


def get_prediction_service() -> PredictionService:
    """예측 서비스 인스턴스 반환"""
    return PredictionService()
