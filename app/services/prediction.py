"""주가 예측 서비스 (XGBoost + LSTM 앙상블)"""
import logging
import os
import pickle
import json
import tempfile
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

# LSTM 윈도우 크기 (daily_train_batch.py와 동일)
LSTM_WINDOW_SIZE = 20

# 앙상블 가중치
XGB_WEIGHT = 0.6
LSTM_WEIGHT = 0.4


class PredictionService:
    """주가 예측 서비스 (XGBoost + LSTM 앙상블)"""

    def __init__(self, use_ml: bool = True, model_dir: str = "./models", use_db: bool = True):
        self.use_ml = use_ml
        self.use_db = use_db
        self.model_dir = Path(model_dir)

        # XGBoost 모델
        self.xgb_model = None
        self.xgb_scaler = None
        self.xgb_info = {}

        # LSTM 모델
        self.lstm_model = None
        self.lstm_feature_scaler = None
        self.lstm_target_scaler = None
        self.lstm_info = {}

        self.feature_columns = []
        self.model_version = None
        self.model_info = {}

        # ML 모델 로드 시도
        if self.use_ml:
            try:
                if self.use_db:
                    self._load_ml_models_from_db()

                # DB 로드 실패 시 파일에서 XGBoost 로드
                if self.xgb_model is None:
                    self._load_ml_model_from_file()

                if self.xgb_model is not None or self.lstm_model is not None:
                    models_loaded = []
                    if self.xgb_model is not None:
                        models_loaded.append("XGBoost")
                    if self.lstm_model is not None:
                        models_loaded.append("LSTM")
                    logger.info(f"ML models loaded: {', '.join(models_loaded)}")
                else:
                    raise Exception("No model available")

            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}. Falling back to rule-based prediction.")
                self.use_ml = False

        # 룰 기반 예측용 스케일러
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _load_ml_models_from_db(self):
        """DB에서 model_type별 활성 모델 로드"""
        try:
            from app.config import get_settings
            from app.models.ml_model import ModelTrainingHistory

            settings = get_settings()
            db_url = settings.database_url.replace("+asyncpg", "")
            engine = create_engine(db_url)
            Session = sessionmaker(bind=engine)

            with Session() as session:
                # XGBoost (또는 기존 GBR) 모델 로드
                xgb_result = session.execute(
                    select(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .where(ModelTrainingHistory.model_type.in_([
                        'XGBoostRegressor', 'GradientBoostingRegressor'
                    ]))
                    .order_by(ModelTrainingHistory.trained_at.desc())
                    .limit(1)
                )
                xgb_record = xgb_result.scalar_one_or_none()

                if xgb_record and xgb_record.model_binary and xgb_record.scaler_binary:
                    self.xgb_model = pickle.loads(xgb_record.model_binary)
                    self.xgb_scaler = pickle.loads(xgb_record.scaler_binary)

                    if xgb_record.feature_columns:
                        self.feature_columns = json.loads(xgb_record.feature_columns)
                    else:
                        self.feature_columns = self._get_default_feature_columns()

                    self.model_version = xgb_record.model_version
                    self.xgb_info = {
                        'id': xgb_record.id,
                        'model_name': xgb_record.model_name,
                        'model_type': xgb_record.model_type,
                        'version': xgb_record.model_version,
                        'test_score': xgb_record.test_score,
                    }
                    self.model_info = {
                        'id': xgb_record.id,
                        'model_name': xgb_record.model_name,
                        'model_type': xgb_record.model_type,
                        'version': xgb_record.model_version,
                        'trained_at': xgb_record.trained_at.isoformat() if xgb_record.trained_at else None,
                        'test_score': xgb_record.test_score,
                        'mae': xgb_record.mae,
                        'rmse': xgb_record.rmse,
                        'train_samples': xgb_record.train_samples
                    }
                    logger.info(f"XGBoost loaded from DB: {xgb_record.model_name} "
                              f"(test_score={xgb_record.test_score:.4f})")
                else:
                    logger.info("No active XGBoost/GBR model found in DB")

                # LSTM 모델 로드
                lstm_result = session.execute(
                    select(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .where(ModelTrainingHistory.model_type == 'LSTM')
                    .order_by(ModelTrainingHistory.trained_at.desc())
                    .limit(1)
                )
                lstm_record = lstm_result.scalar_one_or_none()

                if lstm_record and lstm_record.model_binary and lstm_record.scaler_binary:
                    self._load_lstm_from_record(lstm_record)
                else:
                    logger.info("No active LSTM model found in DB")

        except Exception as e:
            logger.warning(f"Failed to load models from DB: {e}")

    def _load_lstm_from_record(self, record):
        """DB 레코드에서 LSTM 모델 로드"""
        tmp_path = None
        try:
            import keras

            # .keras 바이너리를 임시 파일로 저장 후 로드
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                tmp.write(record.model_binary)
                tmp_path = tmp.name

            self.lstm_model = keras.models.load_model(tmp_path)

            # 스케일러 로드 (dict: feature_scaler, target_scaler)
            scaler_dict = pickle.loads(record.scaler_binary)
            self.lstm_feature_scaler = scaler_dict['feature_scaler']
            self.lstm_target_scaler = scaler_dict['target_scaler']

            # 피처 컬럼이 아직 없으면 LSTM에서 로드
            if not self.feature_columns and record.feature_columns:
                self.feature_columns = json.loads(record.feature_columns)

            self.lstm_info = {
                'id': record.id,
                'model_name': record.model_name,
                'model_type': record.model_type,
                'version': record.model_version,
                'test_score': record.test_score,
            }
            logger.info(f"LSTM loaded from DB: {record.model_name} "
                      f"(test_score={record.test_score:.4f})")

        except Exception as e:
            logger.warning(f"Failed to load LSTM model: {e}")
            self.lstm_model = None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _get_default_feature_columns(self) -> list:
        """기본 피처 컬럼 반환"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_diff',
            'price_change_1d', 'volume_change',
            'news_sentiment_avg', 'news_count',
            'news_positive_ratio', 'news_negative_ratio'
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

        self.xgb_model = joblib.load(model_path)
        self.xgb_scaler = joblib.load(scaler_path)

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
                logger.info(f"Model loaded from file: {model_path.name} "
                          f"(version={self.model_version})")
        else:
            self.feature_columns = self._get_default_feature_columns()
            self.model_version = 'file'
            self.model_info = {'source': 'file'}
            logger.info(f"Model loaded from file (no metadata): {model_path.name}")

    def predict_price(self, stock_code: str, stock_name: str, chart_data: List[Dict]) -> Dict:
        """주가 예측"""
        try:
            logger.info(f"Prediction start for {stock_code}: chart_data length={len(chart_data) if chart_data else 0}, use_ml={self.use_ml}")

            if not chart_data or len(chart_data) < 5:
                logger.warning(f"Insufficient chart data for {stock_code}: len={len(chart_data) if chart_data else 0}")
                return self._get_default_prediction(stock_code, stock_name)

            # 최근 데이터 (KIS API는 최신 데이터가 앞에 있음)
            recent_data = chart_data[:30]

            # OHLCV 데이터 추출
            close_prices = np.array([int(item.get("stck_clpr", 0)) for item in recent_data])
            open_prices = np.array([int(item.get("stck_oprc", 0)) for item in recent_data])
            high_prices = np.array([int(item.get("stck_hgpr", 0)) for item in recent_data])
            low_prices = np.array([int(item.get("stck_lwpr", 0)) for item in recent_data])
            volumes = np.array([int(item.get("acml_vol", 0)) for item in recent_data])

            if len(close_prices) == 0 or close_prices[0] == 0:
                return self._get_default_prediction(stock_code, stock_name)

            current_price = int(close_prices[0])

            # 기술적 지표 계산
            ma5 = np.mean(close_prices[:5]) if len(close_prices) >= 5 else current_price
            ma10 = np.mean(close_prices[:10]) if len(close_prices) >= 10 else current_price
            ma20 = np.mean(close_prices[:20]) if len(close_prices) >= 20 else current_price

            rsi = self._calculate_rsi(close_prices)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
            macd, signal = self._calculate_macd(close_prices)

            trend = self._analyze_trend(current_price, ma5, ma10, ma20, rsi)

            # 예측 가격 계산
            has_any_model = self.use_ml and (self.xgb_model is not None or self.lstm_model is not None)

            if has_any_model:
                predicted_price = self._predict_ensemble(
                    current_price, open_prices[0], high_prices[0], low_prices[0], volumes[0],
                    ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                    close_prices, open_prices, high_prices, low_prices, volumes,
                    stock_code=stock_code, chart_data=chart_data
                )
                logger.info(f"Ensemble prediction for {stock_code}: {predicted_price:.2f}")
            else:
                predicted_price = self._predict_advanced(
                    current_price, ma5, ma10, ma20, rsi,
                    bb_upper, bb_middle, bb_lower, macd, signal
                )
                logger.info(f"Rule-based prediction for {stock_code}: {predicted_price:.2f}")

            # 신뢰도 계산
            confidence = self._calculate_confidence_advanced(
                volumes, close_prices, current_price, bb_upper, bb_lower
            )

            if has_any_model:
                confidence = min(0.95, confidence + 0.1)

            recommendation = self._get_recommendation(
                current_price, predicted_price, trend, rsi, macd, signal
            )

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

    def _predict_ensemble(self, current_price, open_price, high_price, low_price, volume,
                          ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                          close_prices, open_prices, high_prices, low_prices, volumes,
                          stock_code=None, chart_data=None) -> float:
        """XGBoost + LSTM 앙상블 예측"""
        xgb_pred = None
        lstm_pred = None

        # XGBoost 예측
        if self.xgb_model is not None:
            xgb_pred = self._predict_with_xgb(
                current_price, open_price, high_price, low_price, volume,
                ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                close_prices, stock_code=stock_code
            )
            logger.info(f"[XGBoost] prediction: {xgb_pred:.2f}")

        # LSTM 예측
        if self.lstm_model is not None and chart_data is not None:
            lstm_pred = self._predict_with_lstm(chart_data, stock_code=stock_code)
            if lstm_pred is not None:
                logger.info(f"[LSTM] prediction: {lstm_pred:.2f}")

        # 앙상블 결합
        if xgb_pred is not None and lstm_pred is not None:
            predicted = XGB_WEIGHT * xgb_pred + LSTM_WEIGHT * lstm_pred
            logger.info(f"[Ensemble] {XGB_WEIGHT}*XGB + {LSTM_WEIGHT}*LSTM = {predicted:.2f}")
        elif xgb_pred is not None:
            predicted = xgb_pred
        elif lstm_pred is not None:
            predicted = lstm_pred
        else:
            predicted = float(current_price)

        return predicted

    def _predict_with_xgb(self, current_price, open_price, high_price, low_price, volume,
                          ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                          close_prices, stock_code=None) -> float:
        """XGBoost 모델을 사용한 예측"""
        try:
            volume_change = 0.0
            price_change_1d = 0.0

            if len(close_prices) >= 2:
                price_change_1d = (close_prices[0] - close_prices[1]) / close_prices[1]

            news_features = self._get_news_sentiment_features(stock_code) if stock_code else {
                'news_sentiment_avg': 0.0, 'news_count': 0,
                'news_positive_ratio': 0.0, 'news_negative_ratio': 0.0
            }

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
                'volume_change': volume_change,
                **news_features
            }

            features = [features_dict.get(col, 0) for col in self.feature_columns]
            X = pd.DataFrame([features], columns=self.feature_columns)
            X_scaled = self.xgb_scaler.transform(X)
            predicted = self.xgb_model.predict(X_scaled)[0]

            if predicted <= 0 or predicted > current_price * 2:
                logger.warning(f"Abnormal XGBoost prediction: {predicted:.2f}, adjusting")
                predicted = current_price * 1.01

            return float(predicted)

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return float(current_price)

    def _predict_with_lstm(self, chart_data: List[Dict], stock_code: str = None) -> Optional[float]:
        """LSTM 모델을 사용한 예측 (시계열 윈도우)"""
        try:
            if len(chart_data) < LSTM_WINDOW_SIZE:
                logger.warning(f"Insufficient data for LSTM: need {LSTM_WINDOW_SIZE}, got {len(chart_data)}")
                return None

            # chart_data → 일별 피처 DataFrame (시간순 정렬: 과거→현재)
            features_df = self._compute_features_timeseries(chart_data, stock_code)

            if features_df is None or len(features_df) < LSTM_WINDOW_SIZE:
                logger.warning("Could not compute enough timeseries features for LSTM")
                return None

            # 최근 LSTM_WINDOW_SIZE 일 피처
            recent_features = features_df.tail(LSTM_WINDOW_SIZE).values

            # 스케일링
            scaled = self.lstm_feature_scaler.transform(recent_features)

            # (1, window, features) 텐서
            X_input = scaled.reshape(1, LSTM_WINDOW_SIZE, -1)

            # 예측 (스케일된 값)
            pred_scaled = self.lstm_model.predict(X_input, verbose=0)[0][0]

            # 역변환
            predicted = self.lstm_target_scaler.inverse_transform(
                np.array([[pred_scaled]])
            )[0][0]

            if predicted <= 0:
                logger.warning(f"Abnormal LSTM prediction: {predicted:.2f}")
                return None

            return float(predicted)

        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return None

    def _compute_features_timeseries(self, chart_data: List[Dict], stock_code: str = None) -> Optional[pd.DataFrame]:
        """chart_data를 일별 피처 DataFrame으로 변환 (시간순)"""
        try:
            # chart_data를 DataFrame으로 (최신이 앞에 있으므로 역순 정렬)
            rows = []
            for item in reversed(chart_data):
                rows.append({
                    'open': int(item.get("stck_oprc", 0)),
                    'high': int(item.get("stck_hgpr", 0)),
                    'low': int(item.get("stck_lwpr", 0)),
                    'close': int(item.get("stck_clpr", 0)),
                    'volume': int(item.get("acml_vol", 0)),
                })

            df = pd.DataFrame(rows)

            if len(df) == 0 or df['close'].iloc[0] == 0:
                return None

            # 기술적 지표 생성 (preprocess_data.py와 동일 로직)
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()

            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # 볼린저 밴드
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']

            # 변화율
            df['price_change_1d'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()

            # 뉴스 감성 피처 (전체 기간 동일값 사용)
            news_features = self._get_news_sentiment_features(stock_code) if stock_code else {
                'news_sentiment_avg': 0.0, 'news_count': 0,
                'news_positive_ratio': 0.0, 'news_negative_ratio': 0.0
            }
            for key, val in news_features.items():
                df[key] = val

            # NaN 제거
            df = df.dropna()

            # feature_columns 순서로 정렬
            available = [col for col in self.feature_columns if col in df.columns]
            if len(available) < len(self.feature_columns):
                missing = set(self.feature_columns) - set(available)
                for col in missing:
                    df[col] = 0

            return df[self.feature_columns]

        except Exception as e:
            logger.error(f"Failed to compute timeseries features: {e}")
            return None

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0

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

        prices_reversed = prices[::-1]

        ema_fast = self._calculate_ema(prices_reversed, fast)
        ema_slow = self._calculate_ema(prices_reversed, slow)

        macd_line = ema_fast - ema_slow
        signal_line = macd_line

        return float(macd_line), float(signal_line)

    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """EMA 계산"""
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
        """룰 기반 예측 (fallback)"""
        predicted = current

        ma_prediction = (ma5 * 0.5 + ma10 * 0.3 + ma20 * 0.2)

        if rsi > 70:
            rsi_factor = -0.02
        elif rsi < 30:
            rsi_factor = 0.02
        else:
            rsi_factor = (50 - rsi) / 5000

        bb_position = (current - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        if bb_position > 0.8:
            bb_factor = -0.01
        elif bb_position < 0.2:
            bb_factor = 0.01
        else:
            bb_factor = 0

        if macd > signal:
            macd_factor = 0.01
        else:
            macd_factor = -0.01

        total_factor = 1 + rsi_factor + bb_factor + macd_factor
        predicted = ma_prediction * 0.3 + current * 0.7 * total_factor

        return float(predicted)

    def _calculate_confidence_advanced(self, volumes: np.ndarray, prices: np.ndarray,
                                      current: float, bb_upper: float, bb_lower: float) -> float:
        """고급 신뢰도 계산"""
        confidence = 0.5

        if len(volumes) >= 5:
            recent_volume = np.mean(volumes[:5])
            avg_volume = np.mean(volumes)

            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                if volume_ratio > 1.5:
                    confidence += 0.2
                elif volume_ratio > 1.0:
                    confidence += 0.1

        if len(prices) >= 5:
            volatility = np.std(prices[:5]) / np.mean(prices[:5]) if np.mean(prices[:5]) > 0 else 0
            if volatility < 0.02:
                confidence += 0.2
            elif volatility < 0.05:
                confidence += 0.1
            else:
                confidence -= 0.1

        if bb_upper != bb_lower:
            bb_width = (bb_upper - bb_lower) / current
            if bb_width < 0.1:
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

        volume_ratio = recent_volume / avg_volume

        if volume_ratio > 1.5:
            return min(0.9, 0.6 + (volume_ratio - 1.5) * 0.2)
        elif volume_ratio < 0.5:
            return max(0.3, 0.6 - (0.5 - volume_ratio) * 0.2)
        else:
            return 0.6

    def _get_recommendation(self, current: float, predicted: float, trend: str,
                           rsi: float, macd: float, signal: float) -> str:
        """투자의견 생성"""
        change_rate = ((predicted - current) / current) * 100

        score = 0

        if change_rate > 3:
            score += 2
        elif change_rate > 1:
            score += 1
        elif change_rate < -3:
            score -= 2
        elif change_rate < -1:
            score -= 1

        if trend == "강한 상승":
            score += 2
        elif trend == "상승":
            score += 1
        elif trend == "강한 하락":
            score -= 2
        elif trend == "하락":
            score -= 1

        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1

        if macd > signal:
            score += 1
        else:
            score -= 1

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

    def _get_news_sentiment_features(self, stock_code: str) -> dict:
        """DB에서 해당 종목의 최근 뉴스 감성 피처 조회"""
        defaults = {
            'news_sentiment_avg': 0.0,
            'news_count': 0,
            'news_positive_ratio': 0.0,
            'news_negative_ratio': 0.0
        }

        try:
            from app.config import get_settings
            from sqlalchemy import create_engine, text

            settings = get_settings()
            db_url = settings.database_url.replace("+asyncpg", "")
            engine = create_engine(db_url)

            query = text("""
                SELECT
                    COALESCE(AVG(sentiment_score), 0) as news_sentiment_avg,
                    COUNT(*) as news_count,
                    COALESCE(SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END)::float
                        / NULLIF(COUNT(*), 0), 0) as news_positive_ratio,
                    COALESCE(SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END)::float
                        / NULLIF(COUNT(*), 0), 0) as news_negative_ratio
                FROM stock_news
                WHERE stock_code = :stock_code
                    AND is_processed = true
                    AND sentiment_score IS NOT NULL
                    AND published_at >= CURRENT_DATE
            """)

            with engine.connect() as conn:
                result = conn.execute(query, {"stock_code": stock_code})
                row = result.fetchone()

                if row and row[1] > 0:
                    return {
                        'news_sentiment_avg': float(row[0]),
                        'news_count': int(row[1]),
                        'news_positive_ratio': float(row[2]),
                        'news_negative_ratio': float(row[3])
                    }

        except Exception as e:
            logger.warning(f"Failed to get news sentiment features: {e}")

        return defaults

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
            'xgb_loaded': self.xgb_model is not None,
            'lstm_loaded': self.lstm_model is not None,
            'model_version': self.model_version,
            'feature_count': len(self.feature_columns),
            **self.model_info
        }


def get_prediction_service() -> PredictionService:
    """예측 서비스 인스턴스 반환"""
    return PredictionService()
