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

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# Sync DB 엔진 싱글턴 (매 요청마다 생성 방지)
_sync_engine = None
_sync_session_factory = None

def _get_sync_db():
    """Sync DB 엔진/세션 싱글턴 반환"""
    global _sync_engine, _sync_session_factory
    if _sync_engine is None:
        from app.config import get_settings
        settings = get_settings()
        db_url = settings.database_url.replace("+asyncpg", "")
        _sync_engine = create_engine(
            db_url,
            pool_size=3,
            max_overflow=5,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        _sync_session_factory = sessionmaker(bind=_sync_engine)
    return _sync_engine, _sync_session_factory

# LSTM 윈도우 크기 (daily_train_batch.py와 동일)
LSTM_WINDOW_SIZE = 20

# 앙상블 기본 가중치 (test R² 정보 없을 때 사용)
XGB_WEIGHT_DEFAULT = 0.6
LSTM_WEIGHT_DEFAULT = 0.4


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
            from app.models.ml_model import ModelTrainingHistory

            engine, Session = _get_sync_db()

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
        """기본 피처 컬럼 반환 (29개: 기술적+뉴스+재무)"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_diff',
            'price_change_1d', 'volume_change',
            'volume_ratio', 'obv_normalized', 'mfi',
            'news_sentiment_avg', 'news_count',
            'news_positive_ratio', 'news_negative_ratio',
            'per', 'pbr', 'eps_normalized', 'div_yield', 'roe'
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
            xgb_pred = None
            lstm_pred = None
            model_used = "rule_based"

            if has_any_model:
                predicted_price, xgb_pred, lstm_pred, model_used = self._predict_ensemble(
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
                confidence = min(0.80, confidence + 0.05)

            recommendation, factor_scores = self._get_recommendation(
                current_price, predicted_price, trend, rsi, macd, signal,
                bb_upper, bb_lower, volumes, confidence=confidence
            )

            prediction_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

            # 뉴스 감성 (details용 재사용 or 재조회)
            news_features = self._get_news_sentiment_features(stock_code)

            # 재무 데이터 (details용)
            fin_data = self._get_financial_features(stock_code)

            # 볼린저밴드 현재 위치 (0~1)
            bb_position = None
            if bb_upper != bb_lower and bb_upper > 0:
                bb_position = round((current_price - bb_lower) / (bb_upper - bb_lower), 3)

            # 전일 대비 변동률
            price_change_1d = 0.0
            if len(close_prices) >= 2 and close_prices[1] > 0:
                price_change_1d = round(float((close_prices[0] - close_prices[1]) / close_prices[1] * 100), 2)

            # 앙상블 가중치 정보 (응답에 포함)
            ensemble_weights = None
            if xgb_pred is not None and lstm_pred is not None:
                xgb_r2 = max(0, self.xgb_info.get('test_score', 0))
                lstm_r2 = max(0, self.lstm_info.get('test_score', 0))
                total_r2 = xgb_r2 + lstm_r2
                if total_r2 > 0:
                    ensemble_weights = {
                        "xgb_weight": round(xgb_r2 / total_r2, 3),
                        "lstm_weight": round(lstm_r2 / total_r2, 3),
                        "xgb_r2": round(xgb_r2, 4),
                        "lstm_r2": round(lstm_r2, 4),
                    }

            details = {
                "model_used": model_used,
                "ensemble_weights": ensemble_weights,
                "xgb_predicted": round(xgb_pred, 2) if xgb_pred is not None else None,
                "lstm_predicted": round(lstm_pred, 2) if lstm_pred is not None else None,
                "technical_indicators": {
                    "ma5": round(float(ma5), 2),
                    "ma10": round(float(ma10), 2),
                    "ma20": round(float(ma20), 2),
                    "rsi": round(float(rsi), 2),
                    "bb_upper": round(float(bb_upper), 2),
                    "bb_middle": round(float(bb_middle), 2),
                    "bb_lower": round(float(bb_lower), 2),
                    "bb_position": bb_position,
                    "macd": round(float(macd), 2),
                    "macd_signal": round(float(signal), 2),
                    "macd_diff": round(float(macd - signal), 2),
                    "price_change_1d": price_change_1d,
                    "volume_ratio": round(float(volume_ratio), 3),
                    "obv_normalized": round(float(obv_normalized), 3),
                    "mfi": round(float(mfi), 2),
                },
                "news_sentiment": {
                    "score": round(float(news_features.get("news_sentiment_avg", 0)), 2),
                    "count": int(news_features.get("news_count", 0)),
                    "positive_ratio": round(float(news_features.get("news_positive_ratio", 0)), 3),
                    "negative_ratio": round(float(news_features.get("news_negative_ratio", 0)), 3),
                },
                "financial_data": {
                    "per": fin_data.get("per"),
                    "pbr": fin_data.get("pbr"),
                    "eps": fin_data.get("eps"),
                    "div_yield": fin_data.get("div_yield"),
                    "roe": fin_data.get("roe"),
                    "revenue": fin_data.get("revenue"),
                    "operating_profit": fin_data.get("operating_profit"),
                    "net_profit": fin_data.get("net_profit"),
                },
                "recommendation_factors": factor_scores,
            }

            return {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "current_price": current_price,
                "predicted_price": round(predicted_price, 2),
                "prediction_date": prediction_date,
                "confidence": round(confidence, 2),
                "trend": trend,
                "recommendation": recommendation,
                "details": details,
            }

        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            return self._get_default_prediction(stock_code, stock_name)

    def _predict_ensemble(self, current_price, open_price, high_price, low_price, volume,
                          ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                          close_prices, open_prices, high_prices, low_prices, volumes,
                          stock_code=None, chart_data=None):
        """XGBoost + LSTM 앙상블 예측 → (predicted, xgb_pred, lstm_pred, model_used)"""
        xgb_pred = None
        lstm_pred = None

        # XGBoost 예측
        if self.xgb_model is not None:
            xgb_pred = self._predict_with_xgb(
                current_price, open_price, high_price, low_price, volume,
                ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                close_prices, volumes=volumes, high_prices=high_prices, low_prices=low_prices,
                stock_code=stock_code
            )
            logger.info(f"[XGBoost] prediction: {xgb_pred:.2f}")

        # LSTM 예측
        if self.lstm_model is not None and chart_data is not None:
            lstm_pred = self._predict_with_lstm(chart_data, stock_code=stock_code)
            if lstm_pred is not None:
                logger.info(f"[LSTM] prediction: {lstm_pred:.2f}")

        # 앙상블 결합 (모델별 test R² 비례 동적 가중치)
        if xgb_pred is not None and lstm_pred is not None:
            xgb_r2 = max(0, self.xgb_info.get('test_score', 0))
            lstm_r2 = max(0, self.lstm_info.get('test_score', 0))
            total_r2 = xgb_r2 + lstm_r2

            if total_r2 > 0:
                xgb_w = xgb_r2 / total_r2
                lstm_w = lstm_r2 / total_r2
            else:
                xgb_w = XGB_WEIGHT_DEFAULT
                lstm_w = LSTM_WEIGHT_DEFAULT

            predicted = xgb_w * xgb_pred + lstm_w * lstm_pred
            model_used = "ensemble"
            logger.info(f"[Ensemble] {xgb_w:.2f}*XGB(R²={xgb_r2:.4f}) + "
                        f"{lstm_w:.2f}*LSTM(R²={lstm_r2:.4f}) = {predicted:.2f}")
        elif xgb_pred is not None:
            predicted = xgb_pred
            model_used = "xgboost"
        elif lstm_pred is not None:
            predicted = lstm_pred
            model_used = "lstm"
        else:
            predicted = float(current_price)
            model_used = "rule_based"

        return predicted, xgb_pred, lstm_pred, model_used

    def _predict_with_xgb(self, current_price, open_price, high_price, low_price, volume,
                          ma5, ma10, ma20, rsi, bb_upper, bb_middle, bb_lower, macd, signal,
                          close_prices, volumes=None, high_prices=None, low_prices=None,
                          stock_code=None) -> float:
        """XGBoost 모델을 사용한 예측"""
        try:
            price_change_1d = 0.0
            volume_change = 0.0

            if len(close_prices) >= 2:
                price_change_1d = (close_prices[0] - close_prices[1]) / close_prices[1]

            if volumes is not None and len(volumes) >= 2 and volumes[1] > 0:
                volume_change = (volumes[0] - volumes[1]) / volumes[1]

            # volume_ratio: 당일 거래량 / 20일 평균 거래량
            volume_ratio = 0.0
            if volumes is not None and len(volumes) >= 20:
                vol_ma20 = np.mean(volumes[:20])
                if vol_ma20 > 0:
                    volume_ratio = float(volumes[0] / vol_ma20)

            # obv_normalized: OBV / 20일 평균 거래량
            obv_normalized = 0.0
            if volumes is not None and close_prices is not None and len(volumes) >= 20:
                # 역순 (최신이 앞에) → 시간순으로 변환하여 계산
                cp = close_prices[:20][::-1]
                vl = volumes[:20][::-1]
                obv = 0.0
                for i in range(1, len(cp)):
                    if cp[i] > cp[i - 1]:
                        obv += vl[i]
                    elif cp[i] < cp[i - 1]:
                        obv -= vl[i]
                vol_ma20 = np.mean(volumes[:20])
                if vol_ma20 > 0:
                    obv_normalized = obv / vol_ma20

            # mfi: Money Flow Index (14일)
            mfi = 50.0
            if (volumes is not None and high_prices is not None and low_prices is not None
                    and close_prices is not None and len(close_prices) >= 15):
                # 역순 → 시간순
                cp = close_prices[:15][::-1]
                hp = high_prices[:15][::-1]
                lp = low_prices[:15][::-1]
                vl = volumes[:15][::-1]
                tp = (hp + lp + cp) / 3
                pos_flow = 0.0
                neg_flow = 0.0
                for i in range(1, len(tp)):
                    mf = tp[i] * vl[i]
                    if tp[i] > tp[i - 1]:
                        pos_flow += mf
                    else:
                        neg_flow += mf
                if neg_flow > 0:
                    mr = pos_flow / neg_flow
                    mfi = 100 - (100 / (1 + mr))
                elif pos_flow > 0:
                    mfi = 100.0

            news_features = self._get_news_sentiment_features(stock_code) if stock_code else {
                'news_sentiment_avg': 0.0, 'news_count': 0,
                'news_positive_ratio': 0.0, 'news_negative_ratio': 0.0
            }

            fin = self._get_financial_features(stock_code) if stock_code else {
                'per': 0.0, 'pbr': 0.0, 'eps': 0.0, 'div_yield': 0.0, 'roe': 0.0
            }
            eps_normalized = (fin['eps'] / current_price) if current_price > 0 and fin['eps'] else 0.0

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
                'volume_ratio': volume_ratio,
                'obv_normalized': obv_normalized,
                'mfi': mfi,
                **news_features,
                'per': fin['per'],
                'pbr': fin['pbr'],
                'eps_normalized': eps_normalized,
                'div_yield': fin['div_yield'],
                'roe': fin['roe'],
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

            # 거래량 비율
            volume_ma20 = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / volume_ma20

            # OBV 정규화
            obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
            df['obv_normalized'] = obv / volume_ma20

            # MFI (14일)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            raw_money_flow = typical_price * df['volume']
            tp_diff = typical_price.diff()
            positive_flow = raw_money_flow.where(tp_diff > 0, 0).rolling(window=14).sum()
            negative_flow = raw_money_flow.where(tp_diff <= 0, 0).rolling(window=14).sum()
            money_ratio = positive_flow / negative_flow.replace(0, np.nan)
            df['mfi'] = 100 - (100 / (1 + money_ratio))

            # 뉴스 감성 피처 (전체 기간 동일값 사용)
            news_features = self._get_news_sentiment_features(stock_code) if stock_code else {
                'news_sentiment_avg': 0.0, 'news_count': 0,
                'news_positive_ratio': 0.0, 'news_negative_ratio': 0.0
            }
            for key, val in news_features.items():
                df[key] = val

            # 재무 피처 (전체 기간 동일값 사용, EPS는 행별 현재가로 계산)
            fin = self._get_financial_features(stock_code) if stock_code else {
                'per': 0.0, 'pbr': 0.0, 'eps': 0.0, 'div_yield': 0.0, 'roe': 0.0
            }
            df['per'] = fin['per']
            df['pbr'] = fin['pbr']
            df['eps_normalized'] = df['close'].apply(
                lambda c: fin['eps'] / c if c > 0 and fin['eps'] else 0.0
            )
            df['div_yield'] = fin['div_yield']
            df['roe'] = fin['roe']

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
        """신뢰도 계산

        기본 신뢰도를 낮게 잡고 조건 충족 시 소폭 가산.
        데이터 품질(충분한 데이터, 안정적 거래량, 낮은 변동성)을 반영.
        """
        confidence = 0.35

        # 데이터 충분성 (최소 20일 이상이면 가산)
        data_len = len(prices)
        if data_len >= 20:
            confidence += 0.05
        if data_len >= 50:
            confidence += 0.05

        # 거래량 안정성 (최근 vs 평균 비율이 0.7~1.5 범위면 안정적)
        if len(volumes) >= 5:
            recent_volume = np.mean(volumes[:5])
            avg_volume = np.mean(volumes)
            if avg_volume > 0:
                volume_ratio = recent_volume / avg_volume
                if 0.7 <= volume_ratio <= 1.5:
                    confidence += 0.05
                elif volume_ratio > 1.5:
                    # 거래량 급증은 불확실성 증가
                    confidence -= 0.05

        # 가격 변동성 (낮을수록 예측 신뢰도 높음)
        if len(prices) >= 5:
            volatility = np.std(prices[:5]) / np.mean(prices[:5]) if np.mean(prices[:5]) > 0 else 0
            if volatility < 0.02:
                confidence += 0.1
            elif volatility < 0.04:
                confidence += 0.05
            elif volatility > 0.06:
                confidence -= 0.1

        # 볼린저밴드 폭 (좁을수록 안정적)
        if bb_upper and bb_lower and bb_upper != bb_lower and current > 0:
            bb_width = (bb_upper - bb_lower) / current
            if bb_width < 0.05:
                confidence += 0.05
            elif bb_width > 0.15:
                confidence -= 0.05

        return min(0.80, max(0.25, confidence))

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
                           rsi: float, macd: float, signal: float,
                           bb_upper: float = 0, bb_lower: float = 0,
                           volumes: np.ndarray = None, confidence: float = 0.5):
        """멀티팩터 가중 스코어링 기반 투자의견 → (recommendation, factor_scores)

        각 팩터를 -1.0 ~ +1.0 으로 정규화한 뒤 가중 합산.
        신뢰도 연동: confidence가 높으면 ML 예측 비중 ↑, 낮으면 기술적 지표 비중 ↑
          - confidence ≥ 0.6  → W_PRED 최대 0.45
          - confidence ≤ 0.35 → W_PRED 최소 0.15
          - 나머지 가중치는 기술적 지표에 원래 비율대로 재분배
        """
        # 신뢰도 기반 ML 예측 가중치 (기본 0.30, 범위 0.15~0.45)
        BASE_PRED = 0.30
        if confidence >= 0.6:
            W_PRED = min(0.45, BASE_PRED + (confidence - 0.6) * 0.75)
        elif confidence <= 0.35:
            W_PRED = max(0.15, BASE_PRED - (0.35 - confidence) * 1.5)
        else:
            W_PRED = BASE_PRED

        # 나머지 가중치를 기술적 지표에 원래 비율(25:20:15:10)대로 재분배
        tech_remaining = 1.0 - W_PRED
        tech_ratio_sum = 0.70  # 원래 기술적 합계 (0.25+0.20+0.15+0.10)
        W_TREND = tech_remaining * (0.25 / tech_ratio_sum)
        W_RSI = tech_remaining * (0.20 / tech_ratio_sum)
        W_MACD = tech_remaining * (0.15 / tech_ratio_sum)
        W_BB = tech_remaining * (0.10 / tech_ratio_sum)

        # --- 1) 예측 변동률 시그널 ---
        change_pct = ((predicted - current) / current) * 100 if current else 0
        pred_score = max(-1.0, min(1.0, change_pct / 5.0))

        # --- 2) 추세 시그널 (MA 정렬) ---
        trend_map = {
            "강한 상승": 1.0,
            "상승": 0.5,
            "보합": 0.0,
            "하락": -0.5,
            "강한 하락": -1.0,
        }
        trend_score = trend_map.get(trend, 0.0)

        # --- 3) RSI 시그널 (연속 매핑) ---
        if rsi <= 20:
            rsi_score = 1.0
        elif rsi <= 50:
            rsi_score = (50 - rsi) / 30.0
        elif rsi <= 80:
            rsi_score = (50 - rsi) / 30.0
        else:
            rsi_score = -1.0

        # --- 4) MACD 시그널 ---
        macd_diff = macd - signal
        if current > 0:
            norm_macd = (macd_diff / current) * 100
            macd_score = max(-1.0, min(1.0, norm_macd / 0.5))
        else:
            macd_score = 0.0

        # --- 5) 볼린저밴드 시그널 (평균회귀) ---
        if bb_upper and bb_lower and bb_upper != bb_lower:
            bb_pos = (current - bb_lower) / (bb_upper - bb_lower)
            bb_score = max(-1.0, min(1.0, (0.5 - bb_pos) * 2))
        else:
            bb_pos = 0.5
            bb_score = 0.0

        # --- 가중 합산 ---
        total = (W_PRED * pred_score
                 + W_TREND * trend_score
                 + W_RSI * rsi_score
                 + W_MACD * macd_score
                 + W_BB * bb_score)

        # --- 판정 ---
        if total >= 0.65:
            recommendation = "적극 매수"
        elif total >= 0.2:
            recommendation = "매수"
        elif total <= -0.65:
            recommendation = "적극 매도"
        elif total <= -0.2:
            recommendation = "매도"
        else:
            recommendation = "보유"

        factor_scores = {
            "pred_change_pct": round(change_pct, 2),
            "pred_score": round(pred_score, 3),
            "trend_score": round(trend_score, 3),
            "rsi_score": round(rsi_score, 3),
            "macd_score": round(macd_score, 3),
            "bb_score": round(bb_score, 3),
            "total_score": round(total, 3),
            "weights": {"pred": W_PRED, "trend": W_TREND, "rsi": W_RSI, "macd": W_MACD, "bb": W_BB},
        }

        return recommendation, factor_scores

    def _get_news_sentiment_features(self, stock_code: str) -> dict:
        """DB에서 해당 종목의 최근 뉴스 감성 피처 조회"""
        defaults = {
            'news_sentiment_avg': 0.0,
            'news_count': 0,
            'news_positive_ratio': 0.0,
            'news_negative_ratio': 0.0
        }

        try:
            engine, _ = _get_sync_db()

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

    def _get_financial_features(self, stock_code: str) -> dict:
        """DB에서 해당 종목의 최신 재무 데이터 조회"""
        defaults = {
            'per': 0.0, 'pbr': 0.0, 'eps': 0.0,
            'div_yield': 0.0, 'roe': 0.0,
            'revenue': None, 'operating_profit': None, 'net_profit': None,
        }

        try:
            engine, _ = _get_sync_db()

            query = text("""
                SELECT per, pbr, eps, bps, div_yield, roe,
                       revenue, operating_profit, net_profit
                FROM stock_financials
                WHERE stock_code = :stock_code
                ORDER BY date DESC
                LIMIT 1
            """)

            with engine.connect() as conn:
                result = conn.execute(query, {"stock_code": stock_code})
                row = result.fetchone()

                if row:
                    return {
                        'per': float(row[0]) if row[0] is not None else 0.0,
                        'pbr': float(row[1]) if row[1] is not None else 0.0,
                        'eps': float(row[2]) if row[2] is not None else 0.0,
                        'div_yield': float(row[4]) if row[4] is not None else 0.0,
                        'roe': float(row[5]) if row[5] is not None else 0.0,
                        'revenue': float(row[6]) if row[6] is not None else None,
                        'operating_profit': float(row[7]) if row[7] is not None else None,
                        'net_profit': float(row[8]) if row[8] is not None else None,
                    }

        except Exception as e:
            logger.warning(f"Failed to get financial features: {e}")

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
            "recommendation": "보유",
            "details": {
                "model_used": "rule_based",
                "xgb_predicted": None,
                "lstm_predicted": None,
                "technical_indicators": None,
                "news_sentiment": None,
                "recommendation_factors": None,
            }
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
