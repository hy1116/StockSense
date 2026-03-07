"""
일배치 모델 학습 스크립트

매일 실행하여:
1. 최신 데이터 수집
2. 전처리 및 피처 생성
3. XGBoost 모델 학습
4. LSTM 모델 학습
5. 성능 평가
6. DB에 학습 이력 저장
7. model_type별 이전 모델보다 성능 좋으면 활성화

Usage:
    python ml/daily_train_batch.py
    python ml/daily_train_batch.py --activate-anyway  # 성능 무관하게 활성화
"""
import sys
import os
import io
import json
import pickle
import time
import tempfile
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.ml_model import ModelTrainingHistory

from ml.logger import get_logger
logger = get_logger("daily_train_batch")

# LSTM 윈도우 크기
LSTM_WINDOW_SIZE = 20


class DailyModelTrainer:
    """일배치 모델 학습기 (XGBoost + LSTM)"""

    def __init__(self, data_dir: str = "./data", model_dir: str = "./models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 로그 디렉토리 생성
        Path("logs").mkdir(exist_ok=True)

        # DB 연결
        settings = get_settings()
        # async URL을 sync URL로 변환
        db_url = settings.database_url.replace("+asyncpg", "")
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)

        # 피처 컬럼 정의 (24개 기술적+뉴스 + 5개 재무 = 29개)
        self.feature_columns = [
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

        # XGBoost 하이퍼파라미터
        self.xgb_hyperparameters = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

        # LSTM 하이퍼파라미터
        self.lstm_hyperparameters = {
            'window_size': LSTM_WINDOW_SIZE,
            'lstm_units_1': 64,
            'lstm_units_2': 32,
            'dropout': 0.2,
            'dense_units': 16,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10
        }

    def load_training_data(self) -> Optional[pd.DataFrame]:
        """학습 데이터 로드"""
        train_path = self.data_dir / "datasets" / "train.csv"

        if not train_path.exists():
            logger.error(f"학습 데이터 파일이 없습니다: {train_path}")
            logger.info("먼저 preprocess_data.py를 실행하세요.")
            return None

        df = pd.read_csv(train_path)
        logger.info(f"학습 데이터 로드 완료: {len(df)} 레코드")

        return df

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'target_price') -> Tuple[pd.DataFrame, pd.Series]:
        """피처와 타겟 분리"""
        # 사용 가능한 피처만 선택
        available_features = [col for col in self.feature_columns if col in df.columns]

        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"누락된 피처: {missing}")

        # 타겟 컬럼이 없으면 에러
        if target_col not in df.columns:
            raise ValueError(f"타겟 컬럼 없음: {target_col}")

        # 타겟 NaN인 행 제거 (shift로 생긴 꼬리 부분)
        # reset_index: dropna로 생긴 label 인덱스 구멍을 없애야
        # train_lstm의 groupby 시 numpy positional indexing이 올바르게 동작함
        df_clean = df.dropna(subset=[target_col]).reset_index(drop=True)

        X = df_clean[available_features].copy()
        y = df_clean[target_col]

        # inf/NaN 클리닝 (분모 0 등으로 발생한 이상값 처리)
        inf_count = np.isinf(X.values).sum()
        nan_count = X.isnull().values.sum()
        if inf_count > 0 or nan_count > 0:
            logger.warning(f"inf={inf_count}, NaN={nan_count} 값 발견 → 0으로 대체")
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logger.info(f"피처 준비 완료: {len(available_features)}개 피처, {len(X)}개 샘플 (target={target_col})")

        return X, y, df_clean

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """XGBoost 모델 학습 및 평가"""
        start_time = time.time()

        # 데이터 분할 (시계열이므로 shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        logger.info(f"[XGBoost] 데이터 분할: Train={len(X_train)}, Test={len(X_test)}")

        # 스케일링
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 학습
        logger.info("[XGBoost] XGBRegressor 학습 시작...")
        model = XGBRegressor(**self.xgb_hyperparameters)
        model.fit(X_train_scaled, y_train)

        # 예측
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # 성능 평가
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # MAPE (0으로 나누기 방지)
        mape = np.mean(np.abs((y_test - y_test_pred) / y_test.replace(0, np.nan))) * 100

        training_duration = time.time() - start_time

        logger.info(f"[XGBoost] 학습 완료 (소요시간: {training_duration:.2f}초)")
        logger.info(f"[XGBoost] Train R²: {train_score:.4f}")
        logger.info(f"[XGBoost] Test R²: {test_score:.4f}")
        logger.info(f"[XGBoost] MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        return {
            'model': model,
            'scaler': scaler,
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'training_duration': training_duration,
            'feature_columns': list(X.columns),
            'y_test': y_test.values,
            'y_test_pred': y_test_pred,
        }

    def train_lstm(self, X: pd.DataFrame, y: pd.Series, df: pd.DataFrame = None) -> Optional[Dict]:
        """LSTM 모델 학습 및 평가"""
        try:
            import keras
            from keras import layers, callbacks
        except ImportError:
            logger.warning("[LSTM] Keras를 import할 수 없습니다. LSTM 학습을 건너뜁니다.")
            return None

        start_time = time.time()
        window_size = self.lstm_hyperparameters['window_size']

        if len(X) < window_size + 50:
            logger.warning(f"[LSTM] 데이터가 부족합니다 (필요: {window_size + 50}, 현재: {len(X)})")
            return None

        logger.info(f"[LSTM] LSTM 학습 시작 (window_size={window_size})...")

        # 피처 스케일링 (전체 기준)
        feature_scaler = MinMaxScaler()
        X_scaled = feature_scaler.fit_transform(X)

        # 타겟 스케일링: 수익률(return) → StandardScaler, 절대가격 → MinMaxScaler
        from sklearn.preprocessing import StandardScaler
        y_vals = y.values
        is_return_target = (np.abs(y_vals) < 1).mean() > 0.95  # 대부분 -1~1 범위면 수익률로 판단
        target_scaler = StandardScaler() if is_return_target else MinMaxScaler()
        y_scaled = target_scaler.fit_transform(y_vals.reshape(-1, 1)).flatten()

        logger.info(f"[LSTM] 타겟 유형: {'수익률(return)' if is_return_target else '절대가격(price)'}")

        # 종목별 시계열 분할 + 시퀀스 생성
        # 각 종목의 마지막 20%를 테스트셋으로 사용 → 종목 분포 편향 방지
        X_train_list, y_train_list = [], []
        X_test_list, y_test_list = [], []

        if df is not None and 'stock_code' in df.columns:
            for _, group in df.groupby('stock_code'):
                pos_idx = group.index.tolist()
                if len(pos_idx) < window_size + 10:
                    continue
                X_s = X_scaled[pos_idx]
                y_s = y_scaled[pos_idx]

                # 종목 내 시계열 분할 (각 종목 마지막 20% = 테스트)
                stock_split = int(len(X_s) * 0.8)

                seqs_x, seqs_y = [], []
                for i in range(window_size, len(X_s)):
                    seqs_x.append(X_s[i - window_size:i])
                    seqs_y.append(y_s[i])

                # 시퀀스 기준으로도 80/20 분할
                seq_split = int(len(seqs_x) * 0.8)
                X_train_list.extend(seqs_x[:seq_split])
                y_train_list.extend(seqs_y[:seq_split])
                X_test_list.extend(seqs_x[seq_split:])
                y_test_list.extend(seqs_y[seq_split:])

            logger.info(f"[LSTM] 종목별 시계열 분할: Train={len(X_train_list)}, Test={len(X_test_list)}")
        else:
            logger.warning("[LSTM] stock_code 없음 — 전체 배열 기준 분할 (fallback)")
            for i in range(window_size, len(X_scaled)):
                X_train_list.append(X_scaled[i - window_size:i])
                y_train_list.append(y_scaled[i])
            split_idx = int(len(X_train_list) * 0.8)
            X_test_list = X_train_list[split_idx:]
            y_test_list = y_train_list[split_idx:]
            X_train_list = X_train_list[:split_idx]
            y_train_list = y_train_list[:split_idx]

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)

        logger.info(f"[LSTM] 데이터 분할: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"[LSTM] 입력 형태: {X_train.shape}")  # (samples, window, features)

        # LSTM 모델 구성
        model = keras.Sequential([
            layers.LSTM(
                self.lstm_hyperparameters['lstm_units_1'],
                return_sequences=True,
                input_shape=(window_size, X_train.shape[2])
            ),
            layers.Dropout(self.lstm_hyperparameters['dropout']),
            layers.LSTM(self.lstm_hyperparameters['lstm_units_2']),
            layers.Dropout(self.lstm_hyperparameters['dropout']),
            layers.Dense(self.lstm_hyperparameters['dense_units'], activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # 콜백
        cb = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.lstm_hyperparameters['patience'],
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.lstm_hyperparameters['epochs'],
            batch_size=self.lstm_hyperparameters['batch_size'],
            callbacks=cb,
            verbose=1
        )

        # 예측 및 역변환
        y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()
        y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()

        y_train_actual = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

        # 성능 평가
        train_score = r2_score(y_train_actual, y_train_pred)
        test_score = r2_score(y_test_actual, y_test_pred)
        mae = mean_absolute_error(y_test_actual, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))

        # MAPE
        non_zero = y_test_actual != 0
        mape = np.mean(np.abs((y_test_actual[non_zero] - y_test_pred[non_zero]) / y_test_actual[non_zero])) * 100

        training_duration = time.time() - start_time

        logger.info(f"[LSTM] 학습 완료 (소요시간: {training_duration:.2f}초)")
        logger.info(f"[LSTM] Train R²: {train_score:.4f}")
        logger.info(f"[LSTM] Test R²: {test_score:.4f}")
        logger.info(f"[LSTM] MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

        # 모델을 BytesIO로 직렬화 (.keras 포맷)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp:
                tmp_path = tmp.name
            model.save(tmp_path)
            with open(tmp_path, 'rb') as f:
                model_bytes = f.read()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 스케일러를 dict로 직렬화
        scaler_dict = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }

        return {
            'model_bytes': model_bytes,
            'scaler_dict': scaler_dict,
            'train_score': train_score,
            'test_score': test_score,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
            'training_duration': training_duration,
            'feature_columns': list(X.columns),
            'y_test': y_test_actual,
            'y_test_pred': y_test_pred,
        }

    def get_current_active_model_by_type(self, model_type: str) -> Optional[ModelTrainingHistory]:
        """특정 model_type의 현재 활성화된 모델 조회"""
        with self.Session() as session:
            result = session.execute(
                select(ModelTrainingHistory)
                .where(ModelTrainingHistory.is_active == True)
                .where(ModelTrainingHistory.model_type == model_type)
                .order_by(ModelTrainingHistory.trained_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    def get_next_version(self) -> str:
        """다음 버전 번호 생성"""
        with self.Session() as session:
            result = session.execute(
                select(ModelTrainingHistory)
                .order_by(ModelTrainingHistory.id.desc())
                .limit(1)
            )
            latest = result.scalar_one_or_none()

            if latest:
                # v1, v2, ... 형식에서 숫자 추출
                try:
                    version_num = int(latest.model_version.replace('v', ''))
                    return f"v{version_num + 1}"
                except:
                    pass

            return "v1"

    def save_xgboost_to_db(self, results: Dict, activate: bool = False, learned_weights: Optional[Tuple[float, float]] = None, horizon_days: int = 1) -> ModelTrainingHistory:
        """XGBoost 학습 결과를 DB에 저장"""
        model_binary = pickle.dumps(results['model'])
        scaler_binary = pickle.dumps(results['scaler'])

        version = self.get_next_version()
        model_type = f"XGBoostRegressor_{horizon_days}d"
        model_name = f"xgboost_{horizon_days}d_{version}.pkl"

        with self.Session() as session:
            # model_type별 활성화 (같은 type만 비활성화)
            if activate:
                session.execute(
                    update(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .where(ModelTrainingHistory.model_type == model_type)
                    .values(is_active=False)
                )

            history = ModelTrainingHistory(
                model_name=model_name,
                model_type=model_type,
                model_version=version,
                hyperparameters=json.dumps(self.xgb_hyperparameters),
                feature_columns=json.dumps(results['feature_columns']),
                scaler_type="MinMaxScaler",
                train_samples=results['train_samples'],
                test_samples=results['test_samples'],
                total_samples=results['total_samples'],
                train_score=float(results['train_score']),
                test_score=float(results['test_score']),
                mae=float(results['mae']) if results['mae'] is not None else None,
                rmse=float(results['rmse']) if results['rmse'] is not None else None,
                mape=float(results['mape']) if results['mape'] is not None else None,
                model_binary=model_binary,
                scaler_binary=scaler_binary,
                is_active=activate,
                trained_by="batch",
                training_duration_sec=float(results['training_duration']),
                notes=json.dumps({
                    "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "horizon_days": horizon_days,
                    "ensemble_xgb_weight": round(learned_weights[0], 4) if learned_weights else None,
                    "ensemble_lstm_weight": round(learned_weights[1], 4) if learned_weights else None,
                })
            )

            session.add(history)
            session.commit()
            session.refresh(history)

            logger.info(f"[XGBoost_{horizon_days}d] DB 저장 완료: ID={history.id}, Version={version}, Active={activate}")
            return history

    def save_lstm_to_db(self, results: Dict, activate: bool = False, learned_weights: Optional[Tuple[float, float]] = None, horizon_days: int = 1) -> ModelTrainingHistory:
        """LSTM 학습 결과를 DB에 저장"""
        model_binary = results['model_bytes']
        scaler_binary = pickle.dumps(results['scaler_dict'])

        version = self.get_next_version()
        model_type = f"LSTM_{horizon_days}d"
        model_name = f"lstm_{horizon_days}d_{version}.keras"

        with self.Session() as session:
            # model_type별 활성화 (같은 type만 비활성화)
            if activate:
                session.execute(
                    update(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .where(ModelTrainingHistory.model_type == model_type)
                    .values(is_active=False)
                )

            history = ModelTrainingHistory(
                model_name=model_name,
                model_type=model_type,
                model_version=version,
                hyperparameters=json.dumps(self.lstm_hyperparameters),
                feature_columns=json.dumps(results['feature_columns']),
                scaler_type="MinMaxScaler",
                train_samples=results['train_samples'],
                test_samples=results['test_samples'],
                total_samples=results['total_samples'],
                train_score=float(results['train_score']),
                test_score=float(results['test_score']),
                mae=float(results['mae']) if results['mae'] is not None else None,
                rmse=float(results['rmse']) if results['rmse'] is not None else None,
                mape=float(results['mape']) if results['mape'] is not None else None,
                model_binary=model_binary,
                scaler_binary=scaler_binary,
                is_active=activate,
                trained_by="batch",
                training_duration_sec=float(results['training_duration']),
                notes=json.dumps({
                    "trained_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "horizon_days": horizon_days,
                    "ensemble_xgb_weight": round(learned_weights[0], 4) if learned_weights else None,
                    "ensemble_lstm_weight": round(learned_weights[1], 4) if learned_weights else None,
                })
            )

            session.add(history)
            session.commit()
            session.refresh(history)

            logger.info(f"[LSTM_{horizon_days}d] DB 저장 완료: ID={history.id}, Version={version}, Active={activate}")
            return history

    def save_model_files(self, results: Dict, version: str):
        """모델 파일로도 저장 (백업용)"""
        model_path = self.model_dir / f"stock_prediction_{version}.pkl"
        scaler_path = self.model_dir / f"scaler_{version}.pkl"
        metadata_path = self.model_dir / f"metadata_{version}.json"

        # 모델 저장
        with open(model_path, 'wb') as f:
            pickle.dump(results['model'], f)

        # 스케일러 저장
        with open(scaler_path, 'wb') as f:
            pickle.dump(results['scaler'], f)

        # 메타데이터 저장
        metadata = {
            'model_name': f"stock_prediction_{version}.pkl",
            'model_type': 'XGBoostRegressor',
            'version': version,
            'trained_at': datetime.now().isoformat(),
            'train_score': results['train_score'],
            'test_score': results['test_score'],
            'mae': results['mae'],
            'rmse': results['rmse'],
            'mape': results['mape'],
            'n_samples': results['total_samples'],
            'feature_columns': results['feature_columns'],
            'hyperparameters': self.xgb_hyperparameters
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"모델 파일 저장: {model_path}")

        # 최신 버전 복사
        import shutil
        latest_model = self.model_dir / "stock_prediction_v1.pkl"
        latest_scaler = self.model_dir / "scaler.pkl"
        latest_metadata = self.model_dir / "metadata.json"

        shutil.copy(model_path, latest_model)
        shutil.copy(scaler_path, latest_scaler)
        shutil.copy(metadata_path, latest_metadata)

        logger.info("최신 모델 파일 업데이트 완료")

    def learn_ensemble_weights(self, xgb_results: Dict, lstm_results: Dict) -> Tuple[float, float]:
        """두 모델의 test set 예측으로 최적 앙상블 가중치 학습 (1D grid search)"""
        try:
            from scipy.optimize import minimize_scalar

            xgb_pred = np.array(xgb_results['y_test_pred'])
            lstm_pred = np.array(lstm_results['y_test_pred'])
            xgb_true = np.array(xgb_results['y_test'])
            lstm_true = np.array(lstm_results['y_test'])

            # 두 모델의 test set 길이가 다를 수 있음 (LSTM은 window_size 때문에 짧을 수 있음)
            # 마지막 n개 공통 구간 사용 (둘 다 같은 최신 데이터 예측 부분)
            n = min(len(xgb_pred), len(lstm_pred))
            xp = xgb_pred[-n:]
            lp = lstm_pred[-n:]
            # ground truth는 XGBoost 기준 (거의 동일한 기간)
            yt = xgb_true[-n:]

            def mse(w):
                pred = w * xp + (1.0 - w) * lp
                return float(np.mean((pred - yt) ** 2))

            result = minimize_scalar(mse, bounds=(0.0, 1.0), method='bounded')
            xgb_w = float(np.clip(result.x, 0.0, 1.0))
            lstm_w = 1.0 - xgb_w

            logger.info(f"[앙상블 가중치 학습] XGB={xgb_w:.3f}, LSTM={lstm_w:.3f} "
                        f"(MSE at optimal: {result.fun:.2f})")
            return xgb_w, lstm_w

        except Exception as e:
            logger.warning(f"[앙상블 가중치 학습 실패] {e} — 기본값 사용 (XGB=0.6, LSTM=0.4)")
            return 0.6, 0.4

    def should_activate(self, results: Dict, current_model: Optional[ModelTrainingHistory], min_score: float = 0.5) -> bool:
        """새 모델을 활성화할지 결정"""
        # 최소 성능 기준 미달이면 무조건 비활성화
        if results['test_score'] < min_score:
            logger.warning(f"성능 기준 미달 (test_score={results['test_score']:.4f} < {min_score}) - 비활성화")
            return False

        if current_model is None:
            logger.info(f"기존 활성 모델 없음 - 새 모델 활성화 (test_score={results['test_score']:.4f})")
            return True

        # Test R² 비교 (새 모델이 더 좋으면 활성화)
        improvement = results['test_score'] - current_model.test_score

        if improvement > 0:
            logger.info(f"성능 개선: {current_model.test_score:.4f} → {results['test_score']:.4f} (+{improvement:.4f})")
            return True
        else:
            logger.info(f"성능 개선 없음: {current_model.test_score:.4f} → {results['test_score']:.4f} ({improvement:.4f})")
            return False

    def run(self, activate_anyway: bool = False):
        """전체 배치 실행 (단기 1d + 장기 20d, XGBoost + LSTM)"""
        logger.info("=" * 60)
        logger.info("일배치 모델 학습 시작 (1d + 20d, XGBoost + LSTM)")
        logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        horizons = [
            {'days': 1,  'target_col': 'target_price',    'label': '단기(1일)'},
            {'days': 20, 'target_col': 'target_price_20d', 'label': '장기(20거래일)'},
        ]

        try:
            # 1. 데이터 로드 (1회만)
            df = self.load_training_data()
            if df is None:
                return False

            overall_success = True

            for h in horizons:
                horizon_days = h['days']
                target_col = h['target_col']
                label = h['label']

                if target_col not in df.columns:
                    logger.warning(f"[{label}] 타겟 컬럼 없음 ({target_col}) — 건너뜀")
                    overall_success = False
                    continue

                logger.info("=" * 60)
                logger.info(f"[{label}] 학습 시작 (horizon={horizon_days}d, target={target_col})")
                logger.info("=" * 60)

                # 2. 피처 준비 (해당 horizon 타겟 기준으로 NaN 제거)
                X, y, df_h = self.prepare_features(df, target_col=target_col)

                # LSTM은 수익률 타겟 사용 (종목 간 가격 스케일 차이 제거)
                lstm_target_col = 'target_return' if horizon_days == 1 else 'target_return_20d'
                if lstm_target_col in df_h.columns:
                    y_lstm = df_h[lstm_target_col]
                    logger.info(f"[LSTM_{horizon_days}d] 수익률 타겟 사용: {lstm_target_col}")
                else:
                    y_lstm = y  # fallback
                    logger.warning(f"[LSTM_{horizon_days}d] {lstm_target_col} 없음 — 절대가격 타겟 사용")

                # === XGBoost 학습 ===
                logger.info(f"[XGBoost_{horizon_days}d] 학습 시작")
                xgb_results = self.train_xgboost(X, y)

                # === LSTM 학습 ===
                logger.info(f"[LSTM_{horizon_days}d] 학습 시작")
                lstm_results = self.train_lstm(X, y_lstm, df=df_h)

                # === 앙상블 가중치 학습 ===
                learned_weights = None
                if lstm_results is not None:
                    logger.info(f"[앙상블_{horizon_days}d] 최적 가중치 학습")
                    learned_weights = self.learn_ensemble_weights(xgb_results, lstm_results)

                # XGBoost 활성화 판단
                current_xgb = self.get_current_active_model_by_type(f'XGBoostRegressor_{horizon_days}d')
                xgb_activate = activate_anyway or self.should_activate(xgb_results, current_xgb)
                xgb_history = self.save_xgboost_to_db(
                    xgb_results, activate=xgb_activate,
                    learned_weights=learned_weights, horizon_days=horizon_days
                )
                if xgb_activate and horizon_days == 1:
                    self.save_model_files(xgb_results, xgb_history.model_version)

                # LSTM 활성화 판단
                lstm_history = None
                if lstm_results is not None:
                    current_lstm = self.get_current_active_model_by_type(f'LSTM_{horizon_days}d')
                    # LSTM은 tabular 데이터 특성상 XGBoost보다 R² 낮게 나오는 경향 → 기준 완화
                    lstm_activate = activate_anyway or self.should_activate(lstm_results, current_lstm, min_score=0.2)
                    lstm_history = self.save_lstm_to_db(
                        lstm_results, activate=lstm_activate,
                        learned_weights=learned_weights, horizon_days=horizon_days
                    )
                else:
                    logger.warning(f"[LSTM_{horizon_days}d] LSTM 학습 건너뜀")

                # 결과 요약
                logger.info(f"[{label}] XGBoost: R²={xgb_results['test_score']:.4f}, Active={xgb_activate}")
                if lstm_history:
                    logger.info(f"[{label}] LSTM: R²={lstm_results['test_score']:.4f}, Active={lstm_history.is_active}")
                if learned_weights:
                    logger.info(f"[{label}] 앙상블 가중치: XGB={learned_weights[0]:.3f}, LSTM={learned_weights[1]:.3f}")

            logger.info("=" * 60)
            logger.info("일배치 모델 학습 완료")
            logger.info("=" * 60)

            return overall_success

        except Exception as e:
            logger.error(f"배치 실행 실패: {e}", exc_info=True)
            return False


def main():
    parser = argparse.ArgumentParser(description="일배치 ML 모델 학습 (XGBoost + LSTM)")
    parser.add_argument(
        '--activate-anyway',
        action='store_true',
        help='성능 무관하게 새 모델 활성화'
    )

    args = parser.parse_args()

    trainer = DailyModelTrainer()
    success = trainer.run(activate_anyway=args.activate_anyway)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
