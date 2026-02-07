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

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/daily_train.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

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

        # 피처 컬럼 정의
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma10', 'ma20', 'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'macd_diff',
            'price_change_1d', 'volume_change',
            'news_sentiment_avg', 'news_count',
            'news_positive_ratio', 'news_negative_ratio'
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

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """피처와 타겟 분리"""
        # 사용 가능한 피처만 선택
        available_features = [col for col in self.feature_columns if col in df.columns]

        if len(available_features) < len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"누락된 피처: {missing}")

        X = df[available_features]
        y = df['target_price']

        logger.info(f"피처 준비 완료: {len(available_features)}개 피처")

        return X, y

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
            'feature_columns': list(X.columns)
        }

    def train_lstm(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict]:
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

        # 피처/타겟 스케일링
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()

        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

        # 슬라이딩 윈도우로 시퀀스 데이터 생성
        X_seq, y_seq = [], []
        for i in range(window_size, len(X_scaled)):
            X_seq.append(X_scaled[i - window_size:i])
            y_seq.append(y_scaled[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # 시계열 기반 분할 (shuffle=False)
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

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
            'feature_columns': list(X.columns)
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

    def save_xgboost_to_db(self, results: Dict, activate: bool = False) -> ModelTrainingHistory:
        """XGBoost 학습 결과를 DB에 저장"""
        model_binary = pickle.dumps(results['model'])
        scaler_binary = pickle.dumps(results['scaler'])

        version = self.get_next_version()
        model_name = f"xgboost_{version}.pkl"

        with self.Session() as session:
            # model_type별 활성화 (같은 type만 비활성화)
            if activate:
                session.execute(
                    update(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .where(ModelTrainingHistory.model_type == 'XGBoostRegressor')
                    .values(is_active=False)
                )

            history = ModelTrainingHistory(
                model_name=model_name,
                model_type="XGBoostRegressor",
                model_version=version,
                hyperparameters=json.dumps(self.xgb_hyperparameters),
                feature_columns=json.dumps(results['feature_columns']),
                scaler_type="MinMaxScaler",
                train_samples=results['train_samples'],
                test_samples=results['test_samples'],
                total_samples=results['total_samples'],
                train_score=results['train_score'],
                test_score=results['test_score'],
                mae=results['mae'],
                rmse=results['rmse'],
                mape=results['mape'],
                model_binary=model_binary,
                scaler_binary=scaler_binary,
                is_active=activate,
                trained_by="batch",
                training_duration_sec=results['training_duration'],
                notes=f"XGBoost batch training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            session.add(history)
            session.commit()
            session.refresh(history)

            logger.info(f"[XGBoost] DB 저장 완료: ID={history.id}, Version={version}, Active={activate}")
            return history

    def save_lstm_to_db(self, results: Dict, activate: bool = False) -> ModelTrainingHistory:
        """LSTM 학습 결과를 DB에 저장"""
        model_binary = results['model_bytes']
        scaler_binary = pickle.dumps(results['scaler_dict'])

        version = self.get_next_version()
        model_name = f"lstm_{version}.keras"

        with self.Session() as session:
            # model_type별 활성화 (같은 type만 비활성화)
            if activate:
                session.execute(
                    update(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .where(ModelTrainingHistory.model_type == 'LSTM')
                    .values(is_active=False)
                )

            history = ModelTrainingHistory(
                model_name=model_name,
                model_type="LSTM",
                model_version=version,
                hyperparameters=json.dumps(self.lstm_hyperparameters),
                feature_columns=json.dumps(results['feature_columns']),
                scaler_type="MinMaxScaler",
                train_samples=results['train_samples'],
                test_samples=results['test_samples'],
                total_samples=results['total_samples'],
                train_score=results['train_score'],
                test_score=results['test_score'],
                mae=results['mae'],
                rmse=results['rmse'],
                mape=results['mape'],
                model_binary=model_binary,
                scaler_binary=scaler_binary,
                is_active=activate,
                trained_by="batch",
                training_duration_sec=results['training_duration'],
                notes=f"LSTM batch training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            session.add(history)
            session.commit()
            session.refresh(history)

            logger.info(f"[LSTM] DB 저장 완료: ID={history.id}, Version={version}, Active={activate}")
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

    def should_activate(self, results: Dict, current_model: Optional[ModelTrainingHistory]) -> bool:
        """새 모델을 활성화할지 결정"""
        if current_model is None:
            logger.info("기존 활성 모델 없음 - 새 모델 활성화")
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
        """전체 배치 실행 (XGBoost + LSTM)"""
        logger.info("=" * 60)
        logger.info("일배치 모델 학습 시작 (XGBoost + LSTM)")
        logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        try:
            # 1. 데이터 로드 (1회만)
            df = self.load_training_data()
            if df is None:
                return False

            # 2. 피처 준비
            X, y = self.prepare_features(df)

            # === XGBoost 학습 ===
            logger.info("-" * 40)
            logger.info("[XGBoost] 학습 시작")
            logger.info("-" * 40)

            xgb_results = self.train_xgboost(X, y)

            # XGBoost 활성화 판단 (model_type별)
            current_xgb = self.get_current_active_model_by_type('XGBoostRegressor')
            xgb_activate = activate_anyway or self.should_activate(xgb_results, current_xgb)
            xgb_history = self.save_xgboost_to_db(xgb_results, activate=xgb_activate)

            if xgb_activate:
                self.save_model_files(xgb_results, xgb_history.model_version)

            # === LSTM 학습 ===
            logger.info("-" * 40)
            logger.info("[LSTM] 학습 시작")
            logger.info("-" * 40)

            lstm_results = self.train_lstm(X, y)

            lstm_history = None
            if lstm_results is not None:
                # LSTM 활성화 판단 (model_type별)
                current_lstm = self.get_current_active_model_by_type('LSTM')
                lstm_activate = activate_anyway or self.should_activate(lstm_results, current_lstm)
                lstm_history = self.save_lstm_to_db(lstm_results, activate=lstm_activate)
            else:
                logger.warning("[LSTM] LSTM 학습 건너뜀")

            # === 결과 요약 ===
            logger.info("=" * 60)
            logger.info("일배치 모델 학습 완료")
            logger.info(f"[XGBoost] Version={xgb_history.model_version}, "
                       f"Test R²={xgb_results['test_score']:.4f}, Active={xgb_activate}")
            if lstm_history:
                logger.info(f"[LSTM] Version={lstm_history.model_version}, "
                           f"Test R²={lstm_results['test_score']:.4f}, Active={lstm_activate}")
            logger.info("=" * 60)

            return True

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
