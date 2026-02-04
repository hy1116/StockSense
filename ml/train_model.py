import sys
import os
import pickle
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
from datetime import datetime
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class StockPredictionTrainer:
    def __init__(self, data_dir='./data', model_dir='./models', save_to_db=True):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = MinMaxScaler()
        self.save_to_db = save_to_db

        # DB 연결 설정
        if self.save_to_db:
            self._init_db()

    def _init_db(self):
        """DB 연결 초기화"""
        try:
            from dotenv import load_dotenv
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker

            load_dotenv()
            db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocksense")
            # asyncpg를 psycopg2로 변환
            db_url = db_url.replace("+asyncpg", "")

            self.engine = create_engine(db_url)
            self.Session = sessionmaker(bind=self.engine)
            print(f"✅ DB connection initialized")
        except Exception as e:
            print(f"⚠️ DB connection failed: {e}")
            self.save_to_db = False

    def _save_to_database(self, model, metadata: dict, activate: bool = False):
        """학습 결과를 DB에 저장"""
        if not self.save_to_db:
            return None

        try:
            from app.models.ml_model import ModelTrainingHistory

            # 모델 버전 생성 (날짜 기반)
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 모델과 스케일러 직렬화
            model_binary = pickle.dumps(model)
            scaler_binary = pickle.dumps(self.scaler)

            # DB 레코드 생성
            training_record = ModelTrainingHistory(
                model_name=metadata.get('model_name', 'stock_prediction'),
                model_type='GradientBoostingRegressor',
                model_version=version,
                hyperparameters=json.dumps({
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'random_state': 42
                }),
                feature_columns=json.dumps(metadata.get('feature_columns', [])),
                scaler_type='MinMaxScaler',
                train_samples=metadata.get('train_samples', 0),
                test_samples=metadata.get('test_samples', 0),
                total_samples=metadata.get('n_samples', 0),
                train_score=metadata.get('train_score', 0),
                test_score=metadata.get('test_score', 0),
                mae=metadata.get('mae'),
                rmse=metadata.get('rmse'),
                mape=metadata.get('mape'),
                model_binary=model_binary,
                scaler_binary=scaler_binary,
                is_active=False,
                is_production=False,
                trained_by='batch',
                training_duration_sec=metadata.get('training_duration_sec'),
                notes=metadata.get('notes')
            )

            with self.Session() as session:
                session.add(training_record)
                session.commit()
                record_id = training_record.id

                # 자동 활성화 로직
                if activate:
                    self._activate_model(session, record_id)

                print(f"💾 Training history saved to DB (ID: {record_id})")
                return record_id

        except Exception as e:
            print(f"⚠️ Failed to save to DB: {e}")
            return None

    def _activate_model(self, session, model_id: int):
        """모델 활성화 (기존 활성 모델 비활성화)"""
        from app.models.ml_model import ModelTrainingHistory

        # 기존 활성 모델 비활성화
        session.query(ModelTrainingHistory).filter(
            ModelTrainingHistory.is_active == True
        ).update({'is_active': False})

        # 새 모델 활성화
        session.query(ModelTrainingHistory).filter(
            ModelTrainingHistory.id == model_id
        ).update({'is_active': True})

        session.commit()
        print(f"✅ Model ID {model_id} activated!")

    def _should_activate_new_model(self, new_test_score: float) -> bool:
        """새 모델을 활성화해야 하는지 판단"""
        if not self.save_to_db:
            return True

        try:
            from app.models.ml_model import ModelTrainingHistory
            from sqlalchemy import select

            with self.Session() as session:
                result = session.execute(
                    select(ModelTrainingHistory)
                    .where(ModelTrainingHistory.is_active == True)
                    .limit(1)
                )
                current_active = result.scalar_one_or_none()

                if current_active is None:
                    print("📭 No active model found - will activate new model")
                    return True

                if new_test_score > current_active.test_score:
                    print(f"📈 New model ({new_test_score:.4f}) > Current ({current_active.test_score:.4f}) - will activate")
                    return True
                else:
                    print(f"📉 New model ({new_test_score:.4f}) <= Current ({current_active.test_score:.4f}) - keeping current")
                    return False

        except Exception as e:
            print(f"⚠️ Error checking current model: {e}")
            return True

    def load_data(self, stock_code: str = None):
        """데이터 로드"""
        if stock_code:
            # 특정 종목
            df = pd.read_csv(f'{self.data_dir}/processed/{stock_code}_features.csv')
        else:
            # 전체 종목
            df = pd.read_csv(f'{self.data_dir}/datasets/train.csv')
        return df

    def create_features(self, df: pd.DataFrame):
        """피처 엔지니어링"""
        # 기술적 지표
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

        # 가격 변화율
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        # 타겟: 다음날 종가
        df['target'] = df['close'].shift(-1)

        # NaN 제거
        df = df.dropna()

        return df

    def train(self, stock_code: str = None, activate_anyway: bool = False):
        """모델 학습

        Args:
            stock_code: 특정 종목 코드 (None이면 전체)
            activate_anyway: True면 성능 무관하게 새 모델 활성화

        Returns:
            tuple: (model, metadata)
        """
        start_time = time.time()

        print(f"\n{'='*50}")
        print(f"🚀 Starting model training...")
        print(f"   Save to DB: {self.save_to_db}")
        print(f"{'='*50}\n")

        # 1. 데이터 로드
        df = self.load_data(stock_code)
        print(f"📊 Loaded data: {len(df)} records")

        if len(df) == 0:
            raise ValueError("No data to train on. Please run preprocess_data.py first.")

        # 2. 피처/타겟 분리 (전처리된 데이터에는 이미 피처가 있음)
        feature_columns = ['open', 'high', 'low', 'close', 'volume',
                          'ma5', 'ma10', 'ma20', 'rsi',
                          'bb_upper', 'bb_middle', 'bb_lower',
                          'macd', 'macd_signal', 'macd_diff',
                          'price_change_1d', 'volume_change']

        # 데이터에 실제 존재하는 컬럼만 사용
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"📈 Using {len(available_features)} features: {', '.join(available_features[:5])}...")

        X = df[available_features]
        y = df['target_price']  # 전처리 스크립트에서 생성된 타겟 변수

        # 3. 데이터 분할
        print(f"✂️  Splitting data: 80% train, 20% test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # 시계열은 shuffle=False
        )
        print(f"   Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples\n")

        # 4. 스케일링
        print(f"⚙️  Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 5. 모델 학습
        print(f"🎯 Training GradientBoostingRegressor...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        print(f"✅ Training completed!\n")

        # 6. 평가
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        # 추가 메트릭 계산
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # MAPE (0인 값 제외)
        non_zero_mask = y_test != 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
        else:
            mape = None

        training_duration = time.time() - start_time

        print(f"📊 Model Performance:")
        print(f"   Train R² Score: {train_score:.4f}")
        print(f"   Test R² Score: {test_score:.4f}")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        if mape is not None:
            print(f"   MAPE: {mape:.2f}%")
        print(f"   Training time: {training_duration:.2f}s\n")

        # 7. 모델 저장 (파일)
        model_name = f'{stock_code}_model.pkl' if stock_code else 'stock_prediction_v1.pkl'
        model_path = self.model_dir / model_name
        scaler_path = self.model_dir / 'scaler.pkl'

        print(f"💾 Saving model to file...")
        joblib.dump(model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}\n")

        # 8. 메타데이터 준비
        metadata = {
            'model_name': model_name.replace('.pkl', ''),
            'model_type': 'GradientBoostingRegressor',
            'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'stock_code': stock_code,
            'trained_at': datetime.now().isoformat(),
            'train_score': float(train_score),
            'test_score': float(test_score),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_samples': len(df),
            'feature_columns': available_features,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape) if mape is not None else None,
            'training_duration_sec': training_duration
        }

        # 9. 메타데이터 파일 저장
        metadata_path = self.model_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📄 Metadata saved: {metadata_path}")

        # 10. DB 저장 (옵션)
        if self.save_to_db:
            # 자동 활성화 판단
            should_activate = activate_anyway or self._should_activate_new_model(test_score)
            self._save_to_database(model, metadata, activate=should_activate)

        print(f"\n{'='*50}")
        print(f"✅ Model training completed successfully!")
        print(f"   Duration: {training_duration:.2f}s")
        print(f"{'='*50}\n")

        return model, metadata

# 실행 예시
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stock Prediction Model Training')
    parser.add_argument('--stock-code', '-s', type=str, default=None,
                        help='Specific stock code to train (default: all stocks)')
    parser.add_argument('--no-db', action='store_true',
                        help='Skip saving to database')
    parser.add_argument('--activate-anyway', '-a', action='store_true',
                        help='Activate new model regardless of performance')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Data directory path')
    parser.add_argument('--model-dir', type=str, default='./models',
                        help='Model directory path')

    args = parser.parse_args()

    trainer = StockPredictionTrainer(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        save_to_db=not args.no_db
    )

    # 학습 실행
    model, metadata = trainer.train(
        stock_code=args.stock_code,
        activate_anyway=args.activate_anyway
    )

    # 결과 요약
    print(f"\n📋 Training Summary:")
    print(f"   Model: {metadata['model_name']}")
    print(f"   Test R² Score: {metadata['test_score']:.4f}")
    print(f"   Samples: {metadata['n_samples']}")
    print(f"   DB Saved: {trainer.save_to_db}")