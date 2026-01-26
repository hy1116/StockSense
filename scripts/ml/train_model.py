import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
from datetime import datetime
from pathlib import Path

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class StockPredictionTrainer:
    def __init__(self, data_dir='./data', model_dir='./models'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = MinMaxScaler()

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

    def train(self, stock_code: str = None):
        """모델 학습"""
        print(f"\n{'='*50}")
        print(f"🚀 Starting model training...")
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

        print(f"📊 Model Performance:")
        print(f"   Train R² Score: {train_score:.4f}")
        print(f"   Test R² Score: {test_score:.4f}\n")

        # 7. 모델 저장
        model_name = f'{stock_code}_model.pkl' if stock_code else 'stock_prediction_v1.pkl'
        model_path = self.model_dir / model_name
        scaler_path = self.model_dir / 'scaler.pkl'

        print(f"💾 Saving model...")
        joblib.dump(model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"   Model: {model_path}")
        print(f"   Scaler: {scaler_path}\n")

        # 8. 메타데이터 저장
        metadata = {
            'model_name': model_name,
            'stock_code': stock_code,
            'trained_at': datetime.now().isoformat(),
            'train_score': float(train_score),
            'test_score': float(test_score),
            'n_samples': len(df),
            'feature_columns': available_features
        }

        metadata_path = self.model_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📄 Metadata saved: {metadata_path}\n")

        print(f"{'='*50}")
        print(f"✅ Model training completed successfully!")
        print(f"{'='*50}\n")

        return model, metadata

# 실행 예시
if __name__ == '__main__':
    trainer = StockPredictionTrainer()

    # 전체 종목 학습
    model, metadata = trainer.train()

    # 특정 종목 학습 (선택)
    # model, metadata = trainer.train('005930')