"""데이터 전처리 스크립트 (ML 피처 생성)"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


class DataPreprocessor:
    """데이터 전처리 및 피처 엔지니어링"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "historical"
        self.processed_dir = self.data_dir / "processed"
        self.datasets_dir = self.data_dir / "datasets"

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def load_stock_data(self, stock_code: str) -> pd.DataFrame:
        """CSV 파일에서 주가 데이터 로드"""
        filename = self.raw_dir / f"{stock_code}_historical.csv"

        if not filename.exists():
            print(f"⚠️  File not found: {filename}")
            return None

        df = pd.read_csv(filename)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 생성"""
        df = df.copy()

        # 이동평균 (30일 데이터를 고려하여 최대 20일로 제한)
        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()

        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        df["bb_std"] = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["macd_signal"]

        # 거래량 이동평균
        df["volume_ma5"] = df["volume"].rolling(window=5).mean()
        df["volume_ma20"] = df["volume"].rolling(window=20).mean()

        # 가격 변화율
        df["price_change_1d"] = df["close"].pct_change(1)
        df["price_change_5d"] = df["close"].pct_change(5)
        df["price_change_20d"] = df["close"].pct_change(20)

        # 거래량 변화율
        df["volume_change"] = df["volume"].pct_change(1)

        # 변동성 (표준편차)
        df["volatility_5d"] = df["close"].rolling(window=5).std()
        df["volatility_20d"] = df["close"].rolling(window=20).std()

        # 고가/저가 비율
        df["high_low_ratio"] = df["high"] / df["low"]

        # 종가/시가 비율
        df["close_open_ratio"] = df["close"] / df["open"]

        return df

    def create_target(self, df: pd.DataFrame, days_ahead: int = 1) -> pd.DataFrame:
        """타겟 변수 생성 (미래 가격)"""
        df = df.copy()

        # 다음날 종가
        df["target_price"] = df["close"].shift(-days_ahead)

        # 다음날 수익률
        df["target_return"] = (df["target_price"] / df["close"]) - 1

        # 상승/하락 (분류용)
        df["target_direction"] = (df["target_return"] > 0).astype(int)

        return df

    def preprocess_stock(self, stock_code: str, save: bool = True) -> pd.DataFrame:
        """특정 종목 전처리"""
        print(f"\n📊 Preprocessing {stock_code}...")

        # 1. 데이터 로드
        df = self.load_stock_data(stock_code)
        if df is None:
            return None

        print(f"   Loaded {len(df)} records")

        # 2. 기술적 지표 생성
        df = self.create_technical_indicators(df)
        print(f"   Created technical indicators")

        # 3. 타겟 생성
        df = self.create_target(df, days_ahead=1)
        print(f"   Created target variables")

        # 4. NaN 제거
        df_clean = df.dropna()
        print(f"   Cleaned data: {len(df_clean)} records (removed {len(df) - len(df_clean)} NaN rows)")

        # 5. 저장
        if save:
            output_file = self.processed_dir / f"{stock_code}_features.csv"
            df_clean.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"   💾 Saved to {output_file.name}")

        return df_clean

    def combine_all_stocks(self) -> pd.DataFrame:
        """모든 종목 데이터 결합"""
        print("\n🔗 Combining all stock data...")

        all_files = list(self.processed_dir.glob("*_features.csv"))

        if not all_files:
            print("⚠️  No processed files found. Run preprocessing first.")
            return None

        dfs = []
        for file in all_files:
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"   Loaded {file.name}: {len(df)} records")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"\n✅ Combined {len(all_files)} files: Total {len(combined)} records")

        return combined

    def split_dataset(
        self, df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
    ):
        """학습/검증/테스트 데이터 분할"""
        print("\n✂️  Splitting dataset...")

        # 시계열 데이터는 시간 순서대로 분할 (shuffle X)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"   Train: {len(train_df)} records ({train_ratio*100:.0f}%)")
        print(f"   Validation: {len(val_df)} records ({val_ratio*100:.0f}%)")
        print(f"   Test: {len(test_df)} records ({(1-train_ratio-val_ratio)*100:.0f}%)")

        # 저장
        train_df.to_csv(self.datasets_dir / "train.csv", index=False, encoding="utf-8-sig")
        val_df.to_csv(self.datasets_dir / "validation.csv", index=False, encoding="utf-8-sig")
        test_df.to_csv(self.datasets_dir / "test.csv", index=False, encoding="utf-8-sig")

        print(f"   💾 Saved to {self.datasets_dir}")

        return train_df, val_df, test_df

    def process_all(self):
        """전체 파이프라인 실행"""
        print("\n" + "=" * 50)
        print("🚀 Starting data preprocessing pipeline")
        print("=" * 50)

        # 1. 모든 종목 전처리
        raw_files = list(self.raw_dir.glob("*_historical.csv"))

        if not raw_files:
            print("⚠️  No raw data files found.")
            print("   Please run: python scripts/collect_historical_data.py")
            return

        print(f"\nFound {len(raw_files)} stock files to process\n")

        for file in raw_files:
            stock_code = file.stem.replace("_historical", "")
            self.preprocess_stock(stock_code, save=True)

        # 2. 전체 데이터 결합
        combined_df = self.combine_all_stocks()

        if combined_df is None:
            return

        # 3. 학습/검증/테스트 분할
        self.split_dataset(combined_df)

        print("\n" + "=" * 50)
        print("✅ Data preprocessing completed!")
        print("=" * 50)
        print(f"\n📁 Processed files: {self.processed_dir}")
        print(f"📁 Dataset files: {self.datasets_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="데이터 전처리 및 피처 생성")
    parser.add_argument("--stock", type=str, help="특정 종목만 처리 (예: 005930)")

    args = parser.parse_args()

    preprocessor = DataPreprocessor()

    if args.stock:
        # 특정 종목만 처리
        preprocessor.preprocess_stock(args.stock, save=True)
    else:
        # 전체 파이프라인 실행
        preprocessor.process_all()


if __name__ == "__main__":
    main()
