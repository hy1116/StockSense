"""과거 주가 데이터 수집 스크립트"""
import sys
import os
import json
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# .env 파일 로드
load_dotenv()

from app.services.kis_api import KISAPIClient
from ml.stock_repository import StockRepository
import pandas as pd


class HistoricalDataCollector:
    """과거 주가 데이터 수집기"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw" / "historical"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # 환경 변수에서 KIS API 설정 읽기
        app_key = os.getenv("KIS_APP_KEY")
        app_secret = os.getenv("KIS_APP_SECRET")
        account_number = os.getenv("KIS_ACCOUNT_NUMBER")
        account_product_code = os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "01")
        base_url = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
        use_mock = os.getenv("KIS_USE_MOCK", "True").lower() == "true"
        cust_type = os.getenv("KIS_CUST_TYPE", "P")

        if not app_key or not app_secret or not account_number:
            raise ValueError(
                "KIS API credentials not found in environment variables.\n"
                "Please check .env file contains:\n"
                "  KIS_APP_KEY\n"
                "  KIS_APP_SECRET\n"
                "  KIS_ACCOUNT_NUMBER"
            )

        # KIS API 클라이언트
        self.client = KISAPIClient(
            app_key=app_key,
            app_secret=app_secret,
            account_number=account_number,
            account_product_code=account_product_code,
            base_url=base_url,
            use_mock=use_mock,
            cust_type=cust_type,
        )

        # DB에서 수집 대상 종목 조회
        self._load_stocks_from_db()

    def _load_stocks_from_db(self):
        """DB에서 수집 대상 종목 로드"""
        try:
            repo = StockRepository()
            self.major_stocks = repo.get_stock_codes()
            print(f"📋 Loaded {len(self.major_stocks)} stocks from database")
        except Exception as e:
            print(f"⚠️  Failed to load stocks from DB: {e}")
            print("⚠️  Using fallback stock list")
            # DB 연결 실패 시 기본 종목 사용
            self.major_stocks = [
                "005930",  # 삼성전자
                "000660",  # SK하이닉스
                "035420",  # NAVER
                "051910",  # LG화학
                "005380",  # 현대차
            ]

    def collect_stock_data(self, stock_code: str, days: int = 365) -> pd.DataFrame:
        """특정 종목의 과거 데이터 수집

        Args:
            stock_code: 종목 코드
            days: 수집할 일수 (기본 365일)

        Returns:
            DataFrame with OHLCV data
        """
        print(f"📊 Collecting data for {stock_code}...")

        try:
            # KIS API로 일봉 데이터 조회 (최대 100개씩)
            result = self.client.get_daily_chart(stock_code, period="D", count=100)

            if result.get("rt_cd") != "0":
                print(f"❌ Error: {result.get('msg1', 'Unknown error')}")
                return None

            output1 = result.get("output1", {})
            output2 = result.get("output2", [])

            if not output2:
                print(f"⚠️  No data returned for {stock_code}")
                return None

            # 종목명
            stock_name = output1.get("hts_kor_isnm", stock_code)

            # 데이터프레임 변환
            data = []
            for item in output2:
                data.append({
                    "date": item.get("stck_bsop_date", ""),
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "open": int(item.get("stck_oprc", 0)),
                    "high": int(item.get("stck_hgpr", 0)),
                    "low": int(item.get("stck_lwpr", 0)),
                    "close": int(item.get("stck_clpr", 0)),
                    "volume": int(item.get("acml_vol", 0)),
                    "change_price": int(item.get("prdy_vrss", 0)),
                    "change_rate": float(item.get("prdy_ctrt", 0)),
                })

            df = pd.DataFrame(data)

            # 종목코드 형식 맞추기
            df["stock_code"] = df["stock_code"].apply(lambda x: f'="{x}"')

            # 날짜를 datetime으로 변환
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

            # 날짜순 정렬 (오래된 것부터)
            df = df.sort_values("date").reset_index(drop=True)

            print(f"✅ Collected {len(df)} days of data for {stock_name} ({stock_code})")

            return df

        except Exception as e:
            print(f"❌ Failed to collect data for {stock_code}: {str(e)}")
            return None

    def save_to_csv(self, df: pd.DataFrame, stock_code: str):
        """CSV 파일로 저장"""
        if df is None or df.empty:
            return

        filename = self.raw_dir / f"{stock_code}_historical.csv"
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"💾 Saved to {filename}")

    def save_to_json(self, df: pd.DataFrame, stock_code: str):
        """JSON 파일로 저장"""
        if df is None or df.empty:
            return

        filename = self.raw_dir / f"{stock_code}_historical.json"

        # DataFrame을 dict로 변환
        data = {
            "stock_code": stock_code,
            "stock_name": df.iloc[0]["stock_name"] if len(df) > 0 else stock_code,
            "collected_at": datetime.now().isoformat(),
            "data_count": len(df),
            "records": df.to_dict("records"),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved to {filename}")

    def collect_all_major_stocks(self, days: int = 365, format: str = "csv"):
        """주요 종목 전체 수집

        Args:
            days: 수집할 일수
            format: 저장 형식 ('csv', 'json', 'both')
        """
        print(f"\n🚀 Starting data collection for {len(self.major_stocks)} stocks...")
        print(f"📅 Period: {days} days")
        print(f"💾 Format: {format}\n")

        success_count = 0
        fail_count = 0

        for i, stock_code in enumerate(self.major_stocks, 1):
            print(f"\n[{i}/{len(self.major_stocks)}] Processing {stock_code}...")

            df = self.collect_stock_data(stock_code, days)

            if df is not None and not df.empty:
                if format in ["csv", "both"]:
                    self.save_to_csv(df, stock_code)

                if format in ["json", "both"]:
                    self.save_to_json(df, stock_code)

                success_count += 1
            else:
                fail_count += 1

            # API 요청 간격 (Rate limiting 방지)
            import time
            time.sleep(0.5)

        print("\n" + "=" * 50)
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {fail_count}")
        print(f"📁 Data saved to: {self.raw_dir}")
        print("=" * 50)

    def collect_single_stock(self, stock_code: str, days: int = 365, format: str = "csv"):
        """단일 종목 수집"""
        print(f"\n🚀 Collecting data for {stock_code}...")

        df = self.collect_stock_data(stock_code, days)

        if df is not None and not df.empty:
            if format in ["csv", "both"]:
                self.save_to_csv(df, stock_code)

            if format in ["json", "both"]:
                self.save_to_json(df, stock_code)

            print(f"\n✅ Successfully collected data for {stock_code}")
        else:
            print(f"\n❌ Failed to collect data for {stock_code}")


def main():
    parser = argparse.ArgumentParser(description="주가 데이터 수집 스크립트")
    parser.add_argument(
        "--stock",
        type=str,
        help="종목 코드 (예: 005930). 미입력 시 주요 종목 전체 수집",
    )
    parser.add_argument(
        "--days", type=int, default=365, help="수집할 일수 (기본: 365일)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "both"],
        default="csv",
        help="저장 형식 (기본: csv)",
    )

    args = parser.parse_args()

    from ml.logger import TeeStdout
    with TeeStdout("collect_historical_data"):
        collector = HistoricalDataCollector()

        if args.stock:
            collector.collect_single_stock(args.stock, args.days, args.format)
        else:
            collector.collect_all_major_stocks(args.days, args.format)


if __name__ == "__main__":
    main()
