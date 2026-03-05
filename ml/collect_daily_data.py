"""일일 주가 데이터 수집 스크립트 (매일 실행용)"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# .env 파일 로드
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.logger import TeeStdout
from app.services.kis_api import KISAPIClient
from ml.stock_repository import StockRepository
import pandas as pd


class DailyDataCollector:
    """일일 데이터 수집기"""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.daily_dir = self.data_dir / "raw" / "daily"
        self.daily_dir.mkdir(parents=True, exist_ok=True)

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

    def collect_today_data(self, stock_code: str) -> dict:
        """오늘의 주가 데이터 수집"""
        try:
            result = self.client.get_stock_price(stock_code)

            if result.get("rt_cd") != "0":
                print(f"❌ Error for {stock_code}: {result.get('msg1')}")
                return None

            output = result.get("output", {})

            data = {
                "date": datetime.now().strftime("%Y%m%d"),
                "stock_code": stock_code,
                "stock_name": output.get("hts_kor_isnm", stock_code),
                "current_price": int(output.get("stck_prpr", 0)),
                "open": int(output.get("stck_oprc", 0)),
                "high": int(output.get("stck_hgpr", 0)),
                "low": int(output.get("stck_lwpr", 0)),
                "volume": int(output.get("acml_vol", 0)),
                "change_price": int(output.get("prdy_vrss", 0)),
                "change_rate": float(output.get("prdy_ctrt", 0)),
                "collected_at": datetime.now().isoformat(),
            }

            return data

        except Exception as e:
            print(f"❌ Failed to collect {stock_code}: {str(e)}")
            return None

    def save_daily_data(self, data: dict):
        """일일 데이터 저장"""
        if not data:
            return

        stock_code = data["stock_code"]
        date = data["date"]

        # 파일명: {종목코드}_{날짜}.json
        filename = self.daily_dir / f"{stock_code}_{date}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 Saved: {filename.name}")

    def append_to_historical(self, data: dict):
        """과거 데이터 파일에 추가"""
        if not data:
            return

        stock_code = data["stock_code"]
        historical_file = self.data_dir / "raw" / "historical" / f"{stock_code}_historical.csv"

        # 과거 데이터 파일이 있으면 추가
        if historical_file.exists():
            try:
                df_existing = pd.read_csv(historical_file)

                # 새 데이터 추가
                new_row = pd.DataFrame([{
                    "date": pd.to_datetime(data["date"], format="%Y%m%d"),
                    "stock_code": stock_code,
                    "stock_name": data["stock_name"],
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["current_price"],
                    "volume": data["volume"],
                    "change_price": data["change_price"],
                    "change_rate": data["change_rate"],
                }])

                # 날짜 중복 체크
                df_existing["date"] = pd.to_datetime(df_existing["date"])
                if new_row["date"].iloc[0] not in df_existing["date"].values:
                    df_updated = pd.concat([df_existing, new_row], ignore_index=True)
                    df_updated = df_updated.sort_values("date").reset_index(drop=True)
                    df_updated.to_csv(historical_file, index=False, encoding="utf-8-sig")
                    print(f"📝 Updated historical file: {historical_file.name}")
                else:
                    print(f"⏭️  Data already exists in historical file")

            except Exception as e:
                print(f"⚠️  Failed to update historical file: {str(e)}")

    def collect_all(self, update_historical: bool = True):
        """전체 종목 일일 데이터 수집"""
        print(f"\n🚀 Daily data collection started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Collecting {len(self.major_stocks)} stocks...\n")

        success_count = 0
        fail_count = 0

        for stock_code in self.major_stocks:
            data = self.collect_today_data(stock_code)

            if data:
                self.save_daily_data(data)

                if update_historical:
                    self.append_to_historical(data)

                success_count += 1
            else:
                fail_count += 1

            # API rate limiting
            import time
            time.sleep(0.3)

        print("\n" + "=" * 50)
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {fail_count}")
        print(f"📁 Data saved to: {self.daily_dir}")
        print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="일일 주가 데이터 수집")
    parser.add_argument(
        "--no-update-historical",
        action="store_true",
        help="과거 데이터 파일 업데이트 안 함",
    )

    args = parser.parse_args()

    with TeeStdout("collect_daily_data"):
        collector = DailyDataCollector()
        collector.collect_all(update_historical=not args.no_update_historical)


if __name__ == "__main__":
    main()
