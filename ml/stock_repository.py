"""종목 관리를 위한 데이터베이스 Repository"""
import os
from typing import List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()


class StockRepository:
    """종목 DB 관리 (stocks 테이블 통합)"""

    def __init__(self):
        # 환경변수에서 DB URL 읽기 (동기 드라이버 사용)
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:password@localhost:5432/stocksense"
        )

        # asyncpg -> psycopg2로 변환 (동기 사용)
        if "+asyncpg" in database_url:
            database_url = database_url.replace("+asyncpg", "")

        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)

    def get_active_stocks(self) -> List[dict]:
        """활성화된 수집 대상 종목 조회

        Returns:
            List[dict]: 종목 정보 리스트
        """
        with self.Session() as session:
            result = session.execute(
                text("""
                    SELECT stock_code, stock_name, market, priority, description
                    FROM stocks
                    WHERE is_active = true
                    ORDER BY priority DESC, stock_code
                """)
            )
            return [
                {
                    "stock_code": row.stock_code,
                    "stock_name": row.stock_name,
                    "market": row.market,
                    "priority": row.priority,
                    "description": row.description,
                }
                for row in result
            ]

    def get_stock_codes(self) -> List[str]:
        """활성화된 종목 코드만 조회

        Returns:
            List[str]: 종목 코드 리스트
        """
        stocks = self.get_active_stocks()
        return [s["stock_code"] for s in stocks]

    def add_stock(
        self,
        stock_code: str,
        stock_name: str,
        market: str = None,
        priority: int = 0,
        description: str = None,
        is_active: bool = False,
    ) -> bool:
        """종목 추가/업데이트

        Args:
            stock_code: 종목 코드
            stock_name: 종목명
            market: 시장 (KOSPI/KOSDAQ)
            priority: 우선순위
            description: 설명
            is_active: 수집 활성화 여부

        Returns:
            bool: 성공 여부
        """
        with self.Session() as session:
            try:
                session.execute(
                    text("""
                        INSERT INTO stocks
                        (stock_code, stock_name, market, priority, description, is_active)
                        VALUES (:code, :name, :market, :priority, :desc, :is_active)
                        ON CONFLICT (stock_code)
                        DO UPDATE SET
                            stock_name = :name,
                            market = :market,
                            priority = :priority,
                            description = :desc,
                            is_active = :is_active,
                            updated_at = NOW()
                    """),
                    {
                        "code": stock_code,
                        "name": stock_name,
                        "market": market,
                        "priority": priority,
                        "desc": description,
                        "is_active": is_active,
                    }
                )
                session.commit()
                return True
            except Exception as e:
                print(f"❌ Failed to add stock {stock_code}: {e}")
                session.rollback()
                return False

    def deactivate_stock(self, stock_code: str) -> bool:
        """종목 비활성화 (수집 중단)

        Args:
            stock_code: 종목 코드

        Returns:
            bool: 성공 여부
        """
        with self.Session() as session:
            try:
                session.execute(
                    text("""
                        UPDATE stocks
                        SET is_active = false, updated_at = NOW()
                        WHERE stock_code = :code
                    """),
                    {"code": stock_code}
                )
                session.commit()
                return True
            except Exception as e:
                print(f"❌ Failed to deactivate stock {stock_code}: {e}")
                session.rollback()
                return False

    def activate_stock(self, stock_code: str) -> bool:
        """종목 활성화 (수집 재개)

        Args:
            stock_code: 종목 코드

        Returns:
            bool: 성공 여부
        """
        with self.Session() as session:
            try:
                session.execute(
                    text("""
                        UPDATE stocks
                        SET is_active = true, updated_at = NOW()
                        WHERE stock_code = :code
                    """),
                    {"code": stock_code}
                )
                session.commit()
                return True
            except Exception as e:
                print(f"❌ Failed to activate stock {stock_code}: {e}")
                session.rollback()
                return False

    def sync_top_stocks(self, top_n: int = 30) -> dict:
        """시가총액 + 거래량 상위 종목을 KIS API에서 조회하여 DB에 추가

        Args:
            top_n: 각각 상위 몇 개 종목을 수집할지 (기본 30개)

        Returns:
            dict: {"market_cap": int, "volume": int, "total": int}
        """
        from app.services.kis_api import KISAPIClient

        app_key = os.getenv("KIS_APP_KEY")
        app_secret = os.getenv("KIS_APP_SECRET")
        account_number = os.getenv("KIS_ACCOUNT_NUMBER")

        if not app_key or not app_secret or not account_number:
            print("⚠️  KIS API credentials not found, using fallback stocks")
            count = self._init_fallback_stocks()
            return {"market_cap": count, "volume": 0, "total": count}

        try:
            client = KISAPIClient(
                app_key=app_key,
                app_secret=app_secret,
                account_number=account_number,
                account_product_code=os.getenv("KIS_ACCOUNT_PRODUCT_CODE", "01"),
                base_url=os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443"),
                use_mock=os.getenv("KIS_USE_MOCK", "True").lower() == "true",
            )

            added_codes = set()
            market_cap_count = 0
            volume_count = 0

            # 1. 시가총액 상위 종목
            print(f"\n📊 Fetching top {top_n} market cap stocks...")
            result = client.get_market_cap_ranking(top_n=top_n)

            if result.get("rt_cd") == "0":
                for i, item in enumerate(result.get("output", [])[:top_n]):
                    stock_code = item.get("mksc_shrn_iscd", "")
                    stock_name = item.get("hts_kor_isnm", "")
                    market_cap = item.get("stck_avls", "")

                    if not stock_code or not stock_name:
                        continue

                    market = "KOSDAQ" if stock_code.startswith("3") else "KOSPI"
                    priority = 100 - i

                    try:
                        market_cap_억 = int(market_cap) // 100000000
                        desc = f"시가총액 {i+1}위 ({market_cap_억:,}억)"
                    except:
                        desc = f"시가총액 {i+1}위"

                    if stock_code not in added_codes:
                        if self.add_stock(stock_code, stock_name, market, priority, desc, is_active=True):
                            added_codes.add(stock_code)
                            market_cap_count += 1
                            print(f"  ✅ [시총] {stock_code} - {stock_name} ({market})")

            # 2. 거래량 상위 종목
            print(f"\n📈 Fetching top {top_n} volume stocks...")
            result = client.get_volume_ranking()

            if result.get("rt_cd") == "0":
                for i, item in enumerate(result.get("output", [])[:top_n]):
                    stock_code = item.get("mksc_shrn_iscd", "")
                    stock_name = item.get("hts_kor_isnm", "")

                    if not stock_code or not stock_name:
                        continue

                    market = "KOSDAQ" if stock_code.startswith("3") else "KOSPI"

                    if stock_code not in added_codes:
                        priority = 50 - i  # 거래량 상위는 낮은 우선순위
                        desc = f"거래량 {i+1}위"
                        if self.add_stock(stock_code, stock_name, market, priority, desc, is_active=True):
                            added_codes.add(stock_code)
                            volume_count += 1
                            print(f"  ✅ [거래량] {stock_code} - {stock_name} ({market})")

            total = len(added_codes)
            print(f"\n📋 Summary: 시총 {market_cap_count}개 + 거래량 {volume_count}개 = 총 {total}개 종목")

            return {
                "market_cap": market_cap_count,
                "volume": volume_count,
                "total": total
            }

        except Exception as e:
            print(f"❌ Failed to sync stocks: {e}")
            print("⚠️  Using fallback stocks")
            count = self._init_fallback_stocks()
            return {"market_cap": count, "volume": 0, "total": count}

    def _init_fallback_stocks(self) -> int:
        """API 실패 시 사용할 기본 종목 (하드코딩)

        Returns:
            int: 추가된 종목 수
        """
        default_stocks = [
            ("005930", "삼성전자", "KOSPI", 100, "시가총액 1위"),
            ("000660", "SK하이닉스", "KOSPI", 99, "시가총액 상위"),
            ("035420", "NAVER", "KOSPI", 98, "시가총액 상위"),
            ("051910", "LG화학", "KOSPI", 97, "시가총액 상위"),
            ("005380", "현대차", "KOSPI", 96, "시가총액 상위"),
            ("006400", "삼성SDI", "KOSPI", 95, "시가총액 상위"),
            ("035720", "카카오", "KOSPI", 94, "시가총액 상위"),
            ("000270", "기아", "KOSPI", 93, "시가총액 상위"),
            ("207940", "삼성바이오로직스", "KOSPI", 92, "시가총액 상위"),
            ("068270", "셀트리온", "KOSPI", 91, "시가총액 상위"),
            ("028260", "삼성물산", "KOSPI", 90, "시가총액 상위"),
            ("105560", "KB금융", "KOSPI", 89, "시가총액 상위"),
            ("055550", "신한지주", "KOSPI", 88, "시가총액 상위"),
            ("012330", "현대모비스", "KOSPI", 87, "시가총액 상위"),
            ("066570", "LG전자", "KOSPI", 86, "시가총액 상위"),
        ]

        count = 0
        for code, name, market, priority, desc in default_stocks:
            if self.add_stock(code, name, market, priority, desc, is_active=True):
                count += 1
                print(f"✅ Added (fallback): {code} - {name}")

        return count
