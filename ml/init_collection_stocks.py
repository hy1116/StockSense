"""종목 데이터 초기화/동기화 스크립트"""
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.logger import TeeStdout
from ml.stock_repository import StockRepository


def main():
    parser = argparse.ArgumentParser(description="종목 데이터 관리 (stocks 테이블)")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="시총/거래량 상위 종목을 stocks 테이블에 동기화",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="상위 몇 개 종목을 수집할지 (기본: 30)",
    )
    parser.add_argument(
        "--add",
        type=str,
        nargs=2,
        metavar=("CODE", "NAME"),
        help="종목 추가 (예: --add 005930 삼성전자)",
    )
    parser.add_argument(
        "--remove",
        type=str,
        metavar="CODE",
        help="종목 비활성화 (예: --remove 005930)",
    )
    parser.add_argument(
        "--activate",
        type=str,
        metavar="CODE",
        help="종목 활성화 (예: --activate 005930)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="활성화된 종목 목록 조회",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="KOSPI",
        help="시장 (KOSPI/KOSDAQ, 기본: KOSPI)",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=50,
        help="우선순위 (기본: 50)",
    )

    args = parser.parse_args()

    with TeeStdout("init_collection_stocks"):
        repo = StockRepository()

        if args.sync:
            print(f"\n🔄 Syncing top {args.top} market cap + volume stocks...")
            result = repo.sync_top_stocks(top_n=args.top)
            print(f"\n✅ Synced {result['total']} stocks to stocks table")

        elif args.add:
            code, name = args.add
            print(f"\n📌 Adding stock: {code} - {name}")
            if repo.add_stock(code, name, args.market, args.priority):
                print(f"✅ Successfully added: {code}")
            else:
                print(f"❌ Failed to add: {code}")

        elif args.remove:
            code = args.remove
            print(f"\n🔒 Deactivating stock: {code}")
            if repo.deactivate_stock(code):
                print(f"✅ Successfully deactivated: {code}")
            else:
                print(f"❌ Failed to deactivate: {code}")

        elif args.activate:
            code = args.activate
            print(f"\n🔓 Activating stock: {code}")
            if repo.activate_stock(code):
                print(f"✅ Successfully activated: {code}")
            else:
                print(f"❌ Failed to activate: {code}")

        elif args.list:
            print("\n📋 Active stocks:")
            print("-" * 60)
            stocks = repo.get_active_stocks()
            if stocks:
                for i, s in enumerate(stocks, 1):
                    print(f"{i:3}. {s['stock_code']} - {s['stock_name']:15} "
                          f"[{s['market'] or 'N/A':6}] P:{s['priority']:3}")
                print("-" * 60)
                print(f"Total: {len(stocks)} stocks")
            else:
                print("No active stocks found.")
                print("\nRun with --sync to sync stocks from KIS API:")
                print("  python -m ml.init_collection_stocks --sync")

        else:
            parser.print_help()
            print("\n\nExamples:")
            print("  # Sync top 30 market cap + volume stocks")
            print("  python -m ml.init_collection_stocks --sync")
            print()
            print("  # Sync top 50 stocks")
            print("  python -m ml.init_collection_stocks --sync --top 50")
            print()
            print("  # List active stocks")
            print("  python -m ml.init_collection_stocks --list")
            print()
            print("  # Add a new stock manually")
            print("  python -m ml.init_collection_stocks --add 003550 LG --market KOSPI --priority 80")
            print()
            print("  # Deactivate a stock")
            print("  python -m ml.init_collection_stocks --remove 005930")


if __name__ == "__main__":
    main()
