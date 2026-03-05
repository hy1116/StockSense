"""재무 데이터 수집 스크립트

KRX 전체 종목의 PER/PBR/EPS/BPS/DIV를 FinanceDataReader로 일괄 수집하고,
Naver Finance 비공식 API로 ROE/매출/영업이익/순이익을 보완하여 DB에 UPSERT.

Usage:
    python ml/collect_financial_data.py
    python ml/collect_financial_data.py --codes 005930 000660
"""
import sys
import os
import time
import argparse
import logging
from pathlib import Path
from datetime import date
from typing import List, Optional

# Windows 콘솔 인코딩 설정
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.logger import get_logger
logger = get_logger("collect_financial_data")


def fetch_krx_financial_data() -> Optional[object]:
    """FinanceDataReader로 KRX 전체 종목 재무 데이터 수집"""
    try:
        import FinanceDataReader as fdr
        import pandas as pd
    except ImportError:
        logger.error("FinanceDataReader가 설치되지 않았습니다. pip install finance-datareader")
        return None

    # KRX 전체 → 실패 시 KOSPI + KOSDAQ 합산으로 폴백
    for market in ['KRX', None]:
        try:
            if market == 'KRX':
                logger.info("KRX 전체 종목 데이터 수집 중 (FinanceDataReader)...")
                df = fdr.StockListing('KRX')
            else:
                logger.info("KRX 실패, KOSPI + KOSDAQ 개별 수집으로 폴백...")
                kospi = fdr.StockListing('KOSPI')
                kosdaq = fdr.StockListing('KOSDAQ')
                df = pd.concat([kospi, kosdaq], ignore_index=True)

            logger.info(f"종목 수: {len(df)}")
            return df
        except Exception as e:
            logger.warning(f"{'KRX' if market else 'KOSPI+KOSDAQ'} 데이터 수집 실패: {e}")

    return None


def fetch_naver_financial(stock_code: str) -> dict:
    """Naver Finance 비공식 API로 ROE/매출/영업이익/순이익 수집"""
    try:
        import requests
        url = f"https://api.finance.naver.com/service/itemSummary.nhn?itemcode={stock_code}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': f'https://finance.naver.com/item/main.nhn?code={stock_code}'
        }
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        result = {}

        # PER, PBR, EPS (FDR 폴백용)
        for key, field in [('per', 'per'), ('pbr', 'pbr'), ('eps', 'eps')]:
            val = data.get(field)
            if val is not None:
                try:
                    result[key] = float(val)
                except (ValueError, TypeError):
                    pass

        # ROE
        roe_val = data.get('roe')
        if roe_val is not None:
            try:
                result['roe'] = float(roe_val)
            except (ValueError, TypeError):
                pass

        # 매출액 (억원 단위로 변환: 네이버 API는 백만원 단위)
        sales = data.get('sales')
        if sales is not None:
            try:
                result['revenue'] = float(sales) / 100  # 백만 → 억
            except (ValueError, TypeError):
                pass

        # 영업이익
        op = data.get('operatingProfit')
        if op is not None:
            try:
                result['operating_profit'] = float(op) / 100
            except (ValueError, TypeError):
                pass

        # 순이익
        net = data.get('netIncome')
        if net is not None:
            try:
                result['net_profit'] = float(net) / 100
            except (ValueError, TypeError):
                pass

        return result

    except Exception as e:
        logger.debug(f"[{stock_code}] Naver API 실패: {e}")
        return {}


def upsert_financial_data(engine, records: List[dict]) -> int:
    """재무 데이터를 DB에 UPSERT"""
    if not records:
        return 0

    from sqlalchemy import text

    upsert_sql = text("""
        INSERT INTO stock_financials
            (stock_code, stock_name, date, per, pbr, eps, bps, div_yield,
             roe, revenue, operating_profit, net_profit, source)
        VALUES
            (:stock_code, :stock_name, :date, :per, :pbr, :eps, :bps, :div_yield,
             :roe, :revenue, :operating_profit, :net_profit, :source)
        ON CONFLICT (stock_code, date)
        DO UPDATE SET
            stock_name      = EXCLUDED.stock_name,
            per             = COALESCE(EXCLUDED.per, stock_financials.per),
            pbr             = COALESCE(EXCLUDED.pbr, stock_financials.pbr),
            eps             = COALESCE(EXCLUDED.eps, stock_financials.eps),
            bps             = COALESCE(EXCLUDED.bps, stock_financials.bps),
            div_yield       = COALESCE(EXCLUDED.div_yield, stock_financials.div_yield),
            roe             = COALESCE(EXCLUDED.roe, stock_financials.roe),
            revenue         = COALESCE(EXCLUDED.revenue, stock_financials.revenue),
            operating_profit = COALESCE(EXCLUDED.operating_profit, stock_financials.operating_profit),
            net_profit      = COALESCE(EXCLUDED.net_profit, stock_financials.net_profit),
            source          = EXCLUDED.source,
            collected_at    = NOW()
    """)

    count = 0
    with engine.begin() as conn:
        for record in records:
            try:
                conn.execute(upsert_sql, record)
                count += 1
            except Exception as e:
                logger.warning(f"UPSERT 실패 [{record.get('stock_code')}]: {e}")

    return count


def safe_float(val) -> Optional[float]:
    """안전한 float 변환"""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN 체크
            return None
        return f
    except (ValueError, TypeError):
        return None


def _run_naver_only(engine, today, target_codes: Optional[List[str]] = None) -> bool:
    """FDR 실패 시 DB 종목 목록 기반으로 Naver Finance만으로 수집"""
    from sqlalchemy import text as sa_text

    try:
        with engine.connect() as conn:
            if target_codes:
                placeholders = ','.join(f"'{c}'" for c in target_codes)
                rows = conn.execute(sa_text(
                    f"SELECT stock_code, stock_name FROM stocks WHERE stock_code IN ({placeholders})"
                )).fetchall()
            else:
                rows = conn.execute(sa_text(
                    "SELECT stock_code, stock_name FROM stocks"
                )).fetchall()
    except Exception as e:
        logger.error(f"stocks 조회 실패: {e}")
        return True

    if not rows:
        logger.warning("stocks 테이블에 종목이 없어 재무 데이터 수집 불가")
        return True

    logger.info(f"Naver Finance 단독 수집 대상: {len(rows)}개 종목")
    records = []
    for stock_code, stock_name in rows:
        naver_data = fetch_naver_financial(stock_code)
        record = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'date': today,
            'per': naver_data.get('per'),
            'pbr': naver_data.get('pbr'),
            'eps': naver_data.get('eps'),
            'bps': None,
            'div_yield': None,
            'roe': naver_data.get('roe'),
            'revenue': naver_data.get('revenue'),
            'operating_profit': naver_data.get('operating_profit'),
            'net_profit': naver_data.get('net_profit'),
            'source': 'naver',
        }
        records.append(record)
        time.sleep(0.2)

        if len(records) >= 100:
            saved = upsert_financial_data(engine, records)
            logger.info(f"  저장: {saved}건")
            records = []

    if records:
        saved = upsert_financial_data(engine, records)
        logger.info(f"  마지막 배치 저장: {saved}건")

    logger.info("Naver Finance 단독 수집 완료")
    return True


def run(target_codes: Optional[List[str]] = None):
    """전체 재무 데이터 수집 실행"""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocksense")
    db_url = db_url.replace("+asyncpg", "")

    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        # 테이블 존재 확인
        with engine.connect() as conn:
            conn.execute(__import__('sqlalchemy').text("SELECT 1 FROM stock_financials LIMIT 1"))
    except Exception as e:
        logger.error(f"DB 연결 또는 테이블 오류: {e}")
        logger.info("먼저 alembic upgrade head 또는 백엔드를 실행하여 테이블을 생성하세요.")
        return False

    today = date.today()

    # KRX 전체 데이터 수집
    krx_df = fetch_krx_financial_data()

    if krx_df is None:
        logger.warning("FDR 데이터 수집 실패. DB 종목 기반 Naver Finance 단독 수집으로 폴백...")
        return _run_naver_only(engine, today, target_codes)

    # 컬럼명 정규화 (FDR 버전별로 컬럼명이 다를 수 있음)
    col_map = {}
    for col in krx_df.columns:
        col_lower = col.lower()
        if 'code' in col_lower and 'symbol' not in col_lower:
            col_map['code'] = col
        elif col_lower in ('symbol', 'code'):
            col_map['code'] = col
        elif 'name' in col_lower and 'market' not in col_lower:
            col_map['name'] = col
        elif col_lower == 'per':
            col_map['per'] = col
        elif col_lower == 'pbr':
            col_map['pbr'] = col
        elif col_lower == 'eps':
            col_map['eps'] = col
        elif col_lower == 'bps':
            col_map['bps'] = col
        elif col_lower in ('div', 'dividendyield', 'div_yield'):
            col_map['div_yield'] = col

    # Code 컬럼 fallback
    if 'code' not in col_map:
        for col in krx_df.columns:
            if krx_df[col].astype(str).str.match(r'^\d{6}$').any():
                col_map['code'] = col
                break

    logger.info(f"FDR 컬럼 매핑: {col_map}")

    if 'code' not in col_map:
        logger.error("종목코드 컬럼을 찾을 수 없습니다.")
        logger.info(f"사용 가능한 컬럼: {list(krx_df.columns)}")
        return True  # 파이프라인 중단 방지

    # 대상 종목 필터링
    if target_codes:
        code_col = col_map['code']
        krx_df = krx_df[krx_df[code_col].isin(target_codes)]
        logger.info(f"지정 종목 {len(krx_df)}개 처리")
    else:
        logger.info(f"전체 KRX {len(krx_df)}개 종목 처리")

    records = []
    naver_count = 0
    naver_fail_count = 0

    for idx, row in krx_df.iterrows():
        stock_code = str(row.get(col_map.get('code', ''), '')).strip()
        if not stock_code or len(stock_code) != 6:
            continue

        stock_name = str(row.get(col_map.get('name', ''), '')) if 'name' in col_map else None

        record = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'date': today,
            'per': safe_float(row.get(col_map.get('per', ''))),
            'pbr': safe_float(row.get(col_map.get('pbr', ''))),
            'eps': safe_float(row.get(col_map.get('eps', ''))),
            'bps': safe_float(row.get(col_map.get('bps', ''))),
            'div_yield': safe_float(row.get(col_map.get('div_yield', ''))),
            'roe': None,
            'revenue': None,
            'operating_profit': None,
            'net_profit': None,
            'source': 'fdr+naver',
        }

        # Naver API로 ROE/매출/영업이익 보완
        naver_data = fetch_naver_financial(stock_code)
        if naver_data:
            record.update(naver_data)
            naver_count += 1
        else:
            naver_fail_count += 1

        records.append(record)

        # Rate limiting
        time.sleep(0.2)

        # 배치 저장 (100개마다)
        if len(records) >= 100:
            saved = upsert_financial_data(engine, records)
            logger.info(f"  저장 완료: {saved}건 (누적 종목: {idx + 1})")
            records = []

    # 나머지 저장
    if records:
        saved = upsert_financial_data(engine, records)
        logger.info(f"  마지막 배치 저장: {saved}건")

    logger.info(f"재무 데이터 수집 완료. Naver 성공: {naver_count}, 실패: {naver_fail_count}")
    return True


def main():
    parser = argparse.ArgumentParser(description="재무 데이터 수집 (FDR + Naver Finance)")
    parser.add_argument(
        '--codes', nargs='+', type=str,
        help='특정 종목만 처리 (예: --codes 005930 000660)'
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("재무 데이터 수집 시작")
    logger.info("=" * 50)

    try:
        success = run(target_codes=args.codes)
    except Exception as e:
        logger.error(f"재무 데이터 수집 오류: {e}", exc_info=True)
        success = True  # 파이프라인 중단 방지

    logger.info("재무 데이터 수집 완료" if success else "재무 데이터 수집 실패 (파이프라인은 계속)")
    sys.exit(0)  # 항상 0 반환 (파이프라인 중단 방지)


if __name__ == "__main__":
    main()
