"""재무 데이터 수집 스크립트

KRX 전체 종목의 PER/PBR/EPS/BPS/DIV를 FinanceDataReader로 일괄 수집하고,
DART OpenAPI로 ROE/매출/영업이익/순이익을 보완하여 DB에 UPSERT.

Usage:
    python ml/collect_financial_data.py
    python ml/collect_financial_data.py --codes 005930 000660
"""
import sys
import os
import io
import time
import pickle
import zipfile
import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import date, datetime, timedelta
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


DART_CORP_CODE_CACHE = "/tmp/dart_corp_codes.pkl"

REVENUE_ACCOUNT_NAMES = ['매출액', '수익(매출액)', '영업수익', '매출']
OP_PROFIT_ACCOUNT_NAMES = ['영업이익', '영업이익(손실)']
NET_PROFIT_ACCOUNT_NAMES = ['당기순이익', '당기순이익(손실)', '분기순이익']


def fetch_dart_corp_code_map(api_key: str) -> dict:
    """DART corp_code ↔ stock_code 매핑 다운로드 (일 1회 캐시)"""
    import requests

    cache_file = Path(DART_CORP_CODE_CACHE)
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=1):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    logger.info("DART corp_code 매핑 다운로드 중...")
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={api_key}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xml_data = zf.read('CORPCODE.xml')

    root = ET.fromstring(xml_data)
    mapping = {}
    for item in root.findall('list'):
        stock_code = item.findtext('stock_code', '').strip()
        corp_code = item.findtext('corp_code', '').strip()
        if stock_code and len(stock_code) == 6:
            mapping[stock_code] = corp_code

    with open(cache_file, 'wb') as f:
        pickle.dump(mapping, f)

    logger.info(f"DART corp_code 매핑 완료: {len(mapping)}개 종목")
    return mapping


def _dart_get_amount(items: list, account_names: list) -> Optional[float]:
    """재무제표 항목 리스트에서 계정과목명으로 당기 금액(억원) 추출"""
    for item in items:
        if item.get('account_nm') in account_names:
            val = item.get('thstrm_amount', '').replace(',', '').strip()
            if val and val not in ('-', ''):
                try:
                    return float(val) / 1e8  # 원 → 억원
                except ValueError:
                    pass
    return None


def fetch_dart_financials(api_key: str, corp_code: str) -> dict:
    """DART API로 최근 연간 재무제표(매출/영업이익/순이익) + 재무지표(ROE) 수집"""
    import requests

    result = {}
    current_year = date.today().year

    # 재무제표: 연결(CFS) 우선, 없으면 별도(OFS). 작년 → 재작년 순으로 시도
    for year in [current_year - 1, current_year - 2]:
        if result.get('revenue') is not None:
            break
        for fs_div in ['CFS', 'OFS']:
            try:
                resp = requests.get(
                    "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
                    params={
                        'crtfc_key': api_key,
                        'corp_code': corp_code,
                        'bsns_year': str(year),
                        'reprt_code': '11011',  # 사업보고서
                        'fs_div': fs_div,
                    },
                    timeout=10,
                )
                data = resp.json()
                if data.get('status') != '000' or not data.get('list'):
                    continue

                items = data['list']
                revenue = _dart_get_amount(items, REVENUE_ACCOUNT_NAMES)
                op_profit = _dart_get_amount(items, OP_PROFIT_ACCOUNT_NAMES)
                net_profit = _dart_get_amount(items, NET_PROFIT_ACCOUNT_NAMES)

                if any(v is not None for v in [revenue, op_profit, net_profit]):
                    if revenue is not None:
                        result['revenue'] = revenue
                    if op_profit is not None:
                        result['operating_profit'] = op_profit
                    if net_profit is not None:
                        result['net_profit'] = net_profit
                    break
            except Exception as e:
                logger.debug(f"DART 재무제표 조회 실패 ({corp_code}, {year}, {fs_div}): {e}")

    # 재무지표(ROE): 수익성 지표 그룹(M210000)
    for year in [current_year - 1, current_year - 2]:
        try:
            resp = requests.get(
                "https://opendart.fss.or.kr/api/fnlttCmpnyIndx.json",
                params={
                    'crtfc_key': api_key,
                    'corp_code': corp_code,
                    'bsns_year': str(year),
                    'reprt_code': '11011',
                    'idx_cl_code': 'M210000',  # 수익성 지표
                },
                timeout=10,
            )
            data = resp.json()
            if data.get('status') == '000' and data.get('list'):
                for item in data['list']:
                    if item.get('idx_nm') == 'ROE':
                        val = item.get('idx_val', '').replace(',', '').strip()
                        if val and val not in ('-', ''):
                            result['roe'] = float(val)
                            break
            if 'roe' in result:
                break
        except Exception as e:
            logger.debug(f"DART ROE 조회 실패 ({corp_code}, {year}): {e}")

    return result


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


def _run_dart_only(engine, today, api_key: str, corp_code_map: dict,
                   target_codes: Optional[List[str]] = None) -> bool:
    """FDR 실패 시 DB 종목 목록 기반으로 DART만으로 수집"""
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

    logger.info(f"DART 단독 수집 대상: {len(rows)}개 종목")
    records = []
    for stock_code, stock_name in rows:
        corp_code = corp_code_map.get(stock_code)
        dart_data = fetch_dart_financials(api_key, corp_code) if corp_code else {}
        record = {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'date': today,
            'per': None,
            'pbr': None,
            'eps': None,
            'bps': None,
            'div_yield': None,
            'roe': dart_data.get('roe'),
            'revenue': dart_data.get('revenue'),
            'operating_profit': dart_data.get('operating_profit'),
            'net_profit': dart_data.get('net_profit'),
            'source': 'dart',
        }
        records.append(record)
        time.sleep(0.3)  # DART API rate limiting

        if len(records) >= 100:
            saved = upsert_financial_data(engine, records)
            logger.info(f"  저장: {saved}건")
            records = []

    if records:
        saved = upsert_financial_data(engine, records)
        logger.info(f"  마지막 배치 저장: {saved}건")

    logger.info("DART 단독 수집 완료")
    return True


def run(target_codes: Optional[List[str]] = None):
    """전체 재무 데이터 수집 실행"""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv()
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stocksense")
    db_url = db_url.replace("+asyncpg", "")
    dart_api_key = os.getenv("DART_API_KEY", "")

    if not dart_api_key:
        logger.warning("DART_API_KEY 미설정. ROE/매출/영업이익/순이익 수집 불가. .env에 DART_API_KEY를 추가하세요.")

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

    # DART corp_code 매핑 (API키가 있을 때만)
    corp_code_map = {}
    if dart_api_key:
        try:
            corp_code_map = fetch_dart_corp_code_map(dart_api_key)
        except Exception as e:
            logger.warning(f"DART corp_code 매핑 실패: {e}. ROE/매출 등은 수집되지 않습니다.")

    # KRX 전체 데이터 수집
    krx_df = fetch_krx_financial_data()

    if krx_df is None:
        logger.warning("FDR 데이터 수집 실패. DB 종목 기반 DART 단독 수집으로 폴백...")
        return _run_dart_only(engine, today, dart_api_key, corp_code_map, target_codes)

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
        elif col_lower in ('div', 'dividendyield', 'div_yield', 'dividend'):
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
    dart_success = 0
    dart_fail = 0

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
            'source': 'fdr+dart',
        }

        # DART API로 ROE/매출/영업이익/순이익 보완
        if dart_api_key and corp_code_map:
            corp_code = corp_code_map.get(stock_code)
            if corp_code:
                dart_data = fetch_dart_financials(dart_api_key, corp_code)
                if dart_data:
                    record.update(dart_data)
                    dart_success += 1
                else:
                    dart_fail += 1
            time.sleep(0.3)  # DART API rate limiting
        else:
            time.sleep(0.05)

        records.append(record)

        # 배치 저장 (100개마다)
        if len(records) >= 100:
            saved = upsert_financial_data(engine, records)
            logger.info(f"  저장 완료: {saved}건 (누적 종목: {idx + 1}, DART 성공: {dart_success})")
            records = []

    # 나머지 저장
    if records:
        saved = upsert_financial_data(engine, records)
        logger.info(f"  마지막 배치 저장: {saved}건")

    logger.info(f"재무 데이터 수집 완료. DART 성공: {dart_success}, 실패: {dart_fail}")
    return True


def main():
    parser = argparse.ArgumentParser(description="재무 데이터 수집 (FDR + DART OpenAPI)")
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
