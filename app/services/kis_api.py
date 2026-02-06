"""
한국투자증권 Open API 클라이언트
KIS (Korea Investment & Securities) API wrapper
"""
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from functools import lru_cache

from app.services.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class KISAPIClient:
    """한국투자증권 OpenAPI 클라이언트"""

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        account_number: str,
        account_product_code: str = "01",
        base_url: str = "https://openapi.koreainvestment.com:9443",
        use_mock: bool = True,
        cust_type: str = "P",
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_number = account_number
        self.account_product_code = account_product_code
        self.base_url = base_url
        self.use_mock = use_mock
        self.cust_type = cust_type

        # Redis 클라이언트
        self.redis = get_redis_client()
        self.token_key = f"kis_token:{self.app_key}"

        # 메모리 캐시 (fallback)
        self._access_token = None
        self._token_expiry = None

    def _log_request(self, method: str, url: str, headers: Dict, params: Dict = None, data: Dict = None):
        """KIS API 요청 로깅"""
        logger.info(f"→ KIS API {method} {url}")
        logger.info(f"  Headers: {headers}")
        if params:
            logger.info(f"  Params: {params}")
        if data:
            logger.info(f"  Body: {data}")

    def _log_response(self, method: str, url: str, status_code: int, response_data: Dict):
        """KIS API 응답 로깅"""
        logger.info(f"← KIS API {method} {url} | Status: {status_code}")
        logger.info(f"  Response: {response_data}")

    def _get_headers(self, tr_id: str, token_required: bool = True) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": self.cust_type
        }

        if token_required:
            token = self.get_access_token()
            headers["authorization"] = f"Bearer {token}"

        return headers

    def get_access_token(self) -> str:
        """접근 토큰 발급 (Redis 캐시 사용)"""
        logger.info(self.redis.is_available())
        # 1. Redis에서 토큰 조회
        if self.redis.is_available():
            cached_token = self.redis.get(self.token_key)
            if cached_token:
                logger.info("Using cached token from Redis")
                self._access_token = cached_token
                return cached_token

        # 2. 메모리 캐시 확인 (Redis 미사용 시)
        if self._access_token and self._token_expiry:
            if datetime.now() < self._token_expiry:
                logger.info("Using cached token from memory")
                return self._access_token

        # 3. 새로운 토큰 발급
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"Content-Type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        try:
            self._log_request("POST", url, headers, data=data)
            response = requests.post(url, headers=headers, json=data, timeout=10)

            result = response.json() if response.status_code == 200 else {"error": response.text}
            self._log_response("POST", url, response.status_code, result)

            if response.status_code == 200:
                if "access_token" in result:
                    access_token = result["access_token"]
                    expires_in = int(result.get("expires_in", 86400))

                    # Redis에 토큰 저장 (만료 시간 5분 전까지)
                    if self.redis.is_available():
                        cache_expire = expires_in - 300  # 5분 여유
                        self.redis.set(self.token_key, access_token, expire=cache_expire)
                        logger.info(f"Token cached in Redis (expires in {cache_expire}s)")

                    # 메모리에도 저장 (fallback)
                    self._access_token = access_token
                    self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

                    return access_token
                else:
                    raise Exception(f"토큰 발급 실패 - access_token 없음: {result}")
            else:
                raise Exception(f"토큰 발급 실패 (HTTP {response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"토큰 발급 네트워크 오류: {str(e)}")

    def get_balance(self) -> Dict:
        """계좌 잔고 조회"""
        tr_id = "TTTC8434R" if self.use_mock else "VTTC8434R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

        headers = self._get_headers(tr_id, True)
        params = {
            "CANO": self.account_number,
            "ACNT_PRDT_CD": self.account_product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }

        self._log_request("GET", url, headers, params=params)
        response = requests.get(url, headers=headers, params=params)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("GET", url, response.status_code, result)

        if response.status_code == 200:
            return result
        else:
            raise Exception(f"잔고 조회 실패: {response.text}")

    def get_stock_price(self, stock_code: str) -> Dict:
        """주식 현재가 조회"""
        tr_id = "FHKST01010100"
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"

        headers = self._get_headers(tr_id)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code
        }

        self._log_request("GET", url, headers, params=params)
        response = requests.get(url, headers=headers, params=params)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("GET", url, response.status_code, result)

        if response.status_code == 200:
            return result
        else:
            raise Exception(f"현재가 조회 실패: {response.text}")

    def buy_stock(
        self,
        stock_code: str,
        quantity: int,
        price: int,
        order_type: str = "00"
    ) -> Dict:
        """주식 매수

        Args:
            stock_code: 종목코드 (6자리)
            quantity: 주문수량
            price: 주문가격 (지정가) - 시장가는 0
            order_type: 주문유형 (00: 지정가, 01: 시장가)
        """
        tr_id = "TTTC0802U" if self.use_mock else "VTTC0802U"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        headers = self._get_headers(tr_id)
        data = {
            "CANO": self.account_number,
            "ACNT_PRDT_CD": self.account_product_code,
            "PDNO": stock_code,
            "ORD_DVSN": order_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price)
        }

        self._log_request("POST", url, headers, data=data)
        response = requests.post(url, headers=headers, json=data)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("POST", url, response.status_code, result)

        if response.status_code == 200:
            return result
        else:
            raise Exception(f"매수 주문 실패: {response.text}")

    def sell_stock(
        self,
        stock_code: str,
        quantity: int,
        price: int,
        order_type: str = "00"
    ) -> Dict:
        """주식 매도

        Args:
            stock_code: 종목코드 (6자리)
            quantity: 주문수량
            price: 주문가격 (지정가) - 시장가는 0
            order_type: 주문유형 (00: 지정가, 01: 시장가)
        """
        tr_id = "VTTC0801U" if self.use_mock else "TTTC0801U"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        headers = self._get_headers(tr_id)
        data = {
            "CANO": self.account_number,
            "ACNT_PRDT_CD": self.account_product_code,
            "PDNO": stock_code,
            "ORD_DVSN": order_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price)
        }

        self._log_request("POST", url, headers, data=data)
        response = requests.post(url, headers=headers, json=data)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("POST", url, response.status_code, result)

        if response.status_code == 200:
            return result
        else:
            raise Exception(f"매도 주문 실패: {response.text}")

    def get_daily_chart(self, stock_code: str, period: str = "D", count: int = 100) -> Dict:
        """주식 차트 데이터 조회

        Args:
            stock_code: 종목코드 (6자리)
            period: 기간 구분 (D: 일, W: 주, M: 월)
            count: 조회 개수 (최대 100)
        """
        tr_id = "FHKST03010100"
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"

        headers = self._get_headers(tr_id)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code,
            "FID_INPUT_DATE_1": "",
            "FID_INPUT_DATE_2": "",
            "FID_PERIOD_DIV_CODE": period,
            "FID_ORG_ADJ_PRC": "0"
        }

        self._log_request("GET", url, headers, params=params)
        response = requests.get(url, headers=headers, params=params)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("GET", url, response.status_code, result)

        if response.status_code == 200:
            return result
        else:
            raise Exception(f"차트 데이터 조회 실패: {response.text}")

    def get_volume_ranking(self) -> Dict:
        """거래량 순위 조회

        Returns:
            거래량 상위 종목 목록 (최대 30개)
        """
        tr_id = "FHPST01710000"
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/volume-rank"

        headers = self._get_headers(tr_id)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",  # J: 주식, ETF, ETN 전체
            "FID_COND_SCR_DIV_CODE": "20171",  # 거래량 순위
            "FID_INPUT_ISCD": "0000",  # 전체
            "FID_DIV_CLS_CODE": "0",  # 0: 전체
            "FID_BLNG_CLS_CODE": "0",  # 소속 구분
            "FID_TRGT_CLS_CODE": "111111111",  # 대상 구분 (전체)
            "FID_TRGT_EXLS_CLS_CODE": "000000",  # 제외 구분
            "FID_INPUT_PRICE_1": "",  # 가격 범위 시작
            "FID_INPUT_PRICE_2": "",  # 가격 범위 종료
            "FID_VOL_CNT": "",  # 거래량 수
            "FID_INPUT_DATE_1": ""  # 조회 시작일
        }

        self._log_request("GET", url, headers, params=params)
        response = requests.get(url, headers=headers, params=params)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("GET", url, response.status_code, result)

        if response.status_code == 200:
            return result
        else:
            raise Exception(f"거래량 순위 조회 실패: {response.text}")

    def get_market_cap_ranking(self, top_n: int = 30) -> Dict:
        """시가총액 순위 조회

        Args:
            top_n: 조회할 상위 종목 수 (기본 30개)

        Returns:
            시가총액 상위 종목 목록
        """
        tr_id = "FHPST01740000"
        url = f"{self.base_url}/uapi/domestic-stock/v1/ranking/market-cap"

        headers = self._get_headers(tr_id)
        params = {
            "fid_cond_mrkt_div_code": "J",  # J: 주식
            "fid_cond_scr_div_code": "20174",
            "fid_input_iscd": "0000",  # 전체
            "fid_div_cls_code": "0",  # 0: 전체
            "fid_blng_cls_code": "0",
            "fid_trgt_cls_code": "0",
            "fid_trgt_exls_cls_code": "0",
            "fid_input_price_1": "",
            "fid_input_price_2": "",
            "fid_vol_cnt": "",
        }

        self._log_request("GET", url, headers, params=params)
        response = requests.get(url, headers=headers, params=params)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("GET", url, response.status_code, result)

        if response.status_code == 200:
            # 상위 N개만 반환
            if "output" in result:
                result["output"] = result["output"][:top_n]
            return result
        else:
            raise Exception(f"시가총액 순위 조회 실패: {response.text}")
    
    def get_fluctuation_ranking(self, top_n: int = 30) -> Dict:
        """등락률 순위 조회

        Args:
            top_n: 조회할 상위 종목 수 (기본 30개)

        Returns:
            등락률 상위 종목 목록
        """
        tr_id = "FHPST01700000"
        url = f"{self.base_url}/uapi/domestic-stock/v1/ranking/fluctuation"

        headers = self._get_headers(tr_id)
        params = {
            "fid_rsfl_rate2": "",  # 등락 비율2
            "fid_cond_mrkt_div_code": "J",  # 조건 시장 분류 코드
            "fid_cond_scr_div_code": "20170",  # 조건 화면 분류 코드
            "fid_input_iscd": "0000",  # 입력 종목코드
            "fid_rank_sort_cls_code": "0",  # 순위 정렬 구분 코드 (0: 상승율순)
            "fid_input_cnt_1": "0",  # 입력 수1
            "fid_prc_cls_code": "1",  # 가격 구분 코드
            "fid_input_price_1": "",  # 입력 가격1
            "fid_input_price_2": "",  # 입력 가격2
            "fid_vol_cnt": "",  # 거래량 수
            "fid_trgt_cls_code": "0",  # 대상 구분 코드
            "fid_trgt_exls_cls_code": "0",  # 대상 제외 구분 코드
            "fid_div_cls_code": "0",  # 분류 구분 코드
            "fid_rsfl_rate1": ""  # 등락 비율1
        }

        self._log_request("GET", url, headers, params=params)
        response = requests.get(url, headers=headers, params=params)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("GET", url, response.status_code, result)

        if response.status_code == 200:
            # 상위 N개만 반환
            if "output" in result:
                result["output"] = result["output"][:top_n]
            return result
        else:
            raise Exception(f"등락률 순위 조회 실패: {response.text}")

    def get_order_history(self) -> Dict:
        """주문 내역 조회"""
        tr_id = "VTTC8001R" if self.use_mock else "TTTC8001R"
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"

        headers = self._get_headers(tr_id)
        params = {
            "CANO": self.account_number,
            "ACNT_PRDT_CD": self.account_product_code,
            "INQR_STRT_DT": (datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
            "INQR_END_DT": datetime.now().strftime("%Y%m%d"),
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }

        self._log_request("GET", url, headers, params=params)
        response = requests.get(url, headers=headers, params=params)

        result = response.json() if response.status_code == 200 else {"error": response.text}
        self._log_response("GET", url, response.status_code, result)

        if response.status_code == 200:
            return result
        else:
            raise Exception(f"주문 내역 조회 실패: {response.text}")
        


@lru_cache()
def get_kis_client() -> KISAPIClient:
    """KIS API 클라이언트 싱글톤"""
    from app.config import get_settings

    settings = get_settings()

    return KISAPIClient(
        app_key=getattr(settings, 'kis_app_key', ''),
        app_secret=getattr(settings, 'kis_app_secret', ''),
        account_number=getattr(settings, 'kis_account_number', ''),
        account_product_code=getattr(settings, 'kis_account_product_code', '01'),
        base_url=getattr(settings, 'kis_base_url', 'https://openapi.koreainvestment.com:9443'),
        use_mock=getattr(settings, 'kis_use_mock', True),
        cust_type=getattr(settings, 'kis_cust_type', "P")
    )
