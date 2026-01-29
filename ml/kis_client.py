"""
ML용 한국투자증권 Open API 클라이언트 (독립형)
Redis 의존성 없이 동작
"""
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict

logger = logging.getLogger(__name__)


class KISAPIClient:
    """한국투자증권 OpenAPI 클라이언트 (ML 전용)"""

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

        # 메모리 캐시
        self._access_token = None
        self._token_expiry = None

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
        """접근 토큰 발급 (메모리 캐시 사용)"""
        # 캐시된 토큰이 유효한지 확인
        if self._access_token and self._token_expiry:
            if datetime.now() < self._token_expiry:
                return self._access_token

        # 새로운 토큰 발급
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"Content-Type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if "access_token" in result:
                    access_token = result["access_token"]
                    expires_in = int(result.get("expires_in", 86400))

                    # 메모리에 저장 (5분 여유)
                    self._access_token = access_token
                    self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

                    logger.info(f"Token obtained (expires in {expires_in}s)")
                    return access_token
                else:
                    raise Exception(f"토큰 발급 실패 - access_token 없음: {result}")
            else:
                raise Exception(f"토큰 발급 실패 (HTTP {response.status_code}): {response.text}")

        except requests.exceptions.RequestException as e:
            raise Exception(f"토큰 발급 네트워크 오류: {str(e)}")

    def get_stock_price(self, stock_code: str) -> Dict:
        """주식 현재가 조회"""
        tr_id = "FHKST01010100"
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"

        headers = self._get_headers(tr_id)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"현재가 조회 실패: {response.text}")

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

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"차트 데이터 조회 실패: {response.text}")
