"""
ML용 한국투자증권 Open API 클라이언트
Redis 토큰 캐싱 지원 (Redis 실패 시 메모리 캐시로 fallback)
"""
import os
import json
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Redis 키 상수
REDIS_TOKEN_KEY = "kis:access_token"
REDIS_TOKEN_EXPIRY_KEY = "kis:token_expiry"


class RedisTokenCache:
    """Redis 기반 토큰 캐시"""

    def __init__(self):
        self._redis = None
        self._connected = False
        self._connect()

    def _connect(self):
        """Redis 연결 시도"""
        try:
            import redis

            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", 6379))
            db = int(os.getenv("REDIS_DB", 0))
            password = os.getenv("REDIS_PASSWORD", None)

            self._redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password if password else None,
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
            )
            # 연결 테스트
            self._redis.ping()
            self._connected = True
            logger.info(f"Redis connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using memory cache")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_token(self) -> Optional[str]:
        """Redis에서 토큰 조회"""
        if not self._connected:
            return None

        try:
            token = self._redis.get(REDIS_TOKEN_KEY)
            if token:
                # 만료 시간 확인
                expiry_str = self._redis.get(REDIS_TOKEN_EXPIRY_KEY)
                if expiry_str:
                    expiry = datetime.fromisoformat(expiry_str)
                    if datetime.now() < expiry:
                        logger.debug("Token retrieved from Redis cache")
                        return token
            return None
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            return None

    def set_token(self, token: str, expires_in: int):
        """Redis에 토큰 저장"""
        if not self._connected:
            return

        try:
            # 5분 여유를 두고 만료 시간 설정
            expiry = datetime.now() + timedelta(seconds=expires_in - 300)
            ttl = expires_in - 300

            self._redis.setex(REDIS_TOKEN_KEY, ttl, token)
            self._redis.setex(REDIS_TOKEN_EXPIRY_KEY, ttl, expiry.isoformat())
            logger.info(f"Token cached in Redis (TTL: {ttl}s)")
        except Exception as e:
            logger.warning(f"Redis set failed: {e}")


class KISAPIClient:
    """한국투자증권 OpenAPI 클라이언트 (ML 전용)"""

    # 클래스 레벨 Redis 캐시 (싱글톤처럼 공유)
    _redis_cache: Optional[RedisTokenCache] = None

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

        # 메모리 캐시 (Redis 실패 시 fallback)
        self._access_token = None
        self._token_expiry = None

        # Redis 캐시 초기화 (한 번만)
        if KISAPIClient._redis_cache is None:
            KISAPIClient._redis_cache = RedisTokenCache()

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
        """접근 토큰 발급 (Redis > 메모리 캐시 > API 순서)"""

        # 1. Redis 캐시 확인
        if KISAPIClient._redis_cache and KISAPIClient._redis_cache.is_connected:
            cached_token = KISAPIClient._redis_cache.get_token()
            if cached_token:
                return cached_token

        # 2. 메모리 캐시 확인
        if self._access_token and self._token_expiry:
            if datetime.now() < self._token_expiry:
                logger.debug("Token retrieved from memory cache")
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
            response = requests.post(url, headers=headers, json=data, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if "access_token" in result:
                    access_token = result["access_token"]
                    expires_in = int(result.get("expires_in", 86400))

                    # Redis에 저장
                    if KISAPIClient._redis_cache and KISAPIClient._redis_cache.is_connected:
                        KISAPIClient._redis_cache.set_token(access_token, expires_in)

                    # 메모리에도 저장 (fallback용)
                    self._access_token = access_token
                    self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)

                    logger.info(f"New token obtained (expires in {expires_in}s)")
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

    def get_market_cap_ranking(self, market: str = "J", top_n: int = 30) -> Dict:
        """시가총액 순위 조회

        Args:
            market: 시장 구분 (J: 전체, P: KOSPI, Q: KOSDAQ)
            top_n: 조회할 상위 종목 수 (참고용, API는 고정 개수 반환)

        Returns:
            Dict: API 응답 (output 리스트에 종목 정보 포함)
        """
        tr_id = "FHPST01710000"
        url = f"{self.base_url}/uapi/domestic-stock/v1/ranking/market-cap"

        headers = self._get_headers(tr_id)
        params = {
            "FID_COND_MRKT_DIV_CODE": market,
            "FID_COND_SCR_DIV_CODE": "20174",
            "FID_INPUT_ISCD": "0000",  # 전체
            "FID_DIV_CLS_CODE": "0",   # 전체
            "FID_BLNG_CLS_CODE": "0",  # 전체
            "FID_TRGT_CLS_CODE": "0",
            "FID_TRGT_EXLS_CLS_CODE": "0",
            "FID_INPUT_PRICE_1": "",
            "FID_INPUT_PRICE_2": "",
            "FID_VOL_CNT": "",
            "FID_INPUT_DATE_1": "",
        }

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"시가총액 순위 조회 실패: {response.text}")
