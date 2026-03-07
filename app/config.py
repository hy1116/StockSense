from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/stocksense"

    model_path: str = "./models"
    data_path: str = "./data"

    log_level: str = "INFO"

    # Google Gemini API
    gemini_api_key: str = ""

    # Stock API Keys (optional)
    alpha_vantage_api_key: str = ""
    finnhub_api_key: str = ""

    # Korea Investment & Securities API
    kis_app_key: str = ""
    kis_app_secret: str = ""
    kis_account_number: str = ""
    kis_account_product_code: str = "01"
    kis_base_url: str = "https://openapi.koreainvestment.com:9443"
    kis_use_mock: bool = False
    kis_cust_type: str = "P"

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # JWT Configuration
    jwt_secret_key: str = "your-super-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24시간

    # Session Configuration
    session_expire_seconds: int = 60 * 60 * 24  # 24시간

    # Kafka Configuration
    # 로컬 개발: localhost:9092 / Docker 컨테이너: kafka:29092
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_stock_price_topic: str = "stock-price"
    kafka_alert_triggered_topic: str = "price-alert-triggered"
    kafka_poll_interval_seconds: int = 10  # 주가 폴링 주기 (초)

    # CORS Configuration
    cors_origins: str = "https://stocksense.hypepia.com,http://localhost:3000"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings():
    return Settings()
