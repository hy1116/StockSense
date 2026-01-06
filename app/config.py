from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/stocksense"

    model_path: str = "./models"
    data_path: str = "./data"

    log_level: str = "INFO"

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

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings():
    return Settings()
