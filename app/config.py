from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    database_url: str = "sqlite:///./stocksense.db"

    model_path: str = "./models"
    data_path: str = "./data"

    log_level: str = "INFO"

    # Korea Investment & Securities API
    kis_app_key: str = ""
    kis_app_secret: str = ""
    kis_account_number: str = ""
    kis_account_product_code: str = "01"
    kis_base_url: str = "https://openapi.koreainvestment.com:9443"
    kis_use_mock: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings():
    return Settings()
