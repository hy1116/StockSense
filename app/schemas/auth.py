"""인증 관련 스키마"""
from pydantic import BaseModel, Field
from typing import Optional


class LoginRequest(BaseModel):
    """로그인 요청"""
    api_key: str = Field(..., description="한국투자증권 App Key")
    api_secret: str = Field(..., description="한국투자증권 App Secret")
    account_no: str = Field(..., description="계좌번호 (예: 12345678)")
    account_product_code: str = Field(default="01", description="계좌상품코드")


class LoginResponse(BaseModel):
    """로그인 응답"""
    success: bool
    message: str
    account_no: Optional[str] = None
    access_token: Optional[str] = None


class UserInfo(BaseModel):
    """사용자 정보"""
    account_no: str
    is_authenticated: bool = True


class LogoutResponse(BaseModel):
    """로그아웃 응답"""
    success: bool
    message: str
