"""인증 관련 스키마"""
from pydantic import BaseModel, Field
from typing import Optional


class RegisterRequest(BaseModel):
    """회원가입 요청"""
    username: str = Field(..., min_length=4, max_length=50, description="사용자 아이디")
    password: str = Field(..., min_length=6, description="비밀번호")
    kis_api_key: str = Field(..., description="한국투자증권 App Key")
    kis_api_secret: str = Field(..., description="한국투자증권 App Secret")
    kis_account_no: str = Field(..., description="계좌번호")
    kis_account_product_code: str = Field(default="01", description="계좌상품코드")


class RegisterResponse(BaseModel):
    """회원가입 응답"""
    success: bool
    message: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    """로그인 요청"""
    username: str = Field(..., description="사용자 아이디")
    password: str = Field(..., description="비밀번호")


class LoginResponse(BaseModel):
    """로그인 응답"""
    success: bool
    message: str
    username: Optional[str] = None
    account_no: Optional[str] = None
    access_token: Optional[str] = None


class UserInfo(BaseModel):
    """사용자 정보"""
    username: str
    account_no: Optional[str] = None
    is_authenticated: bool = True


class LogoutResponse(BaseModel):
    """로그아웃 응답"""
    success: bool
    message: str


class UpdateKisRequest(BaseModel):
    """KIS API 정보 업데이트 요청"""
    kis_api_key: str = Field(..., description="한국투자증권 App Key")
    kis_api_secret: str = Field(..., description="한국투자증권 App Secret")
    kis_account_no: str = Field(..., description="계좌번호")
    kis_account_product_code: str = Field(default="01", description="계좌상품코드")
