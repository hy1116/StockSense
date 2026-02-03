"""인증 API 라우터"""
import logging
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Response, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.schemas.auth import LoginRequest, LoginResponse, UserInfo, LogoutResponse
from app.services.auth import (
    create_access_token,
    verify_token,
    get_session_manager,
    SessionManager
)
from app.services.kis_api import KISAPIClient
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)
settings = get_settings()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[UserInfo]:
    """현재 인증된 사용자 정보 조회

    Authorization 헤더 또는 Cookie에서 토큰을 확인
    """
    token = None

    # 1. Authorization 헤더에서 토큰 추출
    if credentials:
        token = credentials.credentials

    # 2. Cookie에서 토큰 추출 (fallback)
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    # JWT 검증
    payload = verify_token(token)
    if not payload:
        return None

    session_id = payload.get("session_id")
    if not session_id:
        return None

    # Redis에서 세션 확인
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        return None

    return UserInfo(
        account_no=session["account_no"],
        is_authenticated=True
    )


async def require_auth(
    user: Optional[UserInfo] = Depends(get_current_user)
) -> UserInfo:
    """인증 필수 의존성"""
    if not user:
        raise HTTPException(
            status_code=401,
            detail="인증이 필요합니다"
        )
    return user


def get_user_credentials(session_id: str) -> Optional[dict]:
    """세션에서 사용자 credentials 조회 (KIS API 호출용)"""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        return None

    return session.get("credentials")


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    """로그인 - KIS API 인증 후 JWT 발급"""
    try:
        # 1. KIS API로 토큰 발급 테스트 (credentials 유효성 검증)
        logger.info(f"Login attempt for account: {request.account_no[:4]}****")

        kis_client = KISAPIClient(
            app_key=request.api_key,
            app_secret=request.api_secret,
            account_number=request.account_no,
            account_product_code=request.account_product_code,
            base_url=settings.kis_base_url,
            use_mock=settings.kis_use_mock
        )

        # 간단한 API 호출로 credentials 검증 (잔고 조회)
        try:
            balance = kis_client.get_balance()
            if balance.get("rt_cd") != "0":
                error_msg = balance.get("msg1", "API 인증 실패")
                logger.warning(f"KIS API validation failed: {error_msg}")
                return LoginResponse(
                    success=False,
                    message=f"API 인증 실패: {error_msg}"
                )
        except Exception as e:
            logger.error(f"KIS API call failed: {e}")
            return LoginResponse(
                success=False,
                message=f"API 연결 실패: {str(e)}"
            )

        # 2. Redis에 세션 저장 (암호화된 credentials)
        session_manager = get_session_manager()
        session_id = session_manager.create_session(
            account_no=request.account_no,
            credentials={
                "api_key": request.api_key,
                "api_secret": request.api_secret,
                "account_no": request.account_no,
                "account_product_code": request.account_product_code
            }
        )

        # 3. JWT 토큰 생성
        access_token = create_access_token(
            data={
                "session_id": session_id,
                "account_no": request.account_no
            },
            expires_delta=timedelta(minutes=settings.jwt_expire_minutes)
        )

        # 4. HttpOnly Cookie 설정 (선택적)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=False,  # HTTPS에서는 True로 변경
            samesite="lax",
            max_age=settings.session_expire_seconds
        )

        logger.info(f"Login successful for account: {request.account_no[:4]}****")

        return LoginResponse(
            success=True,
            message="로그인 성공",
            account_no=request.account_no,
            access_token=access_token
        )

    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"로그인 처리 중 오류: {str(e)}")


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    response: Response,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """로그아웃 - 세션 삭제"""
    token = None

    if credentials:
        token = credentials.credentials
    if not token:
        token = request.cookies.get("access_token")

    if token:
        payload = verify_token(token)
        if payload:
            session_id = payload.get("session_id")
            if session_id:
                session_manager = get_session_manager()
                session_manager.delete_session(session_id)

    # Cookie 삭제
    response.delete_cookie("access_token")

    return LogoutResponse(
        success=True,
        message="로그아웃 되었습니다"
    )


@router.get("/me", response_model=UserInfo)
async def get_me(user: UserInfo = Depends(require_auth)):
    """현재 로그인한 사용자 정보"""
    return user


@router.get("/check")
async def check_auth(user: Optional[UserInfo] = Depends(get_current_user)):
    """인증 상태 확인 (에러 없이)"""
    if user:
        return {
            "authenticated": True,
            "account_no": user.account_no
        }
    return {
        "authenticated": False,
        "account_no": None
    }
