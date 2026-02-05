"""인증 API 라우터"""
import logging
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Response, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.user import User
from app.schemas.auth import (
    RegisterRequest, RegisterResponse,
    LoginRequest, LoginResponse,
    UserInfo, LogoutResponse, UpdateKisRequest
)
from app.services.auth import (
    create_access_token, verify_token,
    hash_password, verify_password,
    encrypt_data, decrypt_data,
    get_session_manager, SessionManager
)
from app.services.kis_api import KISAPIClient
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)
settings = get_settings()


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[UserInfo]:
    """현재 인증된 사용자 정보 조회"""
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
        username=session["username"],
        nickname=session.get("nickname"),
        account_no=session.get("account_no"),
        is_authenticated=True
    )


async def require_auth(
    user: Optional[UserInfo] = Depends(get_current_user)
) -> UserInfo:
    """인증 필수 의존성"""
    if not user:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")
    return user


@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """회원가입 - KIS API 검증 후 사용자 생성"""
    try:
        # 1. 아이디 중복 확인
        result = await db.execute(
            select(User).where(User.username == request.username)
        )
        existing_user = result.scalar_one_or_none()

        if existing_user:
            return RegisterResponse(
                success=False,
                message="이미 사용 중인 아이디입니다"
            )

        # 1-1. 닉네임 중복 확인
        result = await db.execute(
            select(User).where(User.nickname == request.nickname)
        )
        existing_nickname = result.scalar_one_or_none()

        if existing_nickname:
            return RegisterResponse(
                success=False,
                message="이미 사용 중인 닉네임입니다"
            )

        # 2. KIS API 검증
        logger.info(f"Validating KIS API for user: {request.username}")

        try:
            kis_client = KISAPIClient(
                app_key=request.kis_api_key,
                app_secret=request.kis_api_secret,
                account_number=request.kis_account_no,
                account_product_code=request.kis_account_product_code,
                base_url=settings.kis_base_url,
                use_mock=settings.kis_use_mock
            )

            balance = kis_client.get_balance()
            if balance.get("rt_cd") != "0":
                error_msg = balance.get("msg1", "API 인증 실패")
                logger.warning(f"KIS API validation failed: {error_msg}")
                return RegisterResponse(
                    success=False,
                    message=f"KIS API 인증 실패: {error_msg}"
                )
        except Exception as e:
            logger.error(f"KIS API call failed: {e}")
            return RegisterResponse(
                success=False,
                message=f"KIS API 연결 실패: {str(e)}"
            )

        # 3. 사용자 생성 (API Key/Secret 암호화)
        new_user = User(
            username=request.username,
            nickname=request.nickname,
            password_hash=hash_password(request.password),
            kis_api_key=encrypt_data(request.kis_api_key),
            kis_api_secret=encrypt_data(request.kis_api_secret),
            kis_account_no=request.kis_account_no,
            kis_account_product_code=request.kis_account_product_code,
            is_active=True
        )

        db.add(new_user)
        await db.commit()

        logger.info(f"User registered: {request.username}")

        return RegisterResponse(
            success=True,
            message="회원가입이 완료되었습니다",
            username=request.username
        )

    except Exception as e:
        logger.error(f"Registration error: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"회원가입 처리 중 오류: {str(e)}")


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """로그인 - 아이디/비밀번호 인증 후 JWT 발급"""
    try:
        # 1. 사용자 조회
        result = await db.execute(
            select(User).where(User.username == request.username)
        )
        user = result.scalar_one_or_none()

        if not user:
            return LoginResponse(
                success=False,
                message="아이디 또는 비밀번호가 올바르지 않습니다"
            )

        # 2. 비밀번호 확인
        if not verify_password(request.password, user.password_hash):
            return LoginResponse(
                success=False,
                message="아이디 또는 비밀번호가 올바르지 않습니다"
            )

        # 3. 활성 사용자 확인
        if not user.is_active:
            return LoginResponse(
                success=False,
                message="비활성화된 계정입니다"
            )

        # 4. 마지막 로그인 시간 업데이트
        user.last_login_at = datetime.now()
        await db.commit()

        # 5. Redis에 세션 저장
        session_manager = get_session_manager()
        session_id = session_manager.create_session(
            user_id=user.id,
            username=user.username,
            nickname=user.nickname,
            account_no=user.kis_account_no
        )

        # 6. JWT 토큰 생성
        access_token = create_access_token(
            data={
                "session_id": session_id,
                "user_id": user.id,
                "username": user.username
            },
            expires_delta=timedelta(minutes=settings.jwt_expire_minutes)
        )

        # 7. HttpOnly Cookie 설정
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=False,  # HTTPS에서는 True로 변경
            samesite="lax",
            max_age=settings.session_expire_seconds
        )

        logger.info(f"Login successful: {user.username}")

        return LoginResponse(
            success=True,
            message="로그인 성공",
            username=user.username,
            nickname=user.nickname,
            account_no=user.kis_account_no,
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
            "username": user.username,
            "nickname": user.nickname,
            "account_no": user.account_no
        }
    return {
        "authenticated": False,
        "username": None,
        "nickname": None,
        "account_no": None
    }


@router.get("/check-username/{username}")
async def check_username(username: str, db: AsyncSession = Depends(get_db)):
    """아이디 중복 확인"""
    result = await db.execute(
        select(User).where(User.username == username)
    )
    existing = result.scalar_one_or_none()

    return {
        "available": existing is None,
        "message": "사용 가능한 아이디입니다" if existing is None else "이미 사용 중인 아이디입니다"
    }


@router.get("/check-nickname/{nickname}")
async def check_nickname(nickname: str, db: AsyncSession = Depends(get_db)):
    """닉네임 중복 확인"""
    result = await db.execute(
        select(User).where(User.nickname == nickname)
    )
    existing = result.scalar_one_or_none()

    return {
        "available": existing is None,
        "message": "사용 가능한 닉네임입니다" if existing is None else "이미 사용 중인 닉네임입니다"
    }
