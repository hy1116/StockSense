"""관심종목 API 라우터"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, text

from app.database import get_db
from app.models.watchlist import Watchlist
from app.models.stock import Stock
from app.schemas.watchlist import (
    WatchlistAddRequest, WatchlistItemResponse,
    WatchlistListResponse, WatchlistCheckResponse
)
from app.services.auth import verify_token, get_session_manager
from app.services.kis_api import get_kis_client, KISAPIClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])
security = HTTPBearer(auto_error=False)


# ===== 인증 의존성 (comment.py 패턴 재사용) =====

async def get_current_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[int]:
    """현재 인증된 사용자 ID 조회"""
    token = None

    if credentials:
        token = credentials.credentials
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    payload = verify_token(token)
    if not payload:
        return None

    session_id = payload.get("session_id")
    if not session_id:
        return None

    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)

    if not session:
        return None

    return session.get("user_id")


async def require_auth_user_id(
    user_id: Optional[int] = Depends(get_current_user_id)
) -> int:
    """인증 필수 - 사용자 ID 반환"""
    if not user_id:
        raise HTTPException(status_code=401, detail="인증이 필요합니다")
    return user_id


# ===== API 엔드포인트 =====

@router.get("/", response_model=WatchlistListResponse)
async def get_watchlist(
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(require_auth_user_id),
    client: KISAPIClient = Depends(get_kis_client)
):
    """관심종목 목록 조회 (현재가 포함)"""
    try:
        # 관심종목 + stocks 테이블 조인으로 종목명/시장 가져오기
        query = (
            select(Watchlist, Stock.stock_name, Stock.market)
            .outerjoin(Stock, Watchlist.stock_code == Stock.stock_code)
            .where(Watchlist.user_id == current_user_id)
            .order_by(Watchlist.created_at.desc())
        )
        result = await db.execute(query)
        rows = result.all()

        items = []
        for watchlist_item, stock_name, market in rows:
            # KIS API로 현재가 조회
            current_price = None
            change_rate = None
            change_price = None

            try:
                price_result = client.get_stock_price(watchlist_item.stock_code)
                if price_result.get("rt_cd") == "0":
                    output = price_result.get("output", {})
                    current_price = int(output.get("stck_prpr", 0))
                    change_rate = float(output.get("prdy_ctrt", 0))
                    change_price = int(output.get("prdy_vrss", 0))
                    # stocks 테이블에 종목명이 없으면 KIS 응답에서 가져오기
                    if not stock_name:
                        stock_name = output.get("hts_kor_isnm", "")
            except Exception as e:
                logger.warning(f"Failed to fetch price for {watchlist_item.stock_code}: {e}")

            items.append(WatchlistItemResponse(
                id=watchlist_item.id,
                stock_code=watchlist_item.stock_code,
                stock_name=stock_name or watchlist_item.stock_code,
                market=market,
                current_price=current_price,
                change_rate=change_rate,
                change_price=change_price,
                created_at=watchlist_item.created_at
            ))

        return WatchlistListResponse(items=items, total=len(items))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="관심종목 조회 중 오류가 발생했습니다")


@router.post("/{stock_code}")
async def add_to_watchlist(
    stock_code: str,
    request_data: WatchlistAddRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(require_auth_user_id)
):
    """관심종목 등록 + stocks 테이블에 수집대상 자동 추가"""
    try:
        # 1. 이미 등록되어 있는지 확인
        existing = await db.execute(
            select(Watchlist).where(
                Watchlist.user_id == current_user_id,
                Watchlist.stock_code == stock_code
            )
        )
        if existing.scalar_one_or_none():
            return {"success": True, "message": "이미 관심종목에 등록되어 있습니다"}

        # 2. 관심종목 추가
        new_item = Watchlist(
            user_id=current_user_id,
            stock_code=stock_code
        )
        db.add(new_item)

        # 3. stocks 테이블에 수집대상으로 자동 추가 (ON CONFLICT 시 is_active=true로 업데이트)
        await db.execute(
            text("""
                INSERT INTO stocks (stock_code, stock_name, market, is_active, priority, description)
                VALUES (:code, :name, :market, true, 0, '관심종목 등록')
                ON CONFLICT (stock_code)
                DO UPDATE SET is_active = true, updated_at = NOW()
            """),
            {
                "code": stock_code,
                "name": request_data.stock_name,
                "market": request_data.market,
            }
        )

        await db.commit()
        logger.info(f"Watchlist added: user={current_user_id}, stock={stock_code}, also added to collection targets")

        return {"success": True, "message": "관심종목에 등록되었습니다"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="관심종목 등록 중 오류가 발생했습니다")


@router.delete("/{stock_code}")
async def remove_from_watchlist(
    stock_code: str,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(require_auth_user_id)
):
    """관심종목 해제 (stocks 테이블은 건드리지 않음)"""
    try:
        result = await db.execute(
            delete(Watchlist).where(
                Watchlist.user_id == current_user_id,
                Watchlist.stock_code == stock_code
            )
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="관심종목에 등록되지 않은 종목입니다")

        await db.commit()
        logger.info(f"Watchlist removed: user={current_user_id}, stock={stock_code}")

        return {"success": True, "message": "관심종목에서 제거되었습니다"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(status_code=500, detail="관심종목 해제 중 오류가 발생했습니다")


@router.get("/check/{stock_code}", response_model=WatchlistCheckResponse)
async def check_watchlist(
    stock_code: str,
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(require_auth_user_id)
):
    """관심종목 등록 여부 확인"""
    try:
        result = await db.execute(
            select(Watchlist).where(
                Watchlist.user_id == current_user_id,
                Watchlist.stock_code == stock_code
            )
        )
        exists = result.scalar_one_or_none() is not None

        return WatchlistCheckResponse(
            is_watchlisted=exists,
            stock_code=stock_code
        )

    except Exception as e:
        logger.error(f"Error checking watchlist: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="관심종목 확인 중 오류가 발생했습니다")
