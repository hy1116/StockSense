"""주가 알림 REST API"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.database import get_db
from app.models.price_alert import PriceAlert
from app.models.user import User
from app.schemas.price_alert import PriceAlertCreate, PriceAlertResponse, PriceAlertUpdate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/alerts", tags=["price-alerts"])


def _require_user(current_user) -> User:
    """인증 필수 헬퍼 - 미인증 시 401"""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다",
        )
    return current_user


# ─── 알림 생성 ─────────────────────────────────────────────────────
@router.post("", response_model=PriceAlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    data: PriceAlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """주가 알림 등록

    - condition: `above` (목표가 이상) / `below` (목표가 이하)
    - 한 사용자가 같은 종목에 여러 조건 등록 가능
    """
    user = _require_user(current_user)

    alert = PriceAlert(
        user_id=user.id,
        stock_code=data.stock_code,
        stock_name=data.stock_name,
        condition=data.condition,
        target_price=data.target_price,
    )
    db.add(alert)
    await db.flush()
    await db.refresh(alert)
    return alert


# ─── 내 알림 목록 ──────────────────────────────────────────────────
@router.get("", response_model=List[PriceAlertResponse])
async def get_my_alerts(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """내 알림 전체 조회 (최신순)"""
    user = _require_user(current_user)

    result = await db.execute(
        select(PriceAlert)
        .where(PriceAlert.user_id == user.id)
        .order_by(PriceAlert.created_at.desc())
    )
    return result.scalars().all()


# ─── 알림 수정 ─────────────────────────────────────────────────────
@router.patch("/{alert_id}", response_model=PriceAlertResponse)
async def update_alert(
    alert_id: int,
    data: PriceAlertUpdate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """알림 조건 수정 or 활성화/비활성화 토글"""
    user = _require_user(current_user)

    result = await db.execute(
        select(PriceAlert).where(
            and_(PriceAlert.id == alert_id, PriceAlert.user_id == user.id)
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다")

    for field, value in data.model_dump(exclude_none=True).items():
        setattr(alert, field, value)

    await db.flush()
    await db.refresh(alert)
    return alert


# ─── 알림 삭제 ─────────────────────────────────────────────────────
@router.delete("/{alert_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """알림 삭제"""
    user = _require_user(current_user)

    result = await db.execute(
        select(PriceAlert).where(
            and_(PriceAlert.id == alert_id, PriceAlert.user_id == user.id)
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다")

    await db.delete(alert)


# ─── 발동된 알림 초기화 ────────────────────────────────────────────
@router.post("/{alert_id}/reset", response_model=PriceAlertResponse)
async def reset_alert(
    alert_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """발동된 알림을 초기화해서 다시 활성화"""
    user = _require_user(current_user)

    result = await db.execute(
        select(PriceAlert).where(
            and_(PriceAlert.id == alert_id, PriceAlert.user_id == user.id)
        )
    )
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(status_code=404, detail="알림을 찾을 수 없습니다")

    alert.is_triggered = False
    alert.triggered_price = None
    alert.triggered_at = None
    alert.is_active = True

    await db.flush()
    await db.refresh(alert)
    return alert
