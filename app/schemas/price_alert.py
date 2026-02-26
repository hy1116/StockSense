"""주가 알림 Pydantic 스키마"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Literal


class PriceAlertCreate(BaseModel):
    """알림 생성 요청"""
    stock_code: str = Field(..., min_length=6, max_length=6, examples=["005930"])
    stock_name: Optional[str] = Field(None, examples=["삼성전자"])
    condition: Literal["above", "below"] = Field(
        ..., description="above: 목표가 이상 알림 / below: 목표가 이하 알림"
    )
    target_price: float = Field(..., gt=0, description="알림 목표 가격 (원)")


class PriceAlertUpdate(BaseModel):
    """알림 수정 요청"""
    is_active: Optional[bool] = None
    target_price: Optional[float] = Field(None, gt=0)
    condition: Optional[Literal["above", "below"]] = None


class PriceAlertResponse(BaseModel):
    """알림 응답"""
    id: int
    user_id: int
    stock_code: str
    stock_name: Optional[str]
    condition: str
    target_price: float
    is_active: bool
    is_triggered: bool
    triggered_price: Optional[float]
    triggered_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True
