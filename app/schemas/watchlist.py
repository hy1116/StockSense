"""관심종목 관련 Pydantic 스키마"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class WatchlistAddRequest(BaseModel):
    """관심종목 추가 요청 (stocks 테이블 자동 등록용)"""
    stock_name: str = Field(..., description="종목명")
    market: Optional[str] = Field(None, description="시장 (KOSPI/KOSDAQ)")


class WatchlistItemResponse(BaseModel):
    """관심종목 개별 항목"""
    id: int = Field(..., description="관심종목 ID")
    stock_code: str = Field(..., description="종목코드")
    stock_name: Optional[str] = Field(None, description="종목명")
    market: Optional[str] = Field(None, description="시장")
    current_price: Optional[int] = Field(None, description="현재가")
    change_rate: Optional[float] = Field(None, description="등락률")
    change_price: Optional[int] = Field(None, description="전일 대비")
    created_at: datetime = Field(..., description="등록일시")

    class Config:
        from_attributes = True


class WatchlistListResponse(BaseModel):
    """관심종목 목록 응답"""
    items: List[WatchlistItemResponse] = Field(default=[], description="관심종목 리스트")
    total: int = Field(..., description="총 관심종목 수")


class WatchlistCheckResponse(BaseModel):
    """관심종목 등록 여부 확인"""
    is_watchlisted: bool = Field(..., description="관심종목 등록 여부")
    stock_code: str = Field(..., description="종목코드")
