"""포트폴리오 관련 Pydantic 스키마"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class StockHolding(BaseModel):
    """보유 주식 정보"""
    stock_code: str = Field(..., description="종목코드")
    stock_name: str = Field(..., description="종목명")
    quantity: int = Field(..., description="보유 수량")
    avg_price: float = Field(..., description="평균 매입가")
    current_price: float = Field(..., description="현재가")
    eval_amount: float = Field(..., description="평가금액")
    profit_loss: float = Field(..., description="평가손익")
    profit_rate: float = Field(..., description="수익률(%)")


class PortfolioSummary(BaseModel):
    """포트폴리오 요약"""
    total_asset: float = Field(..., description="총 자산")
    cash: float = Field(..., description="예수금")
    stock_eval_amount: float = Field(..., description="주식 평가금액")
    total_profit_loss: float = Field(..., description="총 평가손익")
    total_profit_rate: float = Field(..., description="총 수익률(%)")
    holdings: List[StockHolding] = Field(default=[], description="보유 종목 리스트")


class OrderRequest(BaseModel):
    """주문 요청"""
    stock_code: str = Field(..., description="종목코드 (6자리)", min_length=6, max_length=6)
    quantity: int = Field(..., description="주문 수량", gt=0)
    price: int = Field(..., description="주문 가격 (0: 시장가)", ge=0)
    order_type: str = Field(default="00", description="주문 유형 (00: 지정가, 01: 시장가)")


class OrderResponse(BaseModel):
    """주문 응답"""
    order_id: str = Field(..., description="주문번호")
    stock_code: str = Field(..., description="종목코드")
    stock_name: Optional[str] = Field(None, description="종목명")
    order_type: str = Field(..., description="주문 구분 (buy/sell)")
    quantity: int = Field(..., description="주문 수량")
    price: int = Field(..., description="주문 가격")
    order_time: datetime = Field(..., description="주문 시간")
    status: str = Field(..., description="주문 상태")
    message: str = Field(..., description="처리 메시지")


class OrderHistory(BaseModel):
    """주문 내역"""
    order_date: str = Field(..., description="주문일자")
    order_id: str = Field(..., description="주문번호")
    stock_code: str = Field(..., description="종목코드")
    stock_name: str = Field(..., description="종목명")
    order_type: str = Field(..., description="매수/매도 구분")
    order_quantity: int = Field(..., description="주문 수량")
    order_price: int = Field(..., description="주문 가격")
    executed_quantity: int = Field(..., description="체결 수량")
    executed_price: int = Field(..., description="체결 가격")
    status: str = Field(..., description="주문 상태")


class StockPriceInfo(BaseModel):
    """주식 현재가 정보"""
    stock_code: str = Field(..., description="종목코드")
    stock_name: str = Field(..., description="종목명")
    current_price: int = Field(..., description="현재가")
    change_price: int = Field(..., description="전일대비")
    change_rate: float = Field(..., description="등락률(%)")
    volume: int = Field(..., description="거래량")
    high_price: int = Field(..., description="고가")
    low_price: int = Field(..., description="저가")
    open_price: int = Field(..., description="시가")
