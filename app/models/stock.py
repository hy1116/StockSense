"""주식 관련 데이터베이스 모델"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Index
from sqlalchemy.sql import func
from datetime import datetime

from app.database import Base


class Stock(Base):
    """종목 기본 정보 (검색 + 수집 대상 통합)"""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), unique=True, nullable=False, index=True)
    stock_name = Column(String(100), nullable=False)
    market = Column(String(20))  # KOSPI, KOSDAQ
    sector = Column(String(50))  # 섹터
    industry = Column(String(50))  # 업종
    is_active = Column(Boolean, default=True, nullable=False)  # 수집 활성화 여부
    priority = Column(Integer, default=0)  # 우선순위 (높을수록 먼저)
    description = Column(String(200))  # 설명 (예: 시가총액 상위)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_stock_code', 'stock_code'),
        Index('idx_stock_active', 'is_active'),
    )


class Prediction(Base):
    """주가 예측 기록"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(100))

    current_price = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=False)
    prediction_date = Column(String(10))  # YYYY-MM-DD

    confidence = Column(Float)  # 신뢰도 (0-1)
    trend = Column(String(20))  # 상승/하락/보합
    recommendation = Column(String(20))  # 매수/매도/보유

    model_name = Column(String(50))  # 사용한 모델명
    features = Column(Text)  # JSON 형태로 저장된 특징값

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_prediction_stock_date', 'stock_code', 'prediction_date'),
    )


class TradingHistory(Base):
    """거래 내역"""
    __tablename__ = "trading_history"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String(50), unique=True)

    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(100))

    order_type = Column(String(10))  # buy, sell
    order_status = Column(String(20))  # 접수, 체결, 취소

    quantity = Column(Integer, nullable=False)
    order_price = Column(Float)
    executed_price = Column(Float)

    order_time = Column(DateTime, nullable=False)
    executed_time = Column(DateTime)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_trading_stock_time', 'stock_code', 'order_time'),
    )


class Portfolio(Base):
    """포트폴리오 스냅샷"""
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, index=True)

    total_asset = Column(Float)
    cash = Column(Float)
    stock_value = Column(Float)

    total_profit_loss = Column(Float)
    total_profit_rate = Column(Float)

    snapshot_date = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())