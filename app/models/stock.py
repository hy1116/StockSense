"""주식 관련 데이터베이스 모델"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Index, BigInteger
from sqlalchemy.sql import func
from datetime import datetime

from app.database import Base


class Stock(Base):
    """종목 기본 정보 (전체 상장 종목 + 수집 대상 관리)"""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    stock_code = Column(String(10), unique=True, nullable=False, index=True)
    stock_name = Column(String(100), nullable=False, index=True)

    # 상장 정보 (KRX 파일)
    market = Column(String(20))  # KOSPI, KOSDAQ, KONEX
    sector = Column(String(100))  # 업종
    industry = Column(String(100))  # 업종 세부

    # 기업 정보 (선택적)
    listing_date = Column(String(10))  # 상장일 (YYYY-MM-DD)
    par_value = Column(Integer)  # 액면가
    listed_shares = Column(BigInteger)  # 상장주식수

    # 수집 관리 (기본값 False로 변경 - 새로 추가되는 종목은 비활성)
    is_active = Column(Boolean, default=False, nullable=False)  # 수집 활성화 여부
    priority = Column(Integer, default=0)  # 우선순위 (높을수록 먼저)
    category = Column(String(50))  # 시가총액TOP, 거래량TOP 등
    description = Column(String(200))  # 메모

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_stock_code', 'stock_code'),
        Index('idx_stock_name', 'stock_name'),
        Index('idx_stock_active', 'is_active'),
        Index('idx_stock_market', 'market'),
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

    # 예측 평가용 컬럼
    actual_price = Column(Float, nullable=True)  # 실제 종가
    error_rate = Column(Float, nullable=True)  # 오차율 (%)
    direction_correct = Column(Boolean, nullable=True)  # 방향 적중 여부
    is_evaluated = Column(Boolean, default=False, nullable=False)  # 평가 완료 여부

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_prediction_stock_date', 'stock_code', 'prediction_date'),
        Index('idx_prediction_is_evaluated', 'is_evaluated'),
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