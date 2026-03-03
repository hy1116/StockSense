"""주식 재무 데이터 모델"""
from sqlalchemy import Column, Integer, String, Float, Date, UniqueConstraint, Index
from sqlalchemy.sql import func
from sqlalchemy import DateTime

from app.database import Base


class StockFinancial(Base):
    """종목별 재무 데이터 테이블 (PER, PBR, ROE 등)"""
    __tablename__ = "stock_financials"

    id = Column(Integer, primary_key=True, index=True)

    # 종목 정보
    stock_code = Column(String(20), nullable=False, index=True)
    stock_name = Column(String(100), nullable=True)

    # 날짜 (YYYY-MM-DD, 수집일 기준)
    date = Column(Date, nullable=False)

    # FDR 기반 (KRX 전체 일괄 수집)
    per = Column(Float, nullable=True)         # PER (배)
    pbr = Column(Float, nullable=True)         # PBR (배)
    eps = Column(Float, nullable=True)         # EPS (원)
    bps = Column(Float, nullable=True)         # BPS (원)
    div_yield = Column(Float, nullable=True)   # 배당수익률 (%)

    # Naver Finance API 보완
    roe = Column(Float, nullable=True)              # ROE (%)
    revenue = Column(Float, nullable=True)          # 매출액 (억원)
    operating_profit = Column(Float, nullable=True) # 영업이익 (억원)
    net_profit = Column(Float, nullable=True)       # 순이익 (억원)

    # 데이터 소스
    source = Column(String(50), nullable=True)

    # 수집 시간
    collected_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint('stock_code', 'date', name='uq_stock_financials_code_date'),
        Index('idx_stock_financials_code_date', 'stock_code', 'date'),
    )

    def __repr__(self):
        return f"<StockFinancial(stock_code={self.stock_code}, date={self.date}, per={self.per})>"
