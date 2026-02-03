"""Initial tables for StockSense

Revision ID: 001
Revises:
Create Date: 2025-01-30

Tables:
- stocks: 종목 기본 정보 (검색 + 수집 대상 통합)
- predictions: 주가 예측 기록
- trading_history: 거래 내역
- portfolio_snapshots: 포트폴리오 스냅샷
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # stocks 테이블 (검색 + 수집 대상 통합)
    op.create_table(
        'stocks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('stock_name', sa.String(length=100), nullable=False),
        sa.Column('market', sa.String(length=20), nullable=True),
        sa.Column('sector', sa.String(length=50), nullable=True),
        sa.Column('industry', sa.String(length=50), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('priority', sa.Integer(), nullable=True, server_default=sa.text('0')),
        sa.Column('description', sa.String(length=200), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('stock_code')
    )
    op.create_index('idx_stock_code', 'stocks', ['stock_code'], unique=False)
    op.create_index('idx_stock_active', 'stocks', ['is_active'], unique=False)
    op.create_index(op.f('ix_stocks_id'), 'stocks', ['id'], unique=False)
    op.create_index(op.f('ix_stocks_stock_code'), 'stocks', ['stock_code'], unique=True)

    # predictions 테이블
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('stock_name', sa.String(length=100), nullable=True),
        sa.Column('current_price', sa.Float(), nullable=False),
        sa.Column('predicted_price', sa.Float(), nullable=False),
        sa.Column('prediction_date', sa.String(length=10), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('trend', sa.String(length=20), nullable=True),
        sa.Column('recommendation', sa.String(length=20), nullable=True),
        sa.Column('model_name', sa.String(length=50), nullable=True),
        sa.Column('features', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_prediction_stock_date', 'predictions', ['stock_code', 'prediction_date'], unique=False)
    op.create_index(op.f('ix_predictions_id'), 'predictions', ['id'], unique=False)
    op.create_index(op.f('ix_predictions_stock_code'), 'predictions', ['stock_code'], unique=False)

    # trading_history 테이블
    op.create_table(
        'trading_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('order_id', sa.String(length=50), nullable=True),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('stock_name', sa.String(length=100), nullable=True),
        sa.Column('order_type', sa.String(length=10), nullable=True),
        sa.Column('order_status', sa.String(length=20), nullable=True),
        sa.Column('quantity', sa.Integer(), nullable=False),
        sa.Column('order_price', sa.Float(), nullable=True),
        sa.Column('executed_price', sa.Float(), nullable=True),
        sa.Column('order_time', sa.DateTime(), nullable=False),
        sa.Column('executed_time', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order_id')
    )
    op.create_index('idx_trading_stock_time', 'trading_history', ['stock_code', 'order_time'], unique=False)
    op.create_index(op.f('ix_trading_history_id'), 'trading_history', ['id'], unique=False)
    op.create_index(op.f('ix_trading_history_stock_code'), 'trading_history', ['stock_code'], unique=False)

    # portfolio_snapshots 테이블
    op.create_table(
        'portfolio_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('total_asset', sa.Float(), nullable=True),
        sa.Column('cash', sa.Float(), nullable=True),
        sa.Column('stock_value', sa.Float(), nullable=True),
        sa.Column('total_profit_loss', sa.Float(), nullable=True),
        sa.Column('total_profit_rate', sa.Float(), nullable=True),
        sa.Column('snapshot_date', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_portfolio_snapshots_id'), 'portfolio_snapshots', ['id'], unique=False)
    op.create_index(op.f('ix_portfolio_snapshots_snapshot_date'), 'portfolio_snapshots', ['snapshot_date'], unique=False)


def downgrade() -> None:
    op.drop_table('portfolio_snapshots')
    op.drop_table('trading_history')
    op.drop_table('predictions')
    op.drop_table('stocks')
