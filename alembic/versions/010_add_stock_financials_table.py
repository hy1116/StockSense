"""Add stock_financials table

Revision ID: 010
Revises: 009
Create Date: 2026-03-04

Changes:
- Add stock_financials table for PER, PBR, ROE, EPS, BPS, dividend yield, revenue, operating profit, net profit
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '010'
down_revision = '009'
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    exists = conn.execute(sa.text(
        "SELECT 1 FROM information_schema.tables WHERE table_name='stock_financials'"
    )).fetchone()
    if exists:
        return

    op.create_table(
        'stock_financials',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('stock_code', sa.String(20), nullable=False),
        sa.Column('stock_name', sa.String(100), nullable=True),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('per', sa.Float(), nullable=True),
        sa.Column('pbr', sa.Float(), nullable=True),
        sa.Column('eps', sa.Float(), nullable=True),
        sa.Column('bps', sa.Float(), nullable=True),
        sa.Column('div_yield', sa.Float(), nullable=True),
        sa.Column('roe', sa.Float(), nullable=True),
        sa.Column('revenue', sa.Float(), nullable=True),
        sa.Column('operating_profit', sa.Float(), nullable=True),
        sa.Column('net_profit', sa.Float(), nullable=True),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('collected_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint('stock_code', 'date', name='uq_stock_financials_code_date'),
    )
    op.create_index('idx_stock_financials_code_date', 'stock_financials', ['stock_code', 'date'])
    op.create_index('idx_stock_financials_code', 'stock_financials', ['stock_code'])


def downgrade():
    op.drop_index('idx_stock_financials_code', table_name='stock_financials')
    op.drop_index('idx_stock_financials_code_date', table_name='stock_financials')
    op.drop_table('stock_financials')
