"""Add watchlist table

Revision ID: 008
Revises: 007
Create Date: 2026-02-09

Tables:
- watchlist: 사용자별 관심종목
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '008'
down_revision = '007'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # watchlist 테이블
    op.create_table(
        'watchlist',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('user_id', 'stock_code', name='uq_watchlist_user_stock')
    )
    op.create_index(op.f('ix_watchlist_id'), 'watchlist', ['id'], unique=False)
    op.create_index(op.f('ix_watchlist_user_id'), 'watchlist', ['user_id'], unique=False)
    op.create_index(op.f('ix_watchlist_stock_code'), 'watchlist', ['stock_code'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_watchlist_stock_code'), table_name='watchlist')
    op.drop_index(op.f('ix_watchlist_user_id'), table_name='watchlist')
    op.drop_index(op.f('ix_watchlist_id'), table_name='watchlist')
    op.drop_table('watchlist')
