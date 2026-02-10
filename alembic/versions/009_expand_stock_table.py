"""Expand stock table for KRX data

Revision ID: 009
Revises: 008
Create Date: 2026-02-10

Changes:
- Add stock_name index
- Add listing_date, par_value, listed_shares columns
- Expand sector, industry column length
- Add category column
- Change is_active default to False for new stocks
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '009'
down_revision = '008'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. 컬럼 타입 변경 (sector, industry 길이 확장)
    op.alter_column('stocks', 'sector',
                    existing_type=sa.String(length=50),
                    type_=sa.String(length=100),
                    existing_nullable=True)

    op.alter_column('stocks', 'industry',
                    existing_type=sa.String(length=50),
                    type_=sa.String(length=100),
                    existing_nullable=True)

    # 2. market 컬럼 타입 확장 (KONEX 추가)
    op.alter_column('stocks', 'market',
                    existing_type=sa.String(length=20),
                    type_=sa.String(length=20),
                    existing_nullable=True)

    # 3. 새 컬럼 추가
    op.add_column('stocks', sa.Column('listing_date', sa.String(length=10), nullable=True))
    op.add_column('stocks', sa.Column('par_value', sa.Integer(), nullable=True))
    op.add_column('stocks', sa.Column('listed_shares', sa.BigInteger(), nullable=True))
    op.add_column('stocks', sa.Column('category', sa.String(length=50), nullable=True))

    # 4. 인덱스 추가
    op.create_index('idx_stock_name', 'stocks', ['stock_name'], unique=False)
    op.create_index('idx_stock_market', 'stocks', ['market'], unique=False)

    # 5. is_active 기본값 변경
    # 주의: 기존 데이터는 그대로 유지, 새로 추가되는 데이터만 False가 기본값
    op.alter_column('stocks', 'is_active',
                    existing_type=sa.Boolean(),
                    server_default=sa.text('false'),
                    existing_nullable=False)


def downgrade() -> None:
    # 인덱스 삭제
    op.drop_index('idx_stock_market', table_name='stocks')
    op.drop_index('idx_stock_name', table_name='stocks')

    # 컬럼 삭제
    op.drop_column('stocks', 'category')
    op.drop_column('stocks', 'listed_shares')
    op.drop_column('stocks', 'par_value')
    op.drop_column('stocks', 'listing_date')

    # is_active 기본값 복원
    op.alter_column('stocks', 'is_active',
                    existing_type=sa.Boolean(),
                    server_default=sa.text('true'),
                    existing_nullable=False)

    # 컬럼 타입 복원
    op.alter_column('stocks', 'industry',
                    existing_type=sa.String(length=100),
                    type_=sa.String(length=50),
                    existing_nullable=True)

    op.alter_column('stocks', 'sector',
                    existing_type=sa.String(length=100),
                    type_=sa.String(length=50),
                    existing_nullable=True)
