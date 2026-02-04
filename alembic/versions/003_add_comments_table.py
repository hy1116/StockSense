"""Add comments table

Revision ID: 003
Revises: 002
Create Date: 2025-02-04

Tables:
- comments: 주식 종목별 댓글
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # comments 테이블
    op.create_table(
        'comments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('stock_code', sa.String(length=10), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE')
    )
    op.create_index(op.f('ix_comments_id'), 'comments', ['id'], unique=False)
    op.create_index(op.f('ix_comments_stock_code'), 'comments', ['stock_code'], unique=False)
    op.create_index('idx_comment_stock_created', 'comments', ['stock_code', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_comment_stock_created', table_name='comments')
    op.drop_index(op.f('ix_comments_stock_code'), table_name='comments')
    op.drop_index(op.f('ix_comments_id'), table_name='comments')
    op.drop_table('comments')
