"""Add nickname to users table

Revision ID: 004
Revises: 003
Create Date: 2025-02-04

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # nickname 컬럼 추가 (기존 유저는 username을 기본값으로)
    op.add_column('users', sa.Column('nickname', sa.String(length=30), nullable=True))

    # 기존 유저의 nickname을 username으로 설정
    op.execute("UPDATE users SET nickname = username WHERE nickname IS NULL")

    # nullable=False로 변경
    op.alter_column('users', 'nickname', nullable=False)

    # unique index 추가
    op.create_index(op.f('ix_users_nickname'), 'users', ['nickname'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_users_nickname'), table_name='users')
    op.drop_column('users', 'nickname')
