"""Add prediction accuracy columns

Revision ID: 007
Revises: 006
Create Date: 2026-02-08

predictions 테이블에 평가용 컬럼 추가:
- actual_price: 실제 종가
- error_rate: 오차율 (%)
- direction_correct: 방향 적중 여부
- is_evaluated: 평가 완료 여부
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_columns = [col['name'] for col in inspector.get_columns('predictions')]

    if 'actual_price' not in existing_columns:
        op.add_column('predictions', sa.Column('actual_price', sa.Float(), nullable=True))

    if 'error_rate' not in existing_columns:
        op.add_column('predictions', sa.Column('error_rate', sa.Float(), nullable=True))

    if 'direction_correct' not in existing_columns:
        op.add_column('predictions', sa.Column('direction_correct', sa.Boolean(), nullable=True))

    if 'is_evaluated' not in existing_columns:
        op.add_column('predictions', sa.Column(
            'is_evaluated', sa.Boolean(),
            server_default=sa.text('false'), nullable=False
        ))

    # is_evaluated 인덱스 추가
    existing_indexes = [idx['name'] for idx in inspector.get_indexes('predictions')]
    if 'idx_prediction_is_evaluated' not in existing_indexes:
        op.create_index('idx_prediction_is_evaluated', 'predictions', ['is_evaluated'])


def downgrade() -> None:
    op.drop_index('idx_prediction_is_evaluated', table_name='predictions')
    op.drop_column('predictions', 'is_evaluated')
    op.drop_column('predictions', 'direction_correct')
    op.drop_column('predictions', 'error_rate')
    op.drop_column('predictions', 'actual_price')
