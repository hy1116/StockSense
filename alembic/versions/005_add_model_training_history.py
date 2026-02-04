"""Add model training history table

Revision ID: 005
Revises: 004
Create Date: 2025-02-04

Tables:
- model_training_history: ML 모델 학습 이력
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'model_training_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('hyperparameters', sa.Text(), nullable=True),
        sa.Column('feature_columns', sa.Text(), nullable=True),
        sa.Column('scaler_type', sa.String(length=50), server_default='MinMaxScaler', nullable=True),
        sa.Column('train_samples', sa.Integer(), nullable=False),
        sa.Column('test_samples', sa.Integer(), nullable=False),
        sa.Column('total_samples', sa.Integer(), nullable=False),
        sa.Column('train_score', sa.Float(), nullable=False),
        sa.Column('test_score', sa.Float(), nullable=False),
        sa.Column('mae', sa.Float(), nullable=True),
        sa.Column('rmse', sa.Float(), nullable=True),
        sa.Column('mape', sa.Float(), nullable=True),
        sa.Column('model_binary', sa.LargeBinary(), nullable=True),
        sa.Column('scaler_binary', sa.LargeBinary(), nullable=True),
        sa.Column('is_active', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('is_production', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.Column('trained_by', sa.String(length=50), server_default='batch', nullable=True),
        sa.Column('training_duration_sec', sa.Float(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('trained_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_training_history_id'), 'model_training_history', ['id'], unique=False)
    op.create_index('idx_model_is_active', 'model_training_history', ['is_active'], unique=False)
    op.create_index('idx_model_trained_at', 'model_training_history', ['trained_at'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_model_trained_at', table_name='model_training_history')
    op.drop_index('idx_model_is_active', table_name='model_training_history')
    op.drop_index(op.f('ix_model_training_history_id'), table_name='model_training_history')
    op.drop_table('model_training_history')
