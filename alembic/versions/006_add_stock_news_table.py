"""Add stock news table

Revision ID: 006
Revises: 005
Create Date: 2025-02-04

Tables:
- stock_news: 종목별 뉴스 크롤링 데이터
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import TEXT


# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 테이블이 이미 존재하는지 체크
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if 'stock_news' not in tables:
        op.create_table(
            'stock_news',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('stock_code', sa.String(length=20), nullable=False),
            sa.Column('stock_name', sa.String(length=100), nullable=True),
            sa.Column('title', sa.String(length=500), nullable=False),
            sa.Column('content', sa.Text(), nullable=True),
            sa.Column('summary', sa.Text(), nullable=True),
            sa.Column('source', sa.String(length=100), nullable=True),
            sa.Column('author', sa.String(length=100), nullable=True),
            sa.Column('url', sa.String(length=1000), nullable=False),
            sa.Column('image_url', sa.String(length=1000), nullable=True),
            sa.Column('sentiment_score', sa.Integer(), nullable=True),
            sa.Column('sentiment_label', sa.String(length=20), nullable=True),
            sa.Column('published_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('crawled_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
            sa.Column('is_processed', sa.Boolean(), server_default=sa.text('false'), nullable=True),
            sa.Column('is_used_for_training', sa.Boolean(), server_default=sa.text('false'), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_stock_news_id'), 'stock_news', ['id'], unique=False)
        op.create_index(op.f('ix_stock_news_stock_code'), 'stock_news', ['stock_code'], unique=False)
        op.create_index('idx_stock_news_code_date', 'stock_news', ['stock_code', 'published_at'], unique=False)
        op.create_index('idx_stock_news_sentiment', 'stock_news', ['sentiment_label'], unique=False)
        op.create_index('idx_stock_news_crawled', 'stock_news', ['crawled_at'], unique=False)
        op.create_index('idx_stock_news_url', 'stock_news', ['url'], unique=True)


def downgrade() -> None:
    op.drop_index('idx_stock_news_url', table_name='stock_news')
    op.drop_index('idx_stock_news_crawled', table_name='stock_news')
    op.drop_index('idx_stock_news_sentiment', table_name='stock_news')
    op.drop_index('idx_stock_news_code_date', table_name='stock_news')
    op.drop_index(op.f('ix_stock_news_stock_code'), table_name='stock_news')
    op.drop_index(op.f('ix_stock_news_id'), table_name='stock_news')
    op.drop_table('stock_news')
