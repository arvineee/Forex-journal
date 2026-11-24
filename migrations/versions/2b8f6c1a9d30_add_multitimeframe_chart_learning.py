"""Add persisted multi-timeframe chart analyses and outcome labels.

Revision ID: 2b8f6c1a9d30
Revises: f14bbf5fca35
"""
from alembic import op
import sqlalchemy as sa

revision = "2b8f6c1a9d30"
down_revision = "f14bbf5fca35"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ai_chart_analyses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("trade_id", sa.Integer(), sa.ForeignKey("trades.id", ondelete="SET NULL"), nullable=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=True),
        sa.Column("analysis_json", sa.Text(), nullable=False),
        sa.Column("image_files_json", sa.Text(), nullable=False),
        sa.Column("outcome_label", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_ai_chart_analyses_user_id", "ai_chart_analyses", ["user_id"])
    op.create_index("ix_ai_chart_analyses_trade_id", "ai_chart_analyses", ["trade_id"])
    op.create_index("ix_ai_chart_analyses_symbol", "ai_chart_analyses", ["symbol"])
    op.create_index("ix_ai_chart_analyses_created_at", "ai_chart_analyses", ["created_at"])


def downgrade():
    op.drop_index("ix_ai_chart_analyses_created_at", table_name="ai_chart_analyses")
    op.drop_index("ix_ai_chart_analyses_symbol", table_name="ai_chart_analyses")
    op.drop_index("ix_ai_chart_analyses_trade_id", table_name="ai_chart_analyses")
    op.drop_index("ix_ai_chart_analyses_user_id", table_name="ai_chart_analyses")
    op.drop_table("ai_chart_analyses")
