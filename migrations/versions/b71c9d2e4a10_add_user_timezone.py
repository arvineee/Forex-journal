"""Add per-user timezone preference.

Revision ID: b71c9d2e4a10
Revises: f14bbf5fca35
"""
from alembic import op
import sqlalchemy as sa

revision = 'b71c9d2e4a10'
down_revision = 'f14bbf5fca35'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('users', sa.Column('timezone', sa.String(length=64), nullable=False, server_default='Africa/Nairobi'))
    op.alter_column('users', 'timezone', server_default=None)


def downgrade():
    op.drop_column('users', 'timezone')
