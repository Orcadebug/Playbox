"""add session corpora

Revision ID: 20260424_0002
Revises: 20260424_0001
Create Date: 2026-04-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20260424_0002"
down_revision: str | None = "20260424_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "corpora",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("retention", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("ttl_seconds", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_count", sa.Integer(), nullable=False),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("byte_count", sa.Integer(), nullable=False),
        sa.Column("window_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_corpora_status", "corpora", ["status"], unique=False)
    op.create_index("ix_corpora_workspace_id", "corpora", ["workspace_id"], unique=False)
    op.add_column("sources", sa.Column("corpus_id", sa.String(length=36), nullable=True))
    op.create_index("ix_sources_corpus_id", "sources", ["corpus_id"], unique=False)
    op.create_foreign_key(
        "fk_sources_corpus_id_corpora",
        "sources",
        "corpora",
        ["corpus_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_sources_corpus_id_corpora", "sources", type_="foreignkey")
    op.drop_index("ix_sources_corpus_id", table_name="sources")
    op.drop_column("sources", "corpus_id")
    op.drop_index("ix_corpora_workspace_id", table_name="corpora")
    op.drop_index("ix_corpora_status", table_name="corpora")
    op.drop_table("corpora")
