import uuid
from typing import Any

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Source(TimestampMixin, Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workspace_id: Mapped[str] = mapped_column(String(128), index=True, default="default")
    source_type: Mapped[str] = mapped_column(String(32), default="upload")
    name: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parser_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    documents = relationship(
        "Document",
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

