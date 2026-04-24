import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Corpus(TimestampMixin, Base):
    __tablename__ = "corpora"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workspace_id: Mapped[str] = mapped_column(String(128), index=True, default="default")
    name: Mapped[str] = mapped_column(String(255))
    retention: Mapped[str] = mapped_column(String(32), default="session")
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=86_400)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_count: Mapped[int] = mapped_column(Integer, default=0)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    byte_count: Mapped[int] = mapped_column(Integer, default=0)
    window_count: Mapped[int] = mapped_column(Integer, default=0)

    sources = relationship(
        "Source",
        back_populates="corpus",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
