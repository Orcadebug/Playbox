from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.errors import api_error
from app.models import Corpus, Document, Source
from app.services.search import invalidate_workspace_cache

RetentionMode = Literal["session", "persistent"]


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _is_expired(expires_at: datetime | None) -> bool:
    return expires_at is not None and _as_utc(expires_at) <= _now()


class CorpusService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_corpus(
        self,
        *,
        workspace_id: str,
        name: str,
        retention: RetentionMode = "session",
        ttl_seconds: int | None = None,
    ) -> dict:
        settings = get_settings()
        if retention == "persistent":
            raise api_error(
                422,
                "persistent_kb_not_enabled",
                "Persistent KB is a separate future tier and is not enabled for this API.",
            )
        if retention != "session":
            raise api_error(
                422,
                "invalid_retention",
                "retention must be 'session' for short-lived corpora.",
            )

        ttl = ttl_seconds or settings.waver_session_corpus_default_ttl_seconds
        if ttl <= 0 or ttl > settings.waver_session_corpus_max_ttl_seconds:
            raise api_error(
                422,
                "invalid_ttl_seconds",
                "ttl_seconds is outside the allowed session corpus range.",
                details={"max": settings.waver_session_corpus_max_ttl_seconds},
            )

        corpus = Corpus(
            workspace_id=workspace_id,
            name=name,
            retention=retention,
            status="active",
            ttl_seconds=ttl,
            expires_at=_now() + timedelta(seconds=ttl),
        )
        self.session.add(corpus)
        await self.session.commit()
        await self.session.refresh(corpus)
        return self.serialize(corpus)

    async def list_corpora(self, *, workspace_id: str) -> list[dict]:
        await self.expire_due_corpora(workspace_id=workspace_id)
        rows = await self.session.scalars(
            select(Corpus)
            .where(Corpus.workspace_id == workspace_id, Corpus.status == "active")
            .order_by(Corpus.created_at.desc())
        )
        return [self.serialize(corpus) for corpus in rows.all()]

    async def get_active_corpus(self, *, corpus_id: str, workspace_id: str) -> Corpus:
        corpus = await self.session.scalar(
            select(Corpus)
            .options(selectinload(Corpus.sources))
            .where(Corpus.id == corpus_id, Corpus.workspace_id == workspace_id)
        )
        if corpus is None:
            raise api_error(404, "corpus_not_found", "Corpus not found.")
        if corpus.status != "active":
            raise api_error(404, "corpus_not_found", "Corpus not found.")
        if _is_expired(corpus.expires_at):
            corpus.status = "expired"
            await self.session.commit()
            invalidate_workspace_cache(workspace_id)
            raise api_error(410, "corpus_expired", "Corpus has expired.")
        return corpus

    async def source_ids_for_search(self, *, corpus_id: str, workspace_id: str) -> list[str]:
        corpus = await self.get_active_corpus(corpus_id=corpus_id, workspace_id=workspace_id)
        return [source.id for source in corpus.sources]

    async def delete_corpus(self, *, corpus_id: str, workspace_id: str) -> bool:
        corpus = await self.session.scalar(
            select(Corpus).where(Corpus.id == corpus_id, Corpus.workspace_id == workspace_id)
        )
        if corpus is None:
            return False
        await self._delete_corpus_rows(corpus_id)
        await self.session.delete(corpus)
        await self.session.commit()
        invalidate_workspace_cache(workspace_id)
        return True

    async def refresh_counts(self, *, corpus_id: str, workspace_id: str) -> dict:
        corpus = await self.get_active_corpus(corpus_id=corpus_id, workspace_id=workspace_id)
        rows = (
            await self.session.scalars(
                select(Source)
                .options(selectinload(Source.documents))
                .where(Source.corpus_id == corpus_id, Source.workspace_id == workspace_id)
            )
        ).all()
        corpus.source_count = len(rows)
        corpus.document_count = sum(len(source.documents) for source in rows)
        corpus.byte_count = sum(
            int((source.source_metadata or {}).get("byte_count", 0)) for source in rows
        )
        corpus.window_count = sum(
            int((source.source_metadata or {}).get("window_count", len(source.documents)))
            for source in rows
        )
        await self.session.commit()
        await self.session.refresh(corpus)
        invalidate_workspace_cache(workspace_id)
        return self.serialize(corpus)

    async def expire_due_corpora(self, *, workspace_id: str | None = None) -> int:
        query = select(Corpus).where(
            Corpus.status == "active",
            Corpus.expires_at.is_not(None),
            Corpus.expires_at <= _now(),
        )
        if workspace_id is not None:
            query = query.where(Corpus.workspace_id == workspace_id)
        rows = (await self.session.scalars(query)).all()
        for corpus in rows:
            corpus.status = "expired"
            invalidate_workspace_cache(corpus.workspace_id)
        if rows:
            await self.session.commit()
        return len(rows)

    async def delete_expired_corpora(self) -> int:
        expired_ids = (
            await self.session.scalars(
                select(Corpus.id).where(
                    Corpus.status == "expired",
                )
            )
        ).all()
        if not expired_ids:
            return 0
        for corpus_id in expired_ids:
            await self._delete_corpus_rows(corpus_id)
        await self.session.execute(delete(Corpus).where(Corpus.id.in_(expired_ids)))
        await self.session.commit()
        return len(expired_ids)

    async def search_count(self, *, corpus_id: str) -> int:
        return int(
            await self.session.scalar(
                select(func.count())
                .select_from(Document)
                .join(Source)
                .where(Source.corpus_id == corpus_id)
            )
            or 0
        )

    async def _delete_corpus_rows(self, corpus_id: str) -> None:
        source_ids = (
            await self.session.scalars(select(Source.id).where(Source.corpus_id == corpus_id))
        ).all()
        if source_ids:
            await self.session.execute(delete(Document).where(Document.source_id.in_(source_ids)))
            await self.session.execute(delete(Source).where(Source.id.in_(source_ids)))

    @staticmethod
    def serialize(corpus: Corpus) -> dict:
        return {
            "id": corpus.id,
            "workspace_id": corpus.workspace_id,
            "name": corpus.name,
            "retention": corpus.retention,
            "status": corpus.status,
            "ttl_seconds": corpus.ttl_seconds,
            "expires_at": corpus.expires_at.isoformat() if corpus.expires_at else None,
            "source_count": corpus.source_count,
            "document_count": corpus.document_count,
            "byte_count": corpus.byte_count,
            "window_count": corpus.window_count,
            "created_at": corpus.created_at.isoformat() if corpus.created_at else None,
        }
