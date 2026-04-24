from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Corpus, Document, Source
from app.parsers.base import ParsedDocument
from app.parsers.detector import detect_and_parse
from app.services.search import invalidate_workspace_cache


def _active_source_clause(now: datetime):
    return or_(
        Source.corpus_id.is_(None),
        and_(
            Corpus.status == "active",
            or_(Corpus.expires_at.is_(None), Corpus.expires_at > now),
        ),
    )


def _estimate_window_count(documents: list[ParsedDocument], parser_name: str) -> int:
    if parser_name == "plaintext":
        return sum(max(1, len(document.content.splitlines())) for document in documents)
    return sum(1 for document in documents if document.content.strip())


class SourceService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_from_bytes(
        self,
        file_name: str,
        content: bytes,
        media_type: str | None,
        workspace_id: str,
        corpus_id: str | None = None,
    ) -> dict[str, Any]:
        parsed = detect_and_parse(file_name=file_name, content=content, media_type=media_type)
        window_count = _estimate_window_count(parsed.documents, parsed.parser_name)
        return await self._persist_parsed_file(
            name=file_name,
            media_type=media_type,
            parser_name=parsed.parser_name,
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            source_type="session" if corpus_id else "upload",
            documents=parsed.documents,
            source_metadata={
                "document_count": len(parsed.documents),
                "byte_count": len(content),
                "window_count": window_count,
            },
        )

    async def create_from_text(
        self,
        name: str,
        text: str,
        workspace_id: str,
        corpus_id: str | None = None,
    ) -> dict[str, Any]:
        parsed = ParsedDocument(content=text, source_name=name, metadata={"origin": "paste"})
        documents = [parsed]
        return await self._persist_parsed_file(
            name=name,
            media_type="text/plain",
            parser_name="plaintext",
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            source_type="session" if corpus_id else "upload",
            documents=documents,
            source_metadata={
                "document_count": 1,
                "origin": "paste",
                "byte_count": len(text.encode("utf-8")),
                "window_count": _estimate_window_count(documents, "plaintext"),
            },
        )

    async def list_sources(self, workspace_id: str) -> list[dict[str, Any]]:
        now = datetime.now(UTC)
        query = (
            select(Source)
            .outerjoin(Corpus, Source.corpus_id == Corpus.id)
            .options(selectinload(Source.documents))
            .where(Source.workspace_id == workspace_id, _active_source_clause(now))
            .order_by(Source.created_at.desc())
        )
        rows = await self.session.scalars(query)
        return [self._serialize_source(source) for source in rows.all()]

    async def delete_source(self, source_id: str, workspace_id: str = "default") -> bool:
        source = await self.session.scalar(
            select(Source).where(Source.id == source_id, Source.workspace_id == workspace_id)
        )
        if source is None:
            return False
        await self.session.execute(delete(Document).where(Document.source_id == source.id))
        await self.session.delete(source)
        await self.session.commit()
        invalidate_workspace_cache(workspace_id)
        return True

    async def _persist_parsed_file(
        self,
        name: str,
        media_type: str | None,
        parser_name: str,
        workspace_id: str,
        corpus_id: str | None,
        source_type: str,
        documents: list[ParsedDocument],
        source_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        source = Source(
            workspace_id=workspace_id,
            corpus_id=corpus_id,
            source_type=source_type,
            name=name,
            media_type=media_type,
            parser_name=parser_name,
            source_metadata=source_metadata,
        )
        self.session.add(source)
        await self.session.flush()

        for index, document in enumerate(documents):
            raw_title = document.metadata.get("title")
            title = raw_title if isinstance(raw_title, str) else None
            self.session.add(
                Document(
                    source_id=source.id,
                    title=title,
                    content=document.content,
                    order_index=index,
                    document_metadata=document.metadata,
                )
            )

        await self.session.commit()
        await self.session.refresh(source)
        invalidate_workspace_cache(workspace_id)
        return self._serialize_source(source, document_count=len(documents))

    def _serialize_source(
        self, source: Source, document_count: int | None = None
    ) -> dict[str, Any]:
        count = document_count
        if count is None:
            meta = source.source_metadata
            count = int(meta.get("document_count", 0)) if meta else 0
        return {
            "id": source.id,
            "workspace_id": source.workspace_id,
            "corpus_id": source.corpus_id,
            "name": source.name,
            "source_type": source.source_type,
            "media_type": source.media_type,
            "parser_name": source.parser_name,
            "metadata": source.source_metadata,
            "document_count": count,
            "byte_count": int((source.source_metadata or {}).get("byte_count", 0)),
            "window_count": int((source.source_metadata or {}).get("window_count", count)),
            "created_at": source.created_at.isoformat() if source.created_at else None,
        }
