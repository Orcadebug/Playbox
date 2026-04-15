from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Document, Source
from app.parsers.base import ParsedDocument
from app.parsers.detector import detect_and_parse


class SourceService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_from_bytes(
        self,
        file_name: str,
        content: bytes,
        media_type: str | None,
        workspace_id: str,
    ) -> dict[str, Any]:
        parsed = detect_and_parse(file_name=file_name, content=content, media_type=media_type)
        return await self._persist_parsed_file(
            name=file_name,
            media_type=media_type,
            parser_name=parsed.parser_name,
            workspace_id=workspace_id,
            documents=parsed.documents,
            source_metadata={"document_count": len(parsed.documents)},
        )

    async def create_from_text(self, name: str, text: str, workspace_id: str) -> dict[str, Any]:
        parsed = ParsedDocument(content=text, source_name=name, metadata={"origin": "paste"})
        return await self._persist_parsed_file(
            name=name,
            media_type="text/plain",
            parser_name="plaintext",
            workspace_id=workspace_id,
            documents=[parsed],
            source_metadata={"document_count": 1, "origin": "paste"},
        )

    async def list_sources(self, workspace_id: str) -> list[dict[str, Any]]:
        query = (
            select(Source)
            .options(selectinload(Source.documents))
            .where(Source.workspace_id == workspace_id)
            .order_by(Source.created_at.desc())
        )
        rows = await self.session.scalars(query)
        return [self._serialize_source(source) for source in rows.all()]

    async def delete_source(self, source_id: str) -> bool:
        statement = delete(Source).where(Source.id == source_id)
        result = await self.session.execute(statement)
        await self.session.commit()
        return bool(result.rowcount)

    async def _persist_parsed_file(
        self,
        name: str,
        media_type: str | None,
        parser_name: str,
        workspace_id: str,
        documents: list[ParsedDocument],
        source_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        source = Source(
            workspace_id=workspace_id,
            source_type="upload",
            name=name,
            media_type=media_type,
            parser_name=parser_name,
            source_metadata=source_metadata,
        )
        self.session.add(source)
        await self.session.flush()

        for index, document in enumerate(documents):
            title = document.metadata.get("title") if isinstance(document.metadata.get("title"), str) else None
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
        return self._serialize_source(source, document_count=len(documents))

    def _serialize_source(self, source: Source, document_count: int | None = None) -> dict[str, Any]:
        count = document_count
        if count is None:
            count = int(source.source_metadata.get("document_count", 0)) if source.source_metadata else 0
        return {
            "id": source.id,
            "workspace_id": source.workspace_id,
            "name": source.name,
            "source_type": source.source_type,
            "media_type": source.media_type,
            "parser_name": source.parser_name,
            "metadata": source.source_metadata,
            "document_count": count,
            "created_at": source.created_at.isoformat() if source.created_at else None,
        }
