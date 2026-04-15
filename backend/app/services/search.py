from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.answer.citations import attach_citation_labels
from app.answer.generator import AnswerGenerator
from app.models import Source
from app.retrieval.pipeline import RetrievalPipeline, SearchDocument


class SearchService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.pipeline = RetrievalPipeline()
        self.answer_generator = AnswerGenerator()

    async def search(
        self,
        query: str,
        top_k: int,
        source_ids: list[str] | None,
        workspace_id: str,
        skip_answer: bool,
    ) -> dict[str, Any]:
        documents = await self._load_documents(workspace_id=workspace_id, source_ids=source_ids)
        raw_results = self.pipeline.search(query=query, documents=documents, top_k=top_k)
        results = attach_citation_labels([self._serialize_result(result) for result in raw_results])

        answer = None
        answer_error = None
        if not skip_answer:
            answer, answer_error = await self.answer_generator.generate(query=query, results=results)

        return {
            "query": query,
            "results": results,
            "answer": answer,
            "answer_error": answer_error,
            "sources": self._unique_sources(results),
        }

    async def _load_documents(
        self,
        workspace_id: str,
        source_ids: list[str] | None,
    ) -> list[SearchDocument]:
        query = (
            select(Source)
            .options(selectinload(Source.documents))
            .where(Source.workspace_id == workspace_id)
            .order_by(Source.created_at.desc())
        )
        if source_ids:
            query = query.where(Source.id.in_(source_ids))

        rows = await self.session.scalars(query)
        search_docs: list[SearchDocument] = []
        for source in rows.all():
            for document in sorted(source.documents, key=lambda doc: doc.order_index):
                metadata = dict(document.document_metadata or {})
                metadata.update(
                    {
                        "document_id": document.id,
                        "source_id": source.id,
                        "title": document.title or source.name,
                        "source_type": source.source_type,
                    }
                )
                search_docs.append(
                    SearchDocument(
                        file_name=source.name,
                        content=document.content,
                        media_type=source.media_type,
                        metadata=metadata,
                    )
                )
        return search_docs

    def _serialize_result(self, result: Any) -> dict[str, Any]:
        metadata = dict(result.metadata or {})
        return {
            "document_id": metadata.get("document_id"),
            "source_id": metadata.get("source_id"),
            "source_name": result.source_name,
            "title": metadata.get("title", result.source_name),
            "content": result.content,
            "snippet": result.snippet,
            "score": round(float(result.score), 4),
            "metadata": metadata,
            "chunk_id": result.chunk_id,
            "parser_name": result.parser_name,
            "bm25_score": result.bm25_score,
            "rerank_score": result.rerank_score,
        }

    def _unique_sources(self, results: list[dict[str, Any]]) -> list[dict[str, str]]:
        unique: dict[str, dict[str, str]] = {}
        for result in results:
            source_id = result.get("source_id")
            if source_id:
                unique[source_id] = {"id": source_id, "name": result["source_name"]}
        return list(unique.values())
