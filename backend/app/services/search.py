from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.answer.citations import attach_citation_labels
from app.models import Source
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.pipeline import RetrievalPipeline, SearchDocument

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — shared across all requests in the process.
# ---------------------------------------------------------------------------

_bm25_cache: BM25IndexCache | None = None
_pipeline: RetrievalPipeline | None = None


def _get_pipeline() -> RetrievalPipeline:
    global _pipeline, _bm25_cache
    if _pipeline is None:
        from app.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        _bm25_cache = BM25IndexCache(
            ttl=settings.bm25_cache_ttl,
            max_entries=settings.bm25_cache_max_entries,
        )
        _pipeline = RetrievalPipeline(
            bm25_cache=_bm25_cache,
            confidence_threshold=settings.confidence_threshold,
        )
    return _pipeline


def invalidate_workspace_cache(workspace_id: str) -> None:
    """Call after source upload/delete to clear the BM25 index for this workspace."""
    if _bm25_cache is not None:
        _bm25_cache.invalidate(workspace_id)


# ---------------------------------------------------------------------------
# SearchService
# ---------------------------------------------------------------------------

class SearchService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.pipeline = _get_pipeline()

    async def search(
        self,
        query: str,
        top_k: int,
        source_ids: list[str] | None,
        workspace_id: str,
        connector_configs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        documents = await self._load_documents(
            workspace_id=workspace_id,
            source_ids=source_ids,
            connector_configs=connector_configs,
        )
        raw_results = self.pipeline.search(
            query=query, documents=documents, top_k=top_k, workspace_id=workspace_id
        )
        results = attach_citation_labels([self._serialize_result(r) for r in raw_results])
        return {
            "query": query,
            "results": results,
            "sources": self._unique_sources(results),
        }

    async def stream_search(
        self,
        query: str,
        top_k: int,
        source_ids: list[str] | None,
        workspace_id: str,
        connector_configs: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Async generator yielding SSE-formatted data lines with search results."""
        result = await self.search(
            query=query,
            top_k=top_k,
            source_ids=source_ids,
            workspace_id=workspace_id,
            connector_configs=connector_configs,
        )
        yield f"data: {json.dumps(result)}\n\n"

    # --- internals -----------------------------------------------------------

    async def _load_documents(
        self,
        workspace_id: str,
        source_ids: list[str] | None,
        connector_configs: list[dict[str, Any]] | None = None,
    ) -> list[SearchDocument]:
        db_docs = await self._load_db_documents(workspace_id, source_ids)
        connector_docs = await self._fetch_connector_documents(connector_configs or [])
        return db_docs + connector_docs

    async def _load_db_documents(
        self, workspace_id: str, source_ids: list[str] | None
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

    async def _fetch_connector_documents(
        self, connector_configs: list[dict[str, Any]]
    ) -> list[SearchDocument]:
        from app.config import get_settings  # noqa: PLC0415
        from app.connectors.registry import default_registry  # noqa: PLC0415

        settings = get_settings()
        if not settings.enable_live_connectors or not connector_configs:
            return []

        async def _fetch_one(cfg: dict[str, Any]) -> list[SearchDocument]:
            connector_id = cfg.get("connector_id", "")
            connector = default_registry.get(connector_id)
            if connector is None:
                _log.warning("Unknown connector_id %r — skipping", connector_id)
                return []
            try:
                cfg_with_limit = {**cfg, "max_documents": settings.connector_max_documents}
                return await asyncio.wait_for(
                    connector.fetch_as_search_documents(cfg_with_limit),
                    timeout=settings.connector_fetch_timeout,
                )
            except asyncio.TimeoutError:
                _log.warning("Connector %r timed out — skipping", connector_id)
                return []
            except Exception as exc:
                _log.warning("Connector %r failed: %s — skipping", connector_id, exc)
                return []

        results = await asyncio.gather(*(_fetch_one(cfg) for cfg in connector_configs))
        return [doc for batch in results for doc in batch]

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
