from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.answer.citations import attach_citation_labels
from app.config import Settings
from app.models import Source
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.retriever import Bm25Retriever, MultiHeadRetriever, Retriever
from app.retrieval.span_executor import SpanExecutor
from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    ProjectionConfig,
    SparseProjection,
    load_projection,
)
from app.schemas.search import SearchDocument

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — shared across all requests in the process.
# ---------------------------------------------------------------------------

_bm25_cache: BM25IndexCache | None = None
_pipeline: RetrievalPipeline | None = None
_projection: SparseProjection | DeterministicSemanticProjection | None = None
_SECRET_MARKERS = ("token", "secret", "password", "key", "credential")


def _build_retriever(settings: Settings, bm25_cache: BM25IndexCache) -> Retriever:
    if settings.waver_retriever == "bm25":
        return Bm25Retriever(
            cache=bm25_cache,
            use_stemming=settings.bm25_use_stemming,
            use_stopwords=settings.bm25_use_stopwords,
        )
    if settings.waver_retriever == "sps":
        from app.retrieval.exact_phrase import ExactPhraseRetriever  # noqa: PLC0415
        from app.retrieval.sps import SpsRetriever  # noqa: PLC0415

        projection_config = ProjectionConfig(
            hash_features=settings.projection_hash_features,
            dim=settings.projection_dim,
        )
        sps = SpsRetriever(
            cache=bm25_cache,
            projection=load_projection(settings.projection_model_path, projection_config),
            alpha=settings.waver_sps_alpha,
            candidate_multiplier=settings.waver_sps_candidate_multiplier,
            use_stemming=settings.bm25_use_stemming,
            use_stopwords=settings.bm25_use_stopwords,
        )
        if not settings.waver_multihead:
            return sps
        bm25 = Bm25Retriever(
            cache=bm25_cache,
            use_stemming=settings.bm25_use_stemming,
            use_stopwords=settings.bm25_use_stopwords,
        )
        return MultiHeadRetriever(
            heads=[
                ("sps", sps),
                ("bm25", bm25),
                ("phrase", ExactPhraseRetriever()),
            ],
            rrf_k=settings.waver_rrf_k,
        )
    if settings.waver_retriever == "cortical":
        from app.retrieval.cortical import CorticalRetriever, default_trie_builder  # noqa: PLC0415
        from app.retrieval.diffusion import DiffusionConfig  # noqa: PLC0415

        projection_config = ProjectionConfig(
            hash_features=settings.projection_hash_features,
            dim=settings.projection_dim,
        )
        return CorticalRetriever(
            projection=load_projection(settings.projection_model_path, projection_config),
            trie_builder=default_trie_builder(
                max_patterns=settings.waver_trie_max_patterns,
                phrase_ngram=settings.waver_trie_phrase_ngram,
                phrase_weight=settings.waver_trie_phrase_weight,
            ),
            gating_m=settings.waver_gating_m,
            lambdas=(
                settings.waver_lambda_l,
                settings.waver_lambda_s,
                settings.waver_lambda_c,
                settings.waver_lambda_cost,
            ),
            diffusion=DiffusionConfig(
                steps=settings.waver_diffusion_steps,
                beta=settings.waver_diffusion_beta,
                gamma=settings.waver_diffusion_gamma,
                delta=settings.waver_diffusion_delta,
            ),
            candidate_cap=settings.waver_candidate_cap,
            adjacency_max_edges=settings.waver_adjacency_max_edges,
        )
    raise ValueError(f"Unsupported retriever: {settings.waver_retriever}")


def _pipeline_candidate_budget(settings: Settings) -> int:
    if settings.waver_retriever == "cortical":
        return settings.waver_gating_m
    return 100


def _get_pipeline() -> RetrievalPipeline:
    global _pipeline, _bm25_cache
    if _pipeline is None:
        from app.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        _bm25_cache = BM25IndexCache(
            ttl=settings.bm25_cache_ttl,
            max_entries=settings.bm25_cache_max_entries,
            use_stemming=settings.bm25_use_stemming,
            use_stopwords=settings.bm25_use_stopwords,
        )
        _pipeline = RetrievalPipeline(
            bm25_cache=_bm25_cache,
            retriever=_build_retriever(settings, _bm25_cache),
            bm25_candidates=_pipeline_candidate_budget(settings),
            confidence_threshold=settings.confidence_threshold,
            projection=_get_projection(),
        )
    return _pipeline


def _get_projection() -> SparseProjection | DeterministicSemanticProjection:
    global _projection
    if _projection is None:
        from app.config import get_settings  # noqa: PLC0415

        settings = get_settings()
        config = ProjectionConfig(
            hash_features=settings.projection_hash_features,
            dim=settings.projection_dim,
        )
        _projection = load_projection(settings.projection_model_path, config)
    return _projection


def _build_span_executor(pipeline: RetrievalPipeline) -> SpanExecutor:
    return SpanExecutor(
        parser_detector=pipeline.parser_detector,
        reranker=pipeline.reranker,
        projection=_get_projection(),
    )


def invalidate_workspace_cache(workspace_id: str) -> None:
    """Call after source upload/delete to clear the BM25 index for this workspace."""
    if _bm25_cache is not None:
        _bm25_cache.invalidate(workspace_id)


@dataclass(slots=True)
class LoadedSources:
    documents: list[SearchDocument]
    errors: list[dict[str, str | None]]
    cacheable: bool


def _contains_secret_marker(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in _SECRET_MARKERS)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _sanitize_metadata(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if _contains_secret_marker(str(key)):
            continue
        clean[key] = _sanitize_value(value)
    return clean


def _source_error(
    source_type: str,
    message: str,
    *,
    connector_id: str | None = None,
    source_name: str | None = None,
) -> dict[str, str | None]:
    return {
        "source_type": source_type,
        "connector_id": connector_id,
        "source_name": source_name,
        "message": message,
    }


def _is_empty_raw_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, bytes):
        return not content.strip()
    return False


def _normalize_raw_content(content: Any, media_type: str | None) -> tuple[bytes | str, str]:
    if isinstance(content, bytes):
        return content, media_type or "application/octet-stream"
    if isinstance(content, str):
        return content, media_type or "text/plain"
    if isinstance(content, (dict, list)):
        return (
            json.dumps(content, ensure_ascii=False, separators=(",", ":")),
            media_type or "application/json",
        )
    return str(content), media_type or "text/plain"


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
        raw_sources: list[dict[str, Any]] | None = None,
        connector_configs: list[dict[str, Any]] | None = None,
        include_stored_sources: bool = True,
        answer_mode: str = "off",
        budget_hint: str = "auto",
    ) -> dict[str, Any]:
        loaded = await self._load_documents(
            workspace_id=workspace_id,
            source_ids=source_ids,
            raw_sources=raw_sources or [],
            connector_configs=connector_configs,
            include_stored_sources=include_stored_sources,
        )
        executor = _build_span_executor(self.pipeline)
        outcome = executor.execute(
            query=query,
            documents=loaded.documents,
            top_k=top_k,
            cacheable=loaded.cacheable,
            budget_hint=budget_hint,
        )
        results = attach_citation_labels([self._serialize_result(r) for r in outcome.final_results])
        answer = None
        answer_error = None
        if answer_mode == "llm":
            from app.answer.generator import AnswerGenerator  # noqa: PLC0415

            answer, answer_error = await AnswerGenerator().generate(query, results)
        return {
            "query": query,
            "answer": answer,
            "answer_error": answer_error,
            "results": results,
            "sources": self._unique_sources(results),
            "source_errors": loaded.errors,
            "execution": outcome.execution,
        }

    async def stream_search(
        self,
        query: str,
        top_k: int,
        source_ids: list[str] | None,
        workspace_id: str,
        raw_sources: list[dict[str, Any]] | None = None,
        connector_configs: list[dict[str, Any]] | None = None,
        include_stored_sources: bool = True,
        answer_mode: str = "off",
        budget_hint: str = "auto",
    ) -> AsyncIterator[str]:
        """Async generator yielding SSE-formatted data lines with search results."""
        loaded = await self._load_documents(
            workspace_id=workspace_id,
            source_ids=source_ids,
            raw_sources=raw_sources or [],
            connector_configs=connector_configs,
            include_stored_sources=include_stored_sources,
        )
        executor = _build_span_executor(self.pipeline)
        try:
            outcome = executor.execute(
                query=query,
                documents=loaded.documents,
                top_k=top_k,
                cacheable=loaded.cacheable,
                budget_hint=budget_hint,
            )
            sources_loaded_payload = {
                "event": "sources_loaded",
                "query": query,
                "source_errors": loaded.errors,
                "execution": outcome.execution,
            }
            yield f"data: {json.dumps(sources_loaded_payload)}\n\n"

            def _phase_payload(event: str, results_list: list[Any]) -> str:
                serialized = attach_citation_labels(
                    [self._serialize_result(item) for item in results_list]
                )
                payload = {
                    "event": event,
                    "query": query,
                    "results": serialized,
                    "source_errors": loaded.errors,
                    "execution": outcome.execution,
                }
                return f"data: {json.dumps(payload)}\n\n"

            if outcome.exact_results:
                yield _phase_payload("exact_results", outcome.exact_results)
            if outcome.proxy_results:
                yield _phase_payload("proxy_results", outcome.proxy_results)
            if outcome.reranked_results:
                yield _phase_payload("reranked_results", outcome.reranked_results)

            final_results = attach_citation_labels(
                [self._serialize_result(item) for item in outcome.final_results]
            )
            answer = None
            answer_error = None
            if answer_mode == "llm":
                from app.answer.generator import AnswerGenerator  # noqa: PLC0415

                generator = AnswerGenerator()
                streamed_answer = ""
                try:
                    async for token in generator.stream_generate(query, final_results):
                        streamed_answer += token
                        answer_delta_payload = {
                            "event": "answer_delta",
                            "query": query,
                            "delta": token,
                        }
                        yield f"data: {json.dumps(answer_delta_payload)}\n\n"
                except Exception as exc:  # pragma: no cover - network errors
                    _log.warning("Answer streaming failed: %s", exc)
                if streamed_answer.strip():
                    confidence = "low" if not generator.settings.openrouter_api_key else "medium"
                    answer = {"markdown": streamed_answer.strip(), "confidence": confidence}
                else:
                    answer, answer_error = await generator.generate(query, final_results)

            done_payload = {
                "event": "done",
                "query": query,
                "answer": answer,
                "answer_error": answer_error,
                "results": final_results,
                "sources": self._unique_sources(final_results),
                "source_errors": loaded.errors,
                "execution": outcome.execution,
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
        except Exception as exc:  # pragma: no cover - defensive SSE envelope
            _log.exception("Stream search failed")
            payload = {
                "event": "error",
                "query": query,
                "message": str(exc),
            }
            yield f"data: {json.dumps(payload)}\n\n"

    # --- internals -----------------------------------------------------------

    async def _load_documents(
        self,
        workspace_id: str,
        source_ids: list[str] | None,
        raw_sources: list[dict[str, Any]],
        connector_configs: list[dict[str, Any]] | None = None,
        include_stored_sources: bool = True,
    ) -> LoadedSources:
        documents: list[SearchDocument] = []
        errors: list[dict[str, str | None]] = []

        if include_stored_sources:
            documents.extend(await self._load_db_documents(workspace_id, source_ids))

        raw_docs, raw_errors = self._load_raw_documents(raw_sources)
        connector_docs, connector_errors = await self._fetch_connector_documents(
            connector_configs or []
        )
        documents.extend(raw_docs)
        documents.extend(connector_docs)
        errors.extend(raw_errors)
        errors.extend(connector_errors)

        return LoadedSources(
            documents=documents,
            errors=errors,
            cacheable=include_stored_sources and not raw_sources and not connector_configs,
        )

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
                        "source_origin": "stored",
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

    def _load_raw_documents(
        self,
        raw_sources: list[dict[str, Any]],
    ) -> tuple[list[SearchDocument], list[dict[str, str | None]]]:
        documents: list[SearchDocument] = []
        errors: list[dict[str, str | None]] = []
        for index, raw_source in enumerate(raw_sources):
            name = str(raw_source.get("name") or f"Raw source {index + 1}")
            content = raw_source.get("content")
            if _is_empty_raw_content(content):
                errors.append(_source_error("raw", "Raw source has no content", source_name=name))
                continue

            source_id = str(raw_source.get("id") or f"raw:{index}:{name}")
            source_type = str(raw_source.get("source_type") or "raw")
            metadata = _sanitize_metadata(dict(raw_source.get("metadata") or {}))
            metadata.update(
                {
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_origin": "raw",
                    "title": metadata.get("title", name),
                }
            )
            normalized_content, media_type = _normalize_raw_content(
                content,
                raw_source.get("media_type"),
            )
            documents.append(
                SearchDocument(
                    file_name=name,
                    content=normalized_content,
                    media_type=media_type,
                    metadata=metadata,
                )
            )
        return documents, errors

    async def _fetch_connector_documents(
        self, connector_configs: list[dict[str, Any]]
    ) -> tuple[list[SearchDocument], list[dict[str, str | None]]]:
        from app.config import get_settings  # noqa: PLC0415
        from app.connectors.registry import default_registry  # noqa: PLC0415

        settings = get_settings()
        if not connector_configs:
            return [], []

        async def _fetch_one(
            cfg: dict[str, Any],
        ) -> tuple[list[SearchDocument], list[dict[str, str | None]]]:
            connector_id = cfg.get("connector_id", "")
            connector = default_registry.get(connector_id)
            if connector is None:
                _log.warning("Unknown connector_id %r — skipping", connector_id)
                return [], [
                    _source_error(
                        "connector",
                        "Unknown connector",
                        connector_id=str(connector_id) if connector_id else None,
                    )
                ]
            if connector_id != "webhook" and not settings.enable_live_connectors:
                _log.warning("Connector %r disabled — skipping", connector_id)
                return [], [
                    _source_error(
                        "connector",
                        "Live connector is disabled",
                        connector_id=str(connector_id),
                    )
                ]
            try:
                cfg_with_limit = {**cfg, "max_documents": settings.connector_max_documents}
                docs = await asyncio.wait_for(
                    connector.fetch_as_search_documents(cfg_with_limit),
                    timeout=settings.connector_fetch_timeout,
                )
                return self._mark_connector_documents(docs, str(connector_id)), []
            except asyncio.TimeoutError:
                _log.warning("Connector %r timed out — skipping", connector_id)
                return [], [
                    _source_error(
                        "connector",
                        "Connector timed out",
                        connector_id=str(connector_id),
                    )
                ]
            except Exception as exc:
                _log.warning("Connector %r failed: %s — skipping", connector_id, exc)
                return [], [
                    _source_error(
                        "connector",
                        "Connector failed",
                        connector_id=str(connector_id),
                    )
                ]

        results = await asyncio.gather(*(_fetch_one(cfg) for cfg in connector_configs))
        documents: list[SearchDocument] = []
        errors: list[dict[str, str | None]] = []
        for doc_batch, error_batch in results:
            documents.extend(doc_batch)
            errors.extend(error_batch)
        return documents, errors

    def _mark_connector_documents(
        self, documents: list[SearchDocument], connector_id: str
    ) -> list[SearchDocument]:
        marked: list[SearchDocument] = []
        for index, document in enumerate(documents):
            metadata = _sanitize_metadata(dict(document.metadata))
            metadata.update(
                {
                    "source_id": metadata.get(
                        "source_id",
                        f"connector:{connector_id}:{index}:{document.file_name}",
                    ),
                    "source_type": metadata.get("source_type", connector_id),
                    "source_origin": "connector",
                    "connector_id": connector_id,
                    "title": metadata.get("title", document.file_name),
                }
            )
            marked.append(
                SearchDocument(
                    file_name=document.file_name,
                    content=document.content,
                    media_type=document.media_type,
                    metadata=metadata,
                )
            )
        return marked

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
            "spans": result.spans,
            "primary_span": result.primary_span,
            "matched_spans": result.matched_spans,
            "channels": result.channels,
            "channel_scores": result.channel_scores,
            "source_origin": metadata.get("source_origin", "stored"),
        }

    def _unique_sources(self, results: list[dict[str, Any]]) -> list[dict[str, str]]:
        unique: dict[str, dict[str, str]] = {}
        for result in results:
            source_id = result.get("source_id")
            if source_id:
                unique[source_id] = {"id": source_id, "name": result["source_name"]}
        return list(unique.values())
