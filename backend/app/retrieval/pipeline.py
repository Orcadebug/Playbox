from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.parsers.base import (
    ParsedDocument,
    ParsedFile,
    ParserDetector,
    build_default_parser_registry,
)
from app.parsers.plaintext import PlainTextParser
from app.retrieval.bm25 import BM25Index, BM25ScoredChunk
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.chunker import chunk_documents, chunk_documents_iter
from app.retrieval.reranker import AutoReranker, Reranker
from app.schemas.search import SearchDocument, SearchResult

_log = logging.getLogger(__name__)


def _snippet(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "…"


def _make_default_reranker() -> AutoReranker:
    """Build AutoReranker wired to the configured model path."""
    from app.config import get_settings  # noqa: PLC0415 — deferred to avoid circular import

    settings = get_settings()
    model_name = settings.reranker_model.replace("/", "-")
    model_file = "model_quantized.onnx" if settings.reranker_quantized else "model.onnx"
    model_path = Path(settings.model_dir) / model_name / model_file
    return AutoReranker(
        model_path=model_path,
        batch_size=settings.reranker_batch_size,
        max_length=settings.reranker_max_length,
        max_candidates=settings.reranker_max_candidates,
    )


@dataclass(slots=True)
class RetrievalPipeline:
    parser_detector: ParserDetector = field(
        default_factory=lambda: ParserDetector(
            registry=build_default_parser_registry(),
            default_parser=PlainTextParser(),
        )
    )
    reranker: Reranker = field(default_factory=_make_default_reranker)
    chunk_size: int = 500
    chunk_overlap: int = 50
    bm25_candidates: int = 100
    bm25_cache: BM25IndexCache | None = None
    confidence_threshold: float = 0.85

    def search(
        self,
        query: str,
        documents: Sequence[SearchDocument],
        top_k: int = 10,
        workspace_id: str = "default",
    ) -> list[SearchResult]:
        parsed_documents = self._parse_documents(documents)

        # For large corpora use the generator chunker to avoid peak memory spike.
        if len(parsed_documents) > 50:
            chunks = list(chunk_documents_iter(
                parsed_documents, max_tokens=self.chunk_size, overlap=self.chunk_overlap
            ))
        else:
            chunks = chunk_documents(
                parsed_documents, max_tokens=self.chunk_size, overlap=self.chunk_overlap
            )

        if not chunks:
            return []

        # BM25 first stage — use cache when available.
        if self.bm25_cache is not None:
            index = self.bm25_cache.get_or_build(workspace_id, chunks)
        else:
            index = BM25Index(chunks)

        bm25_hits = index.search(query, top_k=self.bm25_candidates)
        reranked = self.reranker.rerank(query, bm25_hits, top_k=top_k)
        results = [self._to_search_result(hit) for hit in reranked[:top_k]]

        # Early exit: if top result has very high confidence and a wide gap to #2,
        # return immediately rather than continuing any downstream processing.
        if (
            len(results) >= 2
            and results[0].score is not None
            and results[0].score > self.confidence_threshold
            and results[0].score - results[1].score > 0.30
        ):
            _log.debug(
                "Early exit: top score=%.3f gap=%.3f",
                results[0].score,
                results[0].score - results[1].score,
            )

        return results

    def _parse_documents(self, documents: Sequence[SearchDocument]) -> list[ParsedDocument]:
        parsed_documents: list[ParsedDocument] = []
        for document in documents:
            parsed_file = self.parser_detector.parse(
                file_name=document.file_name,
                content=document.data,
                media_type=document.media_type,
            )
            parsed_documents.extend(self._merge_metadata(parsed_file, document.metadata))
        return parsed_documents

    def _merge_metadata(
        self, parsed_file: ParsedFile, metadata: dict[str, object]
    ) -> list[ParsedDocument]:
        merged: list[ParsedDocument] = []
        for parsed_document in parsed_file.documents:
            merged_metadata = dict(metadata)
            merged_metadata.update(parsed_document.metadata)
            merged_metadata.setdefault("parser_name", parsed_file.parser_name)
            if parsed_file.media_type is not None:
                merged_metadata.setdefault("media_type", parsed_file.media_type)
            merged.append(
                ParsedDocument(
                    content=parsed_document.content,
                    source_name=parsed_document.source_name,
                    metadata=merged_metadata,
                    location=parsed_document.location,
                )
            )
        return merged

    def _to_search_result(self, hit: BM25ScoredChunk) -> SearchResult:
        metadata = dict(hit.chunk.metadata)
        return SearchResult(
            content=hit.chunk.text,
            snippet=_snippet(hit.chunk.text),
            score=hit.rerank_score if hit.rerank_score is not None else hit.score,
            source_name=hit.chunk.source_name,
            metadata=metadata,
            chunk_id=hit.chunk.chunk_id,
            parser_name=(
                metadata.get("parser_name")
                if isinstance(metadata.get("parser_name"), str)
                else None
            ),
            bm25_score=hit.bm25_score if hit.bm25_score is not None else hit.score,
            rerank_score=hit.rerank_score if hit.rerank_score is not None else hit.score,
        )
