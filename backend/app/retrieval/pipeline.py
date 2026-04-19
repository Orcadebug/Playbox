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
from app.retrieval.bm25 import BM25ScoredChunk
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.chunker import chunk_documents, chunk_documents_iter
from app.retrieval.query_patterns import build_query_patterns
from app.retrieval.reranker import AutoReranker, Reranker
from app.retrieval.retriever import Bm25Retriever, Retriever
from app.retrieval.trie import QueryTrie
from app.schemas.search import SearchDocument, SearchResult

_log = logging.getLogger(__name__)


def _snippet(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "…"


def _metadata_spans(metadata: dict[str, object]) -> list[tuple[int, int]] | None:
    raw_spans = metadata.get("spans")
    if not isinstance(raw_spans, list):
        return None
    spans: list[tuple[int, int]] = []
    for item in raw_spans:
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], int)
            and isinstance(item[1], int)
        ):
            spans.append((item[0], item[1]))
    return spans or None


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _location_dict(location: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in ("page_number", "row_number", "line_number", "section"):
        value = getattr(location, key, None)
        if value is not None:
            result[key] = value
    return result


def _query_spans(query: str, text: str, chunk_id: str) -> list[tuple[int, int]]:
    patterns = build_query_patterns(query)
    if not patterns:
        return []
    from app.schemas.documents import Chunk, SourceLocation  # noqa: PLC0415

    chunk = Chunk(
        chunk_id=chunk_id,
        content=text,
        source_name="span-candidate",
        location=SourceLocation(),
    )
    return QueryTrie(patterns).spans_for_chunk(chunk)


def _span_window(text: str, start: int, end: int, limit: int = 280) -> tuple[int, int]:
    if len(text) <= limit:
        return 0, len(text)
    center = max(start, min(end, (start + end) // 2))
    window_start = max(0, center - limit // 2)
    window_end = min(len(text), window_start + limit)
    if window_end - window_start < limit:
        window_start = max(0, window_end - limit)
    return window_start, window_end


def _build_span_payload(
    text: str,
    start: int,
    end: int,
    source_base: int,
    location: dict[str, object],
) -> dict[str, object]:
    snippet_start, snippet_end = _span_window(text, start, end)
    snippet = text[snippet_start:snippet_end]
    highlight_start = start - snippet_start
    highlight_end = end - snippet_start
    source_start = source_base + start
    source_end = source_base + end
    return {
        "text": text[start:end],
        "snippet": snippet,
        "source_start": source_start,
        "source_end": source_end,
        "snippet_start": source_base + snippet_start,
        "snippet_end": source_base + snippet_end,
        "highlights": [
            {
                "start": highlight_start,
                "end": highlight_end,
                "source_start": source_start,
                "source_end": source_end,
                "text": text[start:end],
            }
        ],
        "location": location,
    }


def _build_context_span(
    text: str,
    source_base: int,
    location: dict[str, object],
    limit: int = 280,
) -> dict[str, object]:
    snippet = text[:limit]
    return {
        "text": snippet,
        "snippet": snippet,
        "source_start": source_base,
        "source_end": source_base + len(snippet),
        "snippet_start": source_base,
        "snippet_end": source_base + len(snippet),
        "highlights": [],
        "location": location,
    }


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
    retriever: Retriever | None = None
    confidence_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = Bm25Retriever(cache=self.bm25_cache)

    def search(
        self,
        query: str,
        documents: Sequence[SearchDocument],
        top_k: int = 10,
        workspace_id: str = "default",
        use_cache: bool = True,
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

        first_stage_hits = self.retriever.search(
            query,
            chunks,
            top_k=self.bm25_candidates,
            workspace_id=workspace_id,
            use_cache=use_cache,
        )
        reranked = self.reranker.rerank(query, first_stage_hits, top_k=top_k)
        results = [self._to_search_result(hit, query) for hit in reranked[:top_k]]

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

    def _to_search_result(self, hit: BM25ScoredChunk, query: str) -> SearchResult:
        metadata = dict(hit.chunk.metadata)
        source_base = _safe_int(
            metadata.get(
                "source_start",
                metadata.get("char_start", metadata.get("byte_start", 0)),
            )
        )
        spans = _metadata_spans(metadata) or _query_spans(query, hit.chunk.text, hit.chunk.chunk_id)
        location = _location_dict(hit.chunk.location)
        matched_spans = [
            _build_span_payload(hit.chunk.text, start, end, source_base, location)
            for start, end in spans
            if 0 <= start < end <= len(hit.chunk.text)
        ]
        if not matched_spans:
            matched_spans = [_build_context_span(hit.chunk.text, source_base, location)]
        primary_span = matched_spans[0]
        return SearchResult(
            content=hit.chunk.text,
            snippet=str(primary_span["snippet"]),
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
            spans=spans or None,
            primary_span=primary_span,
            matched_spans=matched_spans,
        )
