from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from app.parsers.base import ParserDetector, SourceLocation
from app.retrieval.bm25 import BM25ScoredChunk
from app.retrieval.channels import (
    build_exact_channel,
    combined_channel_score,
    score_exact_channel,
    score_proxy_channel,
    score_structure_channel,
    tokenize_query,
)
from app.retrieval.planner import QueryPlan, build_query_plan
from app.retrieval.reranker import Reranker
from app.retrieval.source_executor import SourceWindow, build_source_windows_from_documents
from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    NullProjection,
    SparseProjection,
)
from app.retrieval.trie import QueryTrie
from app.schemas.documents import Chunk
from app.schemas.search import SearchDocument, SearchResult

_CHANNEL_ORDER = ("exact", "semantic", "structure")
_ChannelName = Literal["exact", "semantic", "structure"]


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
    from app.retrieval.query_patterns import build_query_patterns

    patterns = build_query_patterns(query)
    if not patterns:
        return []
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
    offset_basis: str,
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
        "offset_basis": offset_basis,
        "highlights": [
            {
                "start": highlight_start,
                "end": highlight_end,
                "source_start": source_start,
                "source_end": source_end,
                "offset_basis": offset_basis,
                "text": text[start:end],
            }
        ],
        "location": location,
    }


def _build_context_span(
    text: str,
    source_base: int,
    location: dict[str, object],
    offset_basis: str,
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
        "offset_basis": offset_basis,
        "highlights": [],
        "location": location,
    }


@dataclass(slots=True)
class WindowCandidate:
    window: SourceWindow
    source_order: int
    exact_score: float
    proxy_score: float
    structure_score: float
    channel_scores: dict[str, float]
    score: float
    phase: Literal["exact", "proxy"]
    channels: list[str]
    spans: list[tuple[int, int]]


@dataclass(slots=True)
class SpanExecutionOutcome:
    plan: QueryPlan
    exact_results: list[SearchResult]
    proxy_results: list[SearchResult]
    reranked_results: list[SearchResult]
    final_results: list[SearchResult]
    execution: dict[str, Any]


class SpanExecutor:
    def __init__(
        self,
        *,
        parser_detector: ParserDetector,
        reranker: Reranker,
        projection: SparseProjection | DeterministicSemanticProjection | NullProjection,
        enabled_channels: Iterable[str] | None = None,
        rerank_enabled: bool = True,
    ) -> None:
        self.parser_detector = parser_detector
        self.reranker = reranker
        self.projection = projection
        requested = set(enabled_channels) if enabled_channels is not None else set(_CHANNEL_ORDER)
        unknown = requested - set(_CHANNEL_ORDER)
        if unknown:
            raise ValueError(f"Unsupported retrieval channels: {sorted(unknown)}")
        self.enabled_channels: tuple[_ChannelName, ...] = tuple(
            channel for channel in _CHANNEL_ORDER if channel in requested
        )  # type: ignore[assignment]
        if not self.enabled_channels:
            raise ValueError("At least one retrieval channel must be enabled")
        self.rerank_enabled = rerank_enabled

    def execute(
        self,
        *,
        query: str,
        documents: Sequence[SearchDocument],
        top_k: int,
        cacheable: bool = False,
    ) -> SpanExecutionOutcome:
        started_at = perf_counter()
        phase_timings: dict[str, float] = {}
        window_started_at = perf_counter()
        bytes_loaded = sum(len(document.data) for document in documents)
        windows = build_source_windows_from_documents(
            documents,
            parser_detector=self.parser_detector,
        )
        phase_timings["window_load_ms"] = round((perf_counter() - window_started_at) * 1000.0, 3)
        origins = {window.source_origin for window in windows}
        plan = build_query_plan(
            query=query,
            top_k=top_k,
            window_count=len(windows),
            source_origins=origins,
            cacheable=cacheable,
        )
        scanned_windows = windows[: plan.scan_limit]

        if not scanned_windows:
            elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)
            phase_timings["total_ms"] = elapsed_ms
            execution = {
                **plan.to_dict(),
                "active_channels": list(self.enabled_channels),
                "scanned_windows": 0,
                "candidate_count": 0,
                "shortlisted_candidates": 0,
                "elapsed_ms": elapsed_ms,
                "bytes_loaded": bytes_loaded,
                "skipped_windows": len(windows),
                "completion_reason": "no_windows",
                "phase_timings_ms": phase_timings,
                "time_to_first_useful_ms": 0.0,
                "time_to_final_ms": elapsed_ms,
                "phase_counts": {
                    "exact_results": 0,
                    "proxy_results": 0,
                    "reranked_results": 0,
                    "final_results": 0,
                },
            }
            return SpanExecutionOutcome(
                plan=plan,
                exact_results=[],
                proxy_results=[],
                reranked_results=[],
                final_results=[],
                execution=execution,
            )

        exact_enabled = "exact" in self.enabled_channels
        semantic_enabled = "semantic" in self.enabled_channels
        structure_enabled = "structure" in self.enabled_channels

        exact_started_at = perf_counter()
        exact_state = build_exact_channel(query) if exact_enabled else None
        phase_timings["exact_setup_ms"] = round((perf_counter() - exact_started_at) * 1000.0, 3)

        structure_started_at = perf_counter()
        query_tokens = tokenize_query(query) if structure_enabled else []
        phase_timings["structure_setup_ms"] = round(
            (perf_counter() - structure_started_at) * 1000.0,
            3,
        )

        semantic_started_at = perf_counter()
        proxy_scores = (
            score_proxy_channel(query, scanned_windows, self.projection)
            if semantic_enabled
            else [0.0 for _ in scanned_windows]
        )
        phase_timings["semantic_ms"] = round((perf_counter() - semantic_started_at) * 1000.0, 3)

        candidates: list[WindowCandidate] = []
        scoring_started_at = perf_counter()
        for index, window in enumerate(scanned_windows):
            if exact_state is not None:
                exact_score, spans = score_exact_channel(exact_state, window)
            else:
                exact_score, spans = 0.0, []
            proxy_score = float(proxy_scores[index]) if index < len(proxy_scores) else 0.0
            structure_score = (
                score_structure_channel(query_tokens, window) if structure_enabled else 0.0
            )

            channels: list[str] = []
            if exact_score > 0:
                channels.append("exact")
            if proxy_score > 0:
                channels.append("semantic")
            if structure_score > 0:
                channels.append("structure")
            if not channels:
                continue

            total = combined_channel_score(exact_score, proxy_score, structure_score)
            candidates.append(
                WindowCandidate(
                    window=window,
                    source_order=index,
                    exact_score=exact_score,
                    proxy_score=proxy_score,
                    structure_score=structure_score,
                    channel_scores={
                        "exact": exact_score,
                        "semantic": proxy_score,
                        "structure": structure_score,
                    },
                    score=total,
                    phase="exact" if exact_score > 0 else "proxy",
                    channels=channels,
                    spans=spans,
                )
            )
        phase_timings["candidate_score_ms"] = round(
            (perf_counter() - scoring_started_at) * 1000.0,
            3,
        )

        candidates.sort(
            key=lambda item: (
                -item.score,
                -item.exact_score,
                -item.proxy_score,
                item.source_order,
                item.window.window_id,
            )
        )
        limited = candidates[: plan.candidate_limit]
        exact_candidates = [candidate for candidate in limited if candidate.exact_score > 0][:top_k]
        proxy_candidates = limited[:top_k]
        shortlist = limited[: plan.rerank_limit]

        exact_results = [
            self._to_search_result(self._candidate_to_hit(candidate, phase="exact"), query)
            for candidate in exact_candidates
        ]
        proxy_results = [
            self._to_search_result(
                self._candidate_to_hit(
                    candidate,
                    phase="exact" if candidate.exact_score > 0 else "proxy",
                ),
                query,
            )
            for candidate in proxy_candidates
        ]
        time_to_first_useful_ms = 0.0
        if exact_results or proxy_results:
            time_to_first_useful_ms = round((perf_counter() - started_at) * 1000.0, 3)

        rerank_started_at = perf_counter()
        if self.rerank_enabled:
            reranked_hits = self.reranker.rerank(
                query,
                [self._candidate_to_hit(candidate, phase="reranked") for candidate in shortlist],
                top_k=top_k,
            )
            reranked_results = [self._to_search_result(hit, query) for hit in reranked_hits]
        else:
            reranked_results = []
        phase_timings["rerank_ms"] = round((perf_counter() - rerank_started_at) * 1000.0, 3)
        final_results = reranked_results or proxy_results or exact_results

        completion_reason = "partial_scan_limit" if plan.partial else "complete"
        if plan.partial:
            for result in (
                exact_results
                + proxy_results
                + reranked_results
                + final_results
            ):
                result.metadata["retrieval_partial"] = True
                result.metadata["completion_reason"] = completion_reason

        elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)
        phase_timings["total_ms"] = elapsed_ms
        execution = {
            **plan.to_dict(),
            "active_channels": list(self.enabled_channels),
            "scanned_windows": len(scanned_windows),
            "candidate_count": len(candidates),
            "shortlisted_candidates": len(shortlist),
            "elapsed_ms": elapsed_ms,
            "bytes_loaded": bytes_loaded,
            "skipped_windows": max(0, len(windows) - len(scanned_windows)),
            "completion_reason": completion_reason,
            "phase_timings_ms": phase_timings,
            "time_to_first_useful_ms": time_to_first_useful_ms,
            "time_to_final_ms": elapsed_ms,
            "phase_counts": {
                "exact_results": len(exact_results),
                "proxy_results": len(proxy_results),
                "reranked_results": len(reranked_results),
                "final_results": len(final_results),
            },
        }

        return SpanExecutionOutcome(
            plan=plan,
            exact_results=exact_results,
            proxy_results=proxy_results,
            reranked_results=reranked_results,
            final_results=final_results,
            execution=execution,
        )

    def _candidate_to_hit(self, candidate: WindowCandidate, phase: str) -> BM25ScoredChunk:
        metadata = dict(candidate.window.metadata)
        metadata.update(
            {
                "source_id": candidate.window.source_id,
                "source_type": candidate.window.source_type,
                "source_origin": candidate.window.source_origin,
                "parser_name": candidate.window.parser_name,
                "source_start": candidate.window.source_start,
                "source_end": candidate.window.source_end,
                "char_start": candidate.window.source_start,
                "char_end": candidate.window.source_end,
                "byte_start": candidate.window.source_start,
                "byte_end": candidate.window.source_end,
                "spans": candidate.spans,
                "phase": phase,
                "channels": candidate.channels,
                "channel_scores": candidate.channel_scores,
            }
        )
        chunk = Chunk(
            chunk_id=candidate.window.window_id,
            content=candidate.window.text,
            source_name=candidate.window.source_name,
            metadata=metadata,
            location=candidate.window.location,
            token_count=max(1, len(candidate.window.text.split())),
        )
        return BM25ScoredChunk(
            chunk=chunk,
            score=candidate.score,
            bm25_score=candidate.score,
            rerank_score=candidate.score,
        )

    def _to_search_result(self, hit: BM25ScoredChunk, query: str) -> SearchResult:
        metadata = dict(hit.chunk.metadata)
        source_base = _safe_int(
            metadata.get(
                "source_start",
                metadata.get("char_start", metadata.get("byte_start", 0)),
            )
        )
        metadata_spans = _metadata_spans(metadata)
        if metadata_spans is None and "spans" not in metadata:
            spans = _query_spans(query, hit.chunk.text, hit.chunk.chunk_id)
        else:
            spans = metadata_spans or []
        location = _location_dict(hit.chunk.location)
        offset_basis = str(metadata.get("offset_basis") or "source")
        if offset_basis not in {"source", "parsed"}:
            offset_basis = "source"
        matched_spans = [
            _build_span_payload(hit.chunk.text, start, end, source_base, location, offset_basis)
            for start, end in spans
            if 0 <= start < end <= len(hit.chunk.text)
        ]
        if not matched_spans:
            matched_spans = [
                _build_context_span(hit.chunk.text, source_base, location, offset_basis)
            ]
        primary_span = matched_spans[0]
        raw_channels = (
            metadata.get("channels") if isinstance(metadata.get("channels"), list) else []
        )
        channels = [str(channel) for channel in raw_channels]
        raw_channel_scores = (
            metadata.get("channel_scores")
            if isinstance(metadata.get("channel_scores"), dict)
            else {}
        )
        channel_scores = {
            "exact": float(raw_channel_scores.get("exact", 0.0)),
            "semantic": float(raw_channel_scores.get("semantic", 0.0)),
            "structure": float(raw_channel_scores.get("structure", 0.0)),
        }
        if hit.rerank_score is not None:
            channel_scores["rerank"] = float(hit.rerank_score)
        metadata["channel_scores"] = channel_scores
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
            channels=channels,
            channel_scores=channel_scores,
        )


__all__ = ["SpanExecutionOutcome", "SpanExecutor", "WindowCandidate"]
