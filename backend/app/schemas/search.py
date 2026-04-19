from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class SearchDocument:
    file_name: str
    content: bytes | str
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> bytes:
        return self.content if isinstance(self.content, bytes) else self.content.encode("utf-8")


@dataclass(slots=True)
class RawSearchSource:
    name: str
    content: bytes | str
    id: str | None = None
    media_type: str | None = None
    source_type: str = "raw"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceError:
    source_type: str
    message: str
    connector_id: str | None = None
    source_name: str | None = None


@dataclass(slots=True)
class SearchSpan:
    text: str
    snippet: str
    source_start: int
    source_end: int
    snippet_start: int
    snippet_end: int
    highlights: list[dict[str, Any]] = field(default_factory=list)
    location: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchResult:
    content: str
    score: float
    source_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    snippet: str | None = None
    chunk_id: str | None = None
    parser_name: str | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    spans: list[tuple[int, int]] | None = None
    primary_span: dict[str, Any] | None = None
    matched_spans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.content


@dataclass(slots=True)
class SearchRequest:
    query: str
    top_k: int = 10
    bm25_candidates: int = 100
    raw_sources: list[RawSearchSource] = field(default_factory=list)
    connector_configs: list[dict[str, Any]] = field(default_factory=list)
    include_stored_sources: bool = True
    answer_mode: Literal["off", "llm"] = "off"


@dataclass(slots=True)
class SearchResponse:
    query: str
    results: list[SearchResult]

    @property
    def hits(self) -> list[SearchResult]:
        return self.results


SearchHit = SearchResult

__all__ = [
    "SearchDocument",
    "SearchHit",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "RawSearchSource",
    "SourceError",
    "SearchSpan",
]
