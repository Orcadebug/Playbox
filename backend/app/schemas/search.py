from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    @property
    def text(self) -> str:
        return self.content


@dataclass(slots=True)
class SearchRequest:
    query: str
    top_k: int = 10
    bm25_candidates: int = 100


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
]
