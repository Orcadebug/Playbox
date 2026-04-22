from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.retrieval.bm25 import BM25Tokenizer
from app.retrieval.query_patterns import build_query_patterns
from app.retrieval.source_executor import SourceWindow
from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    NullProjection,
    SparseProjection,
)
from app.retrieval.trie import QueryTrie
from app.schemas.documents import Chunk

_REGEX_LITERAL_RE = re.compile(r'regex\("(.+?)"\)')


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def strip_regex_literals(query: str) -> str:
    return _REGEX_LITERAL_RE.sub(" ", query).strip()


def extract_regex_literals(query: str) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for body in _REGEX_LITERAL_RE.findall(query):
        try:
            patterns.append(re.compile(body, re.IGNORECASE))
        except re.error:
            continue
    return patterns


def tokenize_query(query: str) -> list[str]:
    tokenizer = BM25Tokenizer(use_stemming=False, use_stopwords=False)
    return tokenizer.tokenize(strip_regex_literals(query))


@dataclass(slots=True)
class ExactChannelState:
    normalized_query: str
    trie: QueryTrie | None
    regexes: list[re.Pattern[str]]


def build_exact_channel(query: str) -> ExactChannelState:
    stripped = strip_regex_literals(query)
    patterns = build_query_patterns(stripped) if stripped else []
    trie = QueryTrie(patterns) if patterns else None
    normalized = " ".join(stripped.lower().split())
    return ExactChannelState(
        normalized_query=normalized,
        trie=trie,
        regexes=extract_regex_literals(query),
    )


def score_exact_channel(
    state: ExactChannelState,
    window: SourceWindow,
) -> tuple[float, list[tuple[int, int]]]:
    text = window.text
    if not text:
        return 0.0, []

    spans: list[tuple[int, int]] = []
    score = 0.0
    chunk = Chunk(
        chunk_id=window.window_id,
        content=text,
        source_name=window.source_name,
        metadata={},
        location=window.location,
        token_count=max(1, len(text.split())),
    )

    if state.trie is not None:
        trie_score, trie_spans = state.trie.score_and_spans(chunk)
        if trie_score > 0:
            score += float(trie_score)
            spans.extend(trie_spans)

    lowered = text.lower()
    if state.normalized_query and len(state.normalized_query) > 2:
        start = lowered.find(state.normalized_query)
        while start >= 0:
            end = start + len(state.normalized_query)
            spans.append((start, end))
            score += 0.35
            start = lowered.find(state.normalized_query, start + 1)

    for regex in state.regexes:
        for match in regex.finditer(text):
            spans.append((match.start(), match.end()))
            score += 1.0

    return score, _merge_spans(spans)


def score_proxy_channel(
    query: str,
    windows: Sequence[SourceWindow],
    projection: SparseProjection | DeterministicSemanticProjection | NullProjection,
) -> list[float]:
    if not windows:
        return []
    if isinstance(projection, NullProjection):
        return [0.0 for _ in windows]

    chunks = [
        Chunk(
            chunk_id=window.window_id,
            content=window.text,
            source_name=window.source_name,
            metadata={},
            location=window.location,
            token_count=max(1, len(window.text.split())),
        )
        for window in windows
    ]
    query_vec = projection.encode_query(query)
    chunk_vecs = projection.encode_chunks(chunks)
    scores = projection.score(query_vec, chunk_vecs)
    return np.maximum(scores.astype(np.float32), 0.0).tolist()


def score_structure_channel(query_tokens: Sequence[str], window: SourceWindow) -> float:
    if not window.text:
        return 0.0

    score = 0.0
    if window.location.row_number is not None:
        score += 0.12
    if window.location.line_number is not None:
        score += 0.08
    if window.location.page_number is not None:
        score += 0.08
    if window.location.section:
        score += 0.05

    if window.source_origin == "raw":
        score += 0.05
    elif window.source_origin == "connector":
        score += 0.03
    elif window.source_origin == "stored":
        score += 0.02

    if window.neighboring_window_ids:
        score += min(0.08, 0.04 * len(window.neighboring_window_ids))

    metadata_blob = " ".join(
        str(value).lower()
        for value in (
            window.metadata.get("title"),
            window.metadata.get("source_type"),
            window.source_name,
            window.source_type,
        )
        if value is not None
    )
    if query_tokens and metadata_blob:
        overlap = sum(1 for token in query_tokens if token in metadata_blob)
        if overlap:
            score += min(0.18, overlap * 0.06)

    return min(score, 0.45)


def combined_channel_score(exact_score: float, proxy_score: float, structure_score: float) -> float:
    return (exact_score * 0.58) + (proxy_score * 0.30) + (structure_score * 0.12)


__all__ = [
    "ExactChannelState",
    "build_exact_channel",
    "combined_channel_score",
    "extract_regex_literals",
    "score_exact_channel",
    "score_proxy_channel",
    "score_structure_channel",
    "strip_regex_literals",
    "tokenize_query",
]
