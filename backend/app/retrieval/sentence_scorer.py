"""
Two-stage span refinement: after chunk-level retrieval, pick the single
highest-relevance sentence inside the chunk as the primary_span anchor.

No new model. Uses the same projection already loaded by SpsRetriever, so
scoring per chunk is one small matmul — microseconds per result.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

import numpy as np

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_SENTENCE_CHARS = 12

_ENABLED = os.environ.get("WAVER_SENTENCE_RERANK", "true").lower() not in (
    "false", "0", "no",
)


class _ProjectionLike(Protocol):
    def encode_query(self, query: str) -> np.ndarray: ...
    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray: ...


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets for each sentence in text."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        end = match.start()
        if end > cursor:
            spans.append((cursor, end))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    # Trim leading/trailing whitespace per-span without losing offsets.
    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        s, e = start, end
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e - s >= _MIN_SENTENCE_CHARS:
            trimmed.append((s, e))
    return trimmed


def find_best_sentence(
    query: str,
    chunk_text: str,
    projection: _ProjectionLike,
) -> tuple[int, int] | None:
    """Return (start, end) of the highest-scoring sentence in chunk_text, or None
    if refinement doesn't apply (disabled, too-short text, single sentence)."""
    if not _ENABLED or not query.strip() or not chunk_text.strip():
        return None
    sentences = _split_sentences(chunk_text)
    if len(sentences) <= 1:
        return None

    try:
        query_vec = projection.encode_query(query)
        # Encode sentences without going through the Chunk schema — build a
        # tiny adapter payload. Projections that expect Chunk objects use only
        # .text, so a lightweight stand-in keeps this module dep-free.
        class _StubChunk:
            __slots__ = ("text",)

            def __init__(self, t: str) -> None:
                self.text = t

        stubs = [_StubChunk(chunk_text[s:e]) for s, e in sentences]
        vecs = projection.encode_chunks(stubs)  # type: ignore[arg-type]
        scores = projection.score(query_vec, vecs)
    except Exception:
        return None

    if scores.size == 0:
        return None
    best_idx = int(np.argmax(scores))
    if float(scores[best_idx]) <= 0.0:
        return None
    return sentences[best_idx]
