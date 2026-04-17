from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import ahocorasick

from app.schemas.documents import Chunk


@dataclass(slots=True)
class Pattern:
    id: int
    body: str
    weight: float
    kind: Literal["token", "phrase"]


@dataclass(slots=True)
class TrieHit:
    chunk_id: str
    start: int
    end: int
    pattern_id: int
    score: float


def _is_word_char(value: str) -> bool:
    return value.isascii() and value.isalnum()


def _has_word_boundary(text: str, start: int, end: int) -> bool:
    before_ok = start == 0 or not _is_word_char(text[start - 1])
    after_ok = end >= len(text) or not _is_word_char(text[end])
    return before_ok and after_ok


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class QueryTrie:
    def __init__(self, patterns: list[Pattern]) -> None:
        self.patterns = patterns
        self._pattern_by_id = {pattern.id: pattern for pattern in patterns}
        self._automaton = ahocorasick.Automaton()
        for pattern in patterns:
            self._automaton.add_word(pattern.body, pattern.id)
        self._automaton.make_automaton()

    def scan(self, chunk: Chunk) -> Iterator[TrieHit]:
        text = chunk.text.lower()
        for end_index, pattern_id in self._automaton.iter(text):
            pattern = self._pattern_by_id[pattern_id]
            end = end_index + 1
            start = end - len(pattern.body)
            if not _has_word_boundary(text, start, end):
                continue
            yield TrieHit(
                chunk_id=chunk.chunk_id,
                start=start,
                end=end,
                pattern_id=pattern.id,
                score=pattern.weight,
            )

    def score_chunk(self, chunk: Chunk) -> float:
        hits = list(self.scan(chunk))
        return self.score_hits(hits)

    def spans_for_chunk(self, chunk: Chunk) -> list[tuple[int, int]]:
        return _merge_spans([(hit.start, hit.end) for hit in self.scan(chunk)])

    def score_and_spans(self, chunk: Chunk) -> tuple[float, list[tuple[int, int]]]:
        hits = list(self.scan(chunk))
        return self.score_hits(hits), _merge_spans([(hit.start, hit.end) for hit in hits])

    def score_hits(self, hits: list[TrieHit]) -> float:
        if not hits:
            return 0.0
        counts = Counter(hit.pattern_id for hit in hits)
        return sum(
            self._pattern_by_id[pattern_id].weight * math.log1p(count)
            for pattern_id, count in counts.items()
        )
