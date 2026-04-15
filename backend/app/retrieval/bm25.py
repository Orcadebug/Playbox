from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.schemas.documents import Chunk


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(slots=True)
class BM25ScoredChunk:
    chunk: Chunk
    score: float
    bm25_score: float | None = None
    rerank_score: float | None = None


class BM25Index:
    def __init__(self, chunks: Iterable[Chunk] | None = None, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokenized_chunks: list[list[str]] = []
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._avgdl: float = 0.0
        if chunks is not None:
            self.build(chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def build(self, chunks: Iterable[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized_chunks = [_tokenize(chunk.text) for chunk in self._chunks]
        self._doc_freq = defaultdict(int)
        for tokens in self._tokenized_chunks:
            for term in set(tokens):
                self._doc_freq[term] += 1
        if self._tokenized_chunks:
            self._avgdl = sum(len(tokens) for tokens in self._tokenized_chunks) / len(self._tokenized_chunks)
        else:
            self._avgdl = 0.0

    def search(self, query: str, top_k: int = 10) -> list[BM25ScoredChunk]:
        if not self._chunks:
            return []
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored: list[BM25ScoredChunk] = []
        for chunk, tokens in zip(self._chunks, self._tokenized_chunks):
            score = self._score(tokens, query_terms)
            if score > 0:
                scored.append(BM25ScoredChunk(chunk=chunk, score=score, bm25_score=score, rerank_score=None))

        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return scored[:top_k]

    def _score(self, tokens: list[str], query_terms: list[str]) -> float:
        if not tokens or not query_terms:
            return 0.0
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (len(self._chunks) - df + 0.5) / (df + 0.5))
            freq = tf.get(term, 0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self._avgdl if self._avgdl else 1.0))
            score += idf * (numerator / denominator) if denominator else 0.0
        return score
