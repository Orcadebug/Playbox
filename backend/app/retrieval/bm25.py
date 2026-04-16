from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from app.schemas.documents import Chunk

# ---------------------------------------------------------------------------
# Stemmer — optional PyStemmer (Snowball, C-extension, ~1M words/sec).
# Falls back to no stemming if the package is absent.
# ---------------------------------------------------------------------------

try:
    import Stemmer as _PyStemmer  # type: ignore[import-untyped]

    _stemmer = _PyStemmer.Stemmer("english")
    _STEMMER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _stemmer = None  # type: ignore[assignment]
    _STEMMER_AVAILABLE = False


# ---------------------------------------------------------------------------
# English stopwords — hardcoded frozenset, no NLTK dependency required.
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "couldn", "did", "didn", "do",
    "does", "doesn", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "get", "got", "had", "hadn", "has", "hasn", "have", "haven", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "isn", "it", "its", "itself", "just", "ll", "me", "mightn",
    "more", "most", "mustn", "my", "myself", "needn", "no", "nor", "not", "now", "o",
    "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "re", "s", "same", "shan", "she", "should", "shouldn", "so",
    "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "us", "ve", "very", "was", "wasn", "we", "were", "weren", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "wouldn", "y", "you", "your", "yours", "yourself", "yourselves",
})

_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# BM25Tokenizer
# ---------------------------------------------------------------------------

class BM25Tokenizer:
    """
    Tokenizer used by ``BM25Index``.

    Steps (each configurable):
    1. Lowercase + regex word extraction ``[a-z0-9]+``
    2. Stopword removal (English, hardcoded frozenset)
    3. Snowball stemming via PyStemmer (graceful no-op if unavailable)
    """

    __slots__ = ("_use_stemming", "_use_stopwords")

    def __init__(self, use_stemming: bool = True, use_stopwords: bool = True) -> None:
        self._use_stemming = use_stemming and _STEMMER_AVAILABLE
        self._use_stopwords = use_stopwords

    def tokenize(self, text: str) -> list[str]:
        tokens = _WORD_RE.findall(text.lower())
        if self._use_stopwords:
            tokens = [t for t in tokens if t not in _STOPWORDS]
        if self._use_stemming and _stemmer is not None:
            tokens = [_stemmer.stemWord(t) for t in tokens]
        return tokens


# Module-level default tokenizer — avoids repeated construction.
_DEFAULT_TOKENIZER = BM25Tokenizer()


def _tokenize(text: str) -> list[str]:
    """Convenience wrapper using the default tokenizer."""
    return _DEFAULT_TOKENIZER.tokenize(text)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BM25ScoredChunk:
    chunk: Chunk
    score: float
    bm25_score: float | None = None
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(
        self,
        chunks: Iterable[Chunk] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        use_stemming: bool = True,
        use_stopwords: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._tokenizer = BM25Tokenizer(use_stemming=use_stemming, use_stopwords=use_stopwords)
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
        self._tokenized_chunks = [self._tokenizer.tokenize(chunk.text) for chunk in self._chunks]
        self._doc_freq = defaultdict(int)
        for tokens in self._tokenized_chunks:
            for term in set(tokens):
                self._doc_freq[term] += 1
        if self._tokenized_chunks:
            self._avgdl = (
                sum(len(t) for t in self._tokenized_chunks) / len(self._tokenized_chunks)
            )
        else:
            self._avgdl = 0.0

    def search(self, query: str, top_k: int = 10) -> list[BM25ScoredChunk]:
        """Score all chunks against *query* and return top-k by BM25 score.

        Uses a min-heap for O(n log k) complexity instead of a full sort.
        """
        if not self._chunks:
            return []
        query_terms = self._tokenizer.tokenize(query)
        if not query_terms:
            return []

        # Min-heap of (score, chunk_id, BM25ScoredChunk) — negated score for max-heap semantics.
        heap: list[tuple[float, str, BM25ScoredChunk]] = []

        for chunk, tokens in zip(self._chunks, self._tokenized_chunks, strict=False):
            score = self._score(tokens, query_terms)
            if score <= 0:
                continue
            item = BM25ScoredChunk(chunk=chunk, score=score, bm25_score=score, rerank_score=None)
            if len(heap) < top_k:
                heapq.heappush(heap, (score, chunk.chunk_id, item))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, chunk.chunk_id, item))

        # Sort descending by score, then chunk_id for determinism.
        heap.sort(key=lambda x: (-x[0], x[1]))
        return [x[2] for x in heap]

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
            denominator = freq + self.k1 * (
                1 - self.b + self.b * (doc_len / self._avgdl if self._avgdl else 1.0)
            )
            score += idf * (numerator / denominator) if denominator else 0.0
        return score
