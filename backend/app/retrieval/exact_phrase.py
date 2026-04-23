"""
Exact-phrase retriever — binary substring match on query n-grams.

Part of the multi-head union: catches literal quoted strings, error codes, and
short phrases that SPS cosine may dampen and BM25 may rank below fuzzy matches.
No scoring beyond presence (1.0 if any query n-gram appears in chunk, else skip).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.retrieval.bm25 import BM25ScoredChunk, BM25Tokenizer
from app.schemas.documents import Chunk


@dataclass(slots=True)
class ExactPhraseRetriever:
    ngram_min: int = 2
    ngram_max: int = 3
    _tokenizer: BM25Tokenizer = field(
        default_factory=lambda: BM25Tokenizer(use_stemming=False, use_stopwords=True)
    )

    def _phrases(self, query: str) -> list[str]:
        tokens = self._tokenizer.tokenize(query)
        if not tokens:
            return []
        phrases: list[str] = []
        for n in range(self.ngram_min, self.ngram_max + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                phrases.append(" ".join(tokens[i : i + n]))
        # Include the full query (lowercased) as the strongest phrase signal.
        full = query.strip().lower()
        if full and full not in phrases:
            phrases.append(full)
        return phrases

    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        del workspace_id, use_cache
        chunk_list = list(chunks)
        if not chunk_list or not query.strip():
            return []
        phrases = self._phrases(query)
        if not phrases:
            return []

        hits: list[BM25ScoredChunk] = []
        for chunk in chunk_list:
            haystack = chunk.text.lower()
            if any(phrase in haystack for phrase in phrases):
                hits.append(
                    BM25ScoredChunk(
                        chunk=chunk,
                        score=1.0,
                        bm25_score=None,
                        rerank_score=None,
                        channels=["phrase"],
                        channel_scores={"phrase": 1.0},
                    )
                )
        return hits[:top_k]
