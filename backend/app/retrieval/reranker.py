from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.retrieval.bm25 import BM25ScoredChunk


_TERM_RE = re.compile(r"[a-z0-9]+")


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10) -> list[BM25ScoredChunk]:
        raise NotImplementedError


def _lexical_overlap_score(query: str, text: str) -> float:
    query_terms = set(_TERM_RE.findall(query.lower()))
    if not query_terms:
        return 0.0
    candidate_terms = set(_TERM_RE.findall(text.lower()))
    overlap = len(query_terms & candidate_terms)
    if not overlap:
        return 0.0
    return overlap / max(len(query_terms), 1)


@dataclass(slots=True)
class HeuristicReranker:
    """Deterministic fallback used when a model is unavailable."""

    def rerank(self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10) -> list[BM25ScoredChunk]:
        reranked = [
            BM25ScoredChunk(
                chunk=candidate.chunk,
                score=(candidate.score * 0.25) + _lexical_overlap_score(query, candidate.chunk.text),
                bm25_score=candidate.bm25_score if candidate.bm25_score is not None else candidate.score,
                rerank_score=(candidate.score * 0.25) + _lexical_overlap_score(query, candidate.chunk.text),
            )
            for candidate in candidates
        ]
        reranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return reranked[:top_k]


class AutoReranker:
    def __init__(self, model_path: str | Path | None = None, fallback: Reranker | None = None) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.fallback = fallback or HeuristicReranker()
        self._use_fallback = True
        self._session = None
        self._load_model()

    @property
    def is_fallback(self) -> bool:
        return self._use_fallback

    def _load_model(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            self._use_fallback = True
            return
        try:
            import onnxruntime as ort  # type: ignore
        except Exception:
            self._use_fallback = True
            return
        try:
            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        except Exception:
            self._session = None
            self._use_fallback = True
            return
        self._use_fallback = False

    def rerank(self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10) -> list[BM25ScoredChunk]:
        if self._use_fallback or self._session is None:
            return self.fallback.rerank(query, candidates, top_k=top_k)
        return self._onnx_rerank(query, candidates, top_k=top_k)

    def _onnx_rerank(self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10) -> list[BM25ScoredChunk]:
        # The model wire-up is intentionally conservative: if the ONNX model is
        # present but the preprocessing contract is unknown, we still fall back
        # to deterministic lexical ranking rather than emit bad scores.
        return self.fallback.rerank(query, candidates, top_k=top_k)
