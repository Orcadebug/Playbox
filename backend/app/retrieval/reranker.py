from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.retrieval.bm25 import _WORD_RE, BM25ScoredChunk
from services.reranker import build_remote_reranker_client

_log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10
    ) -> list[BM25ScoredChunk]:
        raise NotImplementedError


def _lexical_overlap_score(query: str, text: str) -> float:
    query_terms = set(_WORD_RE.findall(query.lower()))
    if not query_terms:
        return 0.0
    candidate_terms = set(_WORD_RE.findall(text.lower()))
    overlap = len(query_terms & candidate_terms)
    if not overlap:
        return 0.0
    return overlap / max(len(query_terms), 1)


@dataclass(slots=True)
class HeuristicReranker:
    """Deterministic fallback used when a model is unavailable."""

    def rerank(
        self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10
    ) -> list[BM25ScoredChunk]:
        reranked = [
            BM25ScoredChunk(
                chunk=candidate.chunk,
                score=(candidate.score * 0.25)
                + _lexical_overlap_score(query, candidate.chunk.text),
                bm25_score=(
                    candidate.bm25_score if candidate.bm25_score is not None else candidate.score
                ),
                rerank_score=(candidate.score * 0.25)
                + _lexical_overlap_score(query, candidate.chunk.text),
            )
            for candidate in candidates
        ]
        reranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        return reranked[:top_k]


class AutoReranker:
    def __init__(
        self,
        model_path: str | Path | None = None,
        fallback: Reranker | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        max_candidates: int = 200,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.fallback = fallback or HeuristicReranker()
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_candidates = max_candidates
        self._use_fallback = True
        self._remote_client = build_remote_reranker_client()
        self._session = None
        self._tokenizer_instance = None
        self._load_model()

    @property
    def is_fallback(self) -> bool:
        return self._use_fallback

    def _load_model(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            _log.debug("Reranker model not found at %s — using heuristic fallback", self.model_path)
            return

        # Require tokenizer.json in the same directory as the model file.
        tokenizer_path = self.model_path.parent / "tokenizer.json"
        if not tokenizer_path.exists():
            _log.warning(
                "tokenizer.json not found next to %s — run download_models.py. Using fallback.",
                self.model_path,
            )
            return

        try:
            from app.retrieval.tokenizer import CrossEncoderTokenizer  # noqa: PLC0415

            self._tokenizer_instance = CrossEncoderTokenizer(tokenizer_path, self._max_length)
        except Exception as exc:
            _log.warning("Failed to load cross-encoder tokenizer: %s — using fallback", exc)
            return

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]  # noqa: PLC0415

            self._session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        except Exception as exc:
            _log.warning("Failed to load ONNX session: %s — using fallback", exc)
            self._tokenizer_instance = None
            return

        self._use_fallback = False
        _log.info(
            "ONNX cross-encoder loaded from %s (batch_size=%d)", self.model_path, self._batch_size
        )

    def rerank(
        self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10
    ) -> list[BM25ScoredChunk]:
        if self._remote_client is not None:
            remote = self._remote_client.rerank(query, candidates, top_k=top_k)
            if remote is not None:
                return _apply_remote_scores(candidates, remote, top_k=top_k)
        if self._use_fallback or self._session is None:
            return self.fallback.rerank(query, candidates, top_k=top_k)
        # Truncate to max_candidates before neural inference to cap latency.
        if len(candidates) > self._max_candidates:
            candidates = candidates[: self._max_candidates]
        return self._onnx_rerank(query, candidates, top_k=top_k)

    def _onnx_rerank(
        self, query: str, candidates: list[BM25ScoredChunk], top_k: int = 10
    ) -> list[BM25ScoredChunk]:
        try:
            import numpy as np  # type: ignore[import-untyped]  # noqa: PLC0415

            texts = [c.chunk.text for c in candidates]
            results: list[BM25ScoredChunk] = []

            # Get the input names the ONNX model actually expects (e.g., some models
            # don't need token_type_ids).
            input_names = {inp.name for inp in self._session.get_inputs()}

            for batch_start in range(0, len(texts), self._batch_size):
                batch_texts = texts[batch_start : batch_start + self._batch_size]
                batch_candidates = candidates[batch_start : batch_start + self._batch_size]

                encoded = self._tokenizer_instance.encode_pairs(  # type: ignore[union-attr]
                    query, batch_texts, self._max_length
                )
                # Only feed inputs the model declares.
                feed = {k: v for k, v in encoded.items() if k in input_names}

                outputs = self._session.run(None, feed)
                logits = outputs[0]  # shape (batch, 1) or (batch,)
                if logits.ndim == 2:
                    logits = logits[:, 0]

                # Sigmoid → relevance scores in [0, 1]
                scores = (1.0 / (1.0 + np.exp(-logits))).tolist()

                for candidate, score in zip(batch_candidates, scores, strict=False):
                    results.append(
                        BM25ScoredChunk(
                            chunk=candidate.chunk,
                            score=float(score),
                            bm25_score=(
                                candidate.bm25_score
                                if candidate.bm25_score is not None
                                else candidate.score
                            ),
                            rerank_score=float(score),
                        )
                    )

            results.sort(key=lambda x: (-x.score, x.chunk.chunk_id))
            return results[:top_k]

        except Exception as exc:
            _log.error("ONNX reranking failed (%s) — switching to heuristic fallback", exc)
            self._use_fallback = True
            return self.fallback.rerank(query, candidates, top_k=top_k)


def _apply_remote_scores(
    candidates: list[BM25ScoredChunk],
    remote_scores: list[tuple[str, float]],
    *,
    top_k: int,
) -> list[BM25ScoredChunk]:
    by_id = {candidate.chunk.chunk_id: candidate for candidate in candidates}
    reranked: list[BM25ScoredChunk] = []
    for chunk_id, score in remote_scores:
        candidate = by_id.get(chunk_id)
        if candidate is None:
            continue
        reranked.append(
            BM25ScoredChunk(
                chunk=candidate.chunk,
                score=float(score),
                bm25_score=(
                    candidate.bm25_score if candidate.bm25_score is not None else candidate.score
                ),
                rerank_score=float(score),
            )
        )
    reranked.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return reranked[:top_k]
