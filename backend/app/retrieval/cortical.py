from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.retrieval.adjacency import build_adjacency
from app.retrieval.bm25 import BM25ScoredChunk
from app.retrieval.diffusion import DiffusionConfig, diffuse
from app.retrieval.gating import thalamic_gate
from app.retrieval.query_patterns import build_query_patterns
from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    NullProjection,
    SparseProjection,
)
from app.retrieval.trie import QueryTrie
from app.schemas.documents import Chunk


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    max_value = float(np.max(values)) if len(values) else 0.0
    if max_value <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return values / max_value


def _token_costs(chunks: list[Chunk]) -> np.ndarray:
    raw = np.array(
        [
            chunk.token_count if chunk.token_count > 0 else len(chunk.text.split())
            for chunk in chunks
        ],
        dtype=np.float32,
    )
    return _normalize(raw)


def default_trie_builder(
    max_patterns: int = 2000,
    phrase_ngram: int = 2,
    phrase_weight: float = 1.5,
) -> Callable[[str], QueryTrie]:
    def build(query: str) -> QueryTrie:
        return QueryTrie(
            build_query_patterns(
                query,
                max_patterns=max_patterns,
                phrase_ngram=phrase_ngram,
                phrase_weight=phrase_weight,
            )
        )

    return build


@dataclass(slots=True)
class CorticalRetriever:
    projection: SparseProjection | DeterministicSemanticProjection | NullProjection
    trie_builder: Callable[[str], QueryTrie] = field(default_factory=default_trie_builder)
    gating_m: int = 80
    lambdas: tuple[float, float, float, float] = (0.35, 0.35, 0.20, 0.10)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    candidate_cap: int = 2000
    adjacency_max_edges: int = 8

    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        chunk_list = list(chunks)
        if not chunk_list:
            return []

        trie = self.trie_builder(query)
        lexical_scores: list[float] = []
        spans: list[list[tuple[int, int]]] = []
        for chunk in chunk_list:
            score, chunk_spans = trie.score_and_spans(chunk)
            lexical_scores.append(score)
            spans.append(chunk_spans)

        L = _normalize(np.array(lexical_scores, dtype=np.float32))
        query_vec = self.projection.encode_query(query)
        chunk_vecs = self.projection.encode_chunks(chunk_list)
        S = np.asarray(self.projection.score(query_vec, chunk_vecs), dtype=np.float32)

        salience = L + S
        candidate_indices = np.flatnonzero(salience > 0)
        if len(candidate_indices) == 0:
            return []

        ordered_indices = sorted(
            candidate_indices.tolist(),
            key=lambda i: (-float(salience[i]), chunk_list[i].chunk_id),
        )[: self.candidate_cap]
        candidates = [chunk_list[i] for i in ordered_indices]
        candidate_spans = [spans[i] for i in ordered_indices]
        cand_L = L[ordered_indices]
        cand_S = S[ordered_indices]

        A = build_adjacency(candidates, max_edges=self.adjacency_max_edges)
        C = diffuse(
            A,
            cand_L,
            cand_S,
            steps=self.diffusion.steps,
            beta=self.diffusion.beta,
            gamma=self.diffusion.gamma,
            delta=self.diffusion.delta,
        )
        m = min(self.gating_m, top_k)
        return thalamic_gate(
            candidates,
            cand_L,
            cand_S,
            C,
            costs=_token_costs(candidates),
            m=m,
            lambdas=self.lambdas,
            spans=candidate_spans,
        )
