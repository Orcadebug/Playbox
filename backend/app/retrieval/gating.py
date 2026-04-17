from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from app.retrieval.bm25 import BM25ScoredChunk
from app.schemas.documents import Chunk


def thalamic_gate(
    chunks: list[Chunk],
    L: np.ndarray,
    S: np.ndarray,
    C: np.ndarray,
    costs: np.ndarray,
    m: int,
    lambdas: tuple[float, float, float, float],
    spans: Sequence[list[tuple[int, int]]] | None = None,
) -> list[BM25ScoredChunk]:
    if not chunks or m <= 0:
        return []

    R = lambdas[0] * L + lambdas[1] * S + lambdas[2] * C - lambdas[3] * costs
    count = min(m, len(chunks))
    if count == len(chunks):
        selected = np.arange(len(chunks))
    else:
        selected = np.argpartition(-R, count - 1)[:count]

    ordered = sorted(selected.tolist(), key=lambda i: (-float(R[i]), chunks[i].chunk_id))
    results: list[BM25ScoredChunk] = []
    for i in ordered:
        metadata = dict(chunks[i].metadata)
        metadata.update(
            {
                "retriever": "cortical",
                "retrieval_score": float(R[i]),
                "lexical_score": float(L[i]),
                "semantic_score": float(S[i]),
                "diffusion_score": float(C[i]),
                "retrieval_cost": float(costs[i]),
            }
        )
        if spans is not None and spans[i]:
            metadata["spans"] = spans[i]
        chunk = replace(chunks[i], metadata=metadata)
        results.append(BM25ScoredChunk(chunk=chunk, score=float(R[i])))
    return results
