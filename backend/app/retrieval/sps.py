"""
Semantic Projection Search (SPS) retriever.

Fuses BM25 lexical scores with a projected-embedding cosine using a learned
(or deterministic fallback) sparse projection. Query-time cost is a single
HashingVectorizer transform plus a sparse-dense matmul — no neural inference.
Fixes BM25's synonym blindness while preserving exact-match ranking on lexical
queries.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from app.retrieval.bm25 import BM25Index, BM25ScoredChunk
from app.retrieval.bm25_cache import BM25IndexCache, _chunks_hash
from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    NullProjection,
    SparseProjection,
)
from app.schemas.documents import Chunk


@dataclass(slots=True)
class SpsRetriever:
    projection: SparseProjection | DeterministicSemanticProjection | NullProjection
    cache: BM25IndexCache | None = None
    alpha: float = 0.6
    candidate_multiplier: int = 3
    use_stemming: bool = True
    use_stopwords: bool = True

    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        chunk_list = list(chunks)
        if not chunk_list or not query.strip():
            return []

        content_hash = _chunks_hash(chunk_list)
        if self.cache is not None and use_cache:
            index = self.cache.get_or_build(workspace_id, chunk_list)
            embeddings = self.cache.get_embeddings(workspace_id, content_hash)
            if embeddings is None:
                embeddings = self.projection.encode_chunks(chunk_list)
                self.cache.set_embeddings(workspace_id, content_hash, embeddings)
        else:
            index = BM25Index(
                chunk_list,
                use_stemming=self.use_stemming,
                use_stopwords=self.use_stopwords,
            )
            embeddings = self.projection.encode_chunks(chunk_list)

        # Use the index's chunk order for positional alignment with embeddings.
        indexed_chunks = index.chunks
        if len(indexed_chunks) != len(chunk_list):
            # Cache returned a different index build; re-encode to stay aligned.
            embeddings = self.projection.encode_chunks(indexed_chunks)

        candidate_pool = max(top_k * self.candidate_multiplier, top_k)

        bm25_hits = index.search(query, top_k=candidate_pool)
        bm25_by_id: dict[str, float] = {
            hit.chunk.chunk_id: float(hit.bm25_score or hit.score) for hit in bm25_hits
        }

        query_vec = self.projection.encode_query(query)
        sps_scores = self.projection.score(query_vec, embeddings)

        id_to_pos = {chunk.chunk_id: i for i, chunk in enumerate(indexed_chunks)}

        # Top-SPS candidate ids
        if sps_scores.size > 0:
            sps_top_n = min(candidate_pool, sps_scores.size)
            sps_top_idx = np.argpartition(-sps_scores, sps_top_n - 1)[:sps_top_n]
            sps_candidate_ids = {indexed_chunks[i].chunk_id for i in sps_top_idx}
        else:
            sps_candidate_ids = set()

        candidate_ids = set(bm25_by_id.keys()) | sps_candidate_ids
        if not candidate_ids:
            return []

        bm25_raw = np.array(
            [bm25_by_id.get(cid, 0.0) for cid in candidate_ids], dtype=np.float32
        )
        sps_vals = np.array(
            [
                float(sps_scores[id_to_pos[cid]]) if cid in id_to_pos else 0.0
                for cid in candidate_ids
            ],
            dtype=np.float32,
        )

        bm25_max = float(bm25_raw.max()) if bm25_raw.size else 0.0
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        fused = self.alpha * bm25_norm + (1.0 - self.alpha) * sps_vals

        results: list[BM25ScoredChunk] = []
        for cid, score, raw in zip(candidate_ids, fused, bm25_raw, strict=True):
            pos = id_to_pos.get(cid)
            if pos is None:
                continue
            chunk = indexed_chunks[pos]
            results.append(
                BM25ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    bm25_score=float(raw),
                    rerank_score=None,
                )
            )

        results.sort(key=lambda r: (-r.score, r.chunk.chunk_id))
        return results[:top_k]
