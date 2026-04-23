from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.retrieval.bm25 import BM25Index, BM25ScoredChunk
from app.retrieval.bm25_cache import BM25IndexCache
from app.schemas.documents import Chunk


class Retriever(Protocol):
    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        raise NotImplementedError


@dataclass(slots=True)
class Bm25Retriever:
    cache: BM25IndexCache | None = None
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
        if not chunk_list:
            return []

        if self.cache is not None and use_cache:
            index = self.cache.get_or_build(workspace_id, chunk_list)
        else:
            index = BM25Index(
                chunk_list,
                use_stemming=self.use_stemming,
                use_stopwords=self.use_stopwords,
            )
        return index.search(query, top_k=top_k)


@dataclass(slots=True)
class MultiHeadRetriever:
    """Run multiple retriever heads in parallel and fuse via Reciprocal Rank
    Fusion (RRF). RRF is scale-free, so heads with different score ranges
    (BM25 unbounded, SPS in [0,1], phrase binary) combine cleanly.

    Populates each result's `channels` and `channel_scores` so API consumers
    see which head surfaced each chunk and at what raw score.
    """

    heads: list[tuple[str, Retriever]] = field(default_factory=list)
    rrf_k: int = 60

    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        chunk_list = list(chunks)
        if not chunk_list or not self.heads:
            return []

        per_head: dict[str, list[BM25ScoredChunk]] = {}
        for name, head in self.heads:
            per_head[name] = head.search(
                query,
                chunk_list,
                top_k=top_k,
                workspace_id=workspace_id,
                use_cache=use_cache,
            )

        fused: dict[str, dict[str, object]] = {}
        for name, hits in per_head.items():
            for rank, hit in enumerate(hits):
                cid = hit.chunk.chunk_id
                slot = fused.setdefault(
                    cid,
                    {
                        "hit": hit,
                        "score": 0.0,
                        "channels": [],
                        "channel_scores": {},
                        "bm25_score": hit.bm25_score,
                    },
                )
                slot["score"] = float(slot["score"]) + 1.0 / (self.rrf_k + rank + 1)
                slot["channels"].append(name)  # type: ignore[union-attr]
                slot["channel_scores"][name] = float(hit.score)  # type: ignore[index]
                # Keep the most informative bm25_score (non-None wins).
                if slot["bm25_score"] is None and hit.bm25_score is not None:
                    slot["bm25_score"] = hit.bm25_score

        ordered = sorted(
            fused.items(), key=lambda kv: (-float(kv[1]["score"]), kv[0])
        )

        results: list[BM25ScoredChunk] = []
        for _, slot in ordered[:top_k]:
            base: BM25ScoredChunk = slot["hit"]  # type: ignore[assignment]
            results.append(
                BM25ScoredChunk(
                    chunk=base.chunk,
                    score=float(slot["score"]),
                    bm25_score=slot["bm25_score"],  # type: ignore[arg-type]
                    rerank_score=None,
                    channels=list(slot["channels"]),  # type: ignore[arg-type]
                    channel_scores=dict(slot["channel_scores"]),  # type: ignore[arg-type]
                )
            )
        return results
