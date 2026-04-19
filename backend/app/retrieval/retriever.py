from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
