from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.bm25 import BM25ScoredChunk

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class RemoteRerankerClient:
    target: str
    timeout_ms: int = 1500

    @classmethod
    def from_env(cls) -> RemoteRerankerClient | None:
        target = os.environ.get("WAVER_RERANKER_GRPC_TARGET", "").strip()
        if not target:
            return None
        timeout_ms = int(os.environ.get("WAVER_RERANKER_GRPC_TIMEOUT_MS", "1500"))
        return cls(target=target, timeout_ms=timeout_ms)

    def rerank(
        self,
        query: str,
        candidates: list[BM25ScoredChunk],
        top_k: int,
    ) -> list[tuple[str, float]] | None:
        try:
            import grpc  # type: ignore[import-untyped]
        except Exception:
            _log.warning("grpc not installed; remote reranker disabled")
            return None

        try:
            from services.reranker.generated import (  # type: ignore[import-not-found]
                reranker_pb2,
                reranker_pb2_grpc,
            )
        except Exception:
            _log.warning("reranker gRPC stubs not generated; remote reranker disabled")
            return None

        payload = reranker_pb2.RerankRequest(
            query=query,
            top_k=top_k,
            candidates=[
                reranker_pb2.Candidate(id=item.chunk.chunk_id, text=item.chunk.text)
                for item in candidates
            ],
        )
        try:
            with grpc.insecure_channel(self.target) as channel:
                stub = reranker_pb2_grpc.RerankerStub(channel)
                response = stub.Rerank(payload, timeout=self.timeout_ms / 1000.0)
        except Exception as exc:
            _log.warning("Remote reranker failed (%s); using local fallback", exc)
            return None

        return [(item.id, float(item.score)) for item in response.scored_candidates]


def build_remote_reranker_client() -> RemoteRerankerClient | None:
    return RemoteRerankerClient.from_env()
