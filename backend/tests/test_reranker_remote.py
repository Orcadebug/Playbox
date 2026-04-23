from __future__ import annotations

from app.retrieval.bm25 import BM25ScoredChunk
from app.retrieval.reranker import AutoReranker
from app.schemas.documents import Chunk, SourceLocation
from services.reranker import build_remote_reranker_client


def test_remote_reranker_client_disabled_without_target(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("WAVER_RERANKER_GRPC_TARGET", raising=False)

    client = build_remote_reranker_client()

    assert client is None


def test_auto_reranker_prefers_remote_scores(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeClient:
        def rerank(self, query, candidates, top_k=10):  # type: ignore[no-untyped-def]
            assert query == "billing refund"
            return [
                (candidates[1].chunk.chunk_id, 0.95),
                (candidates[0].chunk.chunk_id, 0.10),
            ]

    monkeypatch.setattr("app.retrieval.reranker.build_remote_reranker_client", lambda: FakeClient())

    reranker = AutoReranker(model_path=None)
    candidates = [
        BM25ScoredChunk(
            chunk=Chunk(
                chunk_id="a",
                content="billing refund request",
                source_name="a.txt",
                location=SourceLocation(),
            ),
            score=0.3,
            bm25_score=0.3,
        ),
        BM25ScoredChunk(
            chunk=Chunk(
                chunk_id="b",
                content="duplicate charge refund escalation",
                source_name="b.txt",
                location=SourceLocation(),
            ),
            score=0.2,
            bm25_score=0.2,
        ),
    ]

    reranked = reranker.rerank("billing refund", candidates, top_k=2)

    assert [item.chunk.chunk_id for item in reranked] == ["b", "a"]
