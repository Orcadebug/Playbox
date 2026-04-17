from __future__ import annotations

import numpy as np
from scipy import sparse

from app.config import Settings
from app.retrieval.adjacency import build_adjacency
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.cortical import CorticalRetriever, default_trie_builder
from app.retrieval.diffusion import diffuse
from app.retrieval.gating import thalamic_gate
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import HeuristicReranker
from app.retrieval.sparse_projection import NullProjection, ProjectionConfig, load_projection
from app.schemas.documents import Chunk, SourceLocation
from app.schemas.search import SearchDocument
from app.services.search import _build_retriever, _pipeline_candidate_budget


def _chunk(
    chunk_id: str,
    content: str,
    source_id: str = "source-1",
    chunk_index: int = 0,
    page_number: int | None = None,
    row_number: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        source_name=f"{source_id}.txt",
        metadata={"source_id": source_id, "chunk_index": chunk_index},
        location=SourceLocation(page_number=page_number, row_number=row_number),
        token_count=len(content.split()),
    )


def _retriever(projection: NullProjection | None = None) -> CorticalRetriever:
    return CorticalRetriever(
        projection=projection or NullProjection(ProjectionConfig(hash_features=16, dim=4)),
        trie_builder=default_trie_builder(),
        gating_m=5,
        candidate_cap=20,
    )


def test_adjacency_same_source_adjacent_chunks_weight_one() -> None:
    chunks = [
        _chunk("a", "billing refund", chunk_index=0),
        _chunk("b", "billing details", chunk_index=1),
    ]

    A = build_adjacency(chunks)

    assert A[0, 1] == 1.0
    assert A[1, 0] == 1.0


def test_adjacency_caps_edges_per_node() -> None:
    chunks = [
        _chunk(f"c{i}", f"row {i}", chunk_index=i, page_number=1)
        for i in range(10)
    ]

    A = build_adjacency(chunks, max_edges=3)

    assert np.all(np.diff(A.indptr) <= 3)


def test_diffusion_converges_with_no_input_produces_zero() -> None:
    A = sparse.csr_matrix((3, 3), dtype=np.float32)
    C = diffuse(A, np.zeros(3), np.zeros(3), steps=3)

    assert np.allclose(C, 0)


def test_diffusion_amplifies_neighborhood_of_single_hit() -> None:
    A = sparse.csr_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(2, 2), dtype=np.float32)
    C = diffuse(A, np.array([1.0, 0.0]), np.zeros(2), steps=3)

    assert C[1] > 0


def test_gate_respects_m() -> None:
    chunks = [_chunk(f"c{i}", f"chunk {i}", chunk_index=i) for i in range(5)]
    hits = thalamic_gate(
        chunks,
        L=np.array([0.1, 0.5, 0.2, 0.9, 0.3], dtype=np.float32),
        S=np.zeros(5, dtype=np.float32),
        C=np.zeros(5, dtype=np.float32),
        costs=np.zeros(5, dtype=np.float32),
        m=2,
        lambdas=(1.0, 0.0, 0.0, 0.0),
    )

    assert len(hits) == 2
    assert [hit.chunk.chunk_id for hit in hits] == ["c3", "c1"]


def test_cortical_end_to_end_on_billing_fixture() -> None:
    pipeline = RetrievalPipeline(
        retriever=_retriever(),
        reranker=HeuristicReranker(),
        bm25_candidates=20,
    )
    results = pipeline.search(
        "customer charged twice",
        [
            SearchDocument(
                file_name="tickets.csv",
                content=(
                    b"customer,issue,area\n"
                    b"Acme,customer charged twice,billing\n"
                    b"Beta,shipping delay,fulfillment\n"
                ),
                media_type="text/csv",
                metadata={"workspace": "demo"},
            )
        ],
        top_k=3,
    )

    assert results
    assert "charged twice" in results[0].content
    assert results[0].spans
    assert results[0].metadata["retriever"] == "cortical"


def test_cortical_degrades_to_trie_only_when_projection_missing(tmp_path) -> None:
    projection = load_projection(
        tmp_path / "missing.npz",
        ProjectionConfig(hash_features=16, dim=4),
    )
    chunks = [
        _chunk("billing", "customer charged twice", chunk_index=0),
        _chunk("shipping", "shipping delay", chunk_index=1),
    ]

    hits = _retriever(projection).search(
        "customer charged twice",
        chunks,
        top_k=5,
        workspace_id="ws",
    )

    assert hits
    assert hits[0].chunk.chunk_id == "billing"
    assert hits[0].chunk.metadata["spans"]


def test_service_builds_cortical_retriever_with_gate_budget(tmp_path) -> None:
    settings = Settings(
        waver_retriever="cortical",
        projection_model_path=tmp_path / "missing.npz",
        projection_hash_features=16,
        projection_dim=4,
        waver_gating_m=12,
    )

    retriever = _build_retriever(settings, BM25IndexCache())

    assert isinstance(retriever, CorticalRetriever)
    assert retriever.gating_m == 12
    assert _pipeline_candidate_budget(settings) == 12
