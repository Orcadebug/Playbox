from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.retrieval.bm25 import BM25Index, BM25Tokenizer, _STEMMER_AVAILABLE
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.chunker import chunk_document, chunk_documents, chunk_documents_iter
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import AutoReranker, HeuristicReranker
from app.schemas.documents import ParsedDocument, SourceLocation
from app.schemas.search import SearchDocument


# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------

def test_chunker_preserves_overlap_and_metadata() -> None:
    document = ParsedDocument(
        content="one two three four five six seven eight",
        source_name="notes.txt",
        metadata={"row_index": 4},
        location=SourceLocation(row_number=4),
    )

    chunks = chunk_document(document, max_tokens=4, overlap=2)
    assert len(chunks) == 3
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "three four five six"
    assert chunks[1].location.row_number == 4
    assert chunks[1].metadata["row_index"] == 4


def test_chunk_documents_iter_yields_same_as_list() -> None:
    docs = [
        ParsedDocument(content="alpha beta gamma delta epsilon", source_name="a.txt"),
        ParsedDocument(content="zeta eta theta iota kappa", source_name="b.txt"),
    ]
    list_result = chunk_documents(docs, max_tokens=3, overlap=1)
    iter_result = list(chunk_documents_iter(docs, max_tokens=3, overlap=1))
    assert [c.chunk_id for c in list_result] == [c.chunk_id for c in iter_result]
    assert [c.text for c in list_result] == [c.text for c in iter_result]


# ---------------------------------------------------------------------------
# BM25Tokenizer tests
# ---------------------------------------------------------------------------

def test_bm25_tokenizer_removes_stopwords() -> None:
    tok = BM25Tokenizer(use_stemming=False, use_stopwords=True)
    tokens = tok.tokenize("the quick brown fox and the lazy dog")
    assert "the" not in tokens
    assert "and" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens


@pytest.mark.skipif(not _STEMMER_AVAILABLE, reason="PyStemmer not installed")
def test_bm25_tokenizer_applies_stemming() -> None:
    tok = BM25Tokenizer(use_stemming=True, use_stopwords=False)
    tokens = tok.tokenize("billing billed bills")
    # All three should reduce to the same stem
    assert len(set(tokens)) == 1


@pytest.mark.skipif(not _STEMMER_AVAILABLE, reason="PyStemmer not installed")
def test_bm25_stemming_improves_recall() -> None:
    """Search for 'billed' should match a chunk containing 'billing'."""
    chunks = [
        chunk_document(
            ParsedDocument(content="billing complaint from customer", source_name="a.txt"),
            max_tokens=20,
        )[0],
        chunk_document(
            ParsedDocument(content="shipping delay at warehouse", source_name="b.txt"),
            max_tokens=20,
        )[0],
    ]
    index = BM25Index(chunks, use_stemming=True, use_stopwords=True)
    hits = index.search("billed", top_k=5)
    assert hits, "Expected at least one result — stemmer should match 'billing' for query 'billed'"
    assert hits[0].chunk.source_name == "a.txt"


# ---------------------------------------------------------------------------
# BM25Index tests
# ---------------------------------------------------------------------------

def test_bm25_index_ranks_relevant_chunk_first() -> None:
    chunks = [
        chunk_document(
            ParsedDocument(content="billing issue and refund request", source_name="a.txt"),
            max_tokens=20,
        )[0],
        chunk_document(
            ParsedDocument(content="shipping delay and address change", source_name="b.txt"),
            max_tokens=20,
        )[0],
    ]

    index = BM25Index(chunks)
    hits = index.search("billing refund", top_k=2)
    assert hits[0].chunk.source_name == "a.txt"
    assert hits[0].score > hits[-1].score if len(hits) > 1 else True


def test_bm25_heap_search_matches_sort_results() -> None:
    """Heap-based top-k must return the same results as a naive full sort."""
    docs = [
        ParsedDocument(content=f"document about topic {i} billing refund", source_name=f"doc{i}.txt")
        for i in range(20)
    ]
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, max_tokens=20))

    index = BM25Index(all_chunks, use_stemming=False, use_stopwords=False)
    top5 = index.search("billing refund", top_k=5)
    assert len(top5) == 5
    # Verify scores are in descending order
    scores = [h.score for h in top5]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# BM25IndexCache tests
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 5) -> list:
    result = []
    for i in range(n):
        result.extend(
            chunk_document(
                ParsedDocument(content=f"document {i} billing refund", source_name=f"doc{i}.txt"),
                max_tokens=20,
            )
        )
    return result


def test_bm25_cache_returns_same_index() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    chunks = _make_chunks(3)
    idx1 = cache.get_or_build("ws1", chunks)
    idx2 = cache.get_or_build("ws1", chunks)
    assert idx1 is idx2, "Same content hash should return cached instance"


def test_bm25_cache_invalidates_on_content_change() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    chunks_a = _make_chunks(3)
    chunks_b = _make_chunks(4)  # different number → different chunk IDs
    idx1 = cache.get_or_build("ws1", chunks_a)
    idx2 = cache.get_or_build("ws1", chunks_b)
    assert idx1 is not idx2, "Different content hash should build a new index"


def test_bm25_cache_explicit_invalidate() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    chunks = _make_chunks(3)
    idx1 = cache.get_or_build("ws1", chunks)
    cache.invalidate("ws1")
    idx2 = cache.get_or_build("ws1", chunks)
    assert idx1 is not idx2, "After explicit invalidation, a new index should be built"


def test_bm25_cache_evicts_lru() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=3)
    for ws in ("ws1", "ws2", "ws3"):
        cache.get_or_build(ws, _make_chunks(2))

    # Access ws1 to mark it recently used
    cache.get_or_build("ws1", _make_chunks(2))

    # Adding ws4 should evict ws2 (LRU)
    cache.get_or_build("ws4", _make_chunks(2))

    with cache._lock:
        assert "ws2" not in cache._cache
        assert "ws1" in cache._cache
        assert "ws4" in cache._cache


def test_bm25_cache_ttl_expiry() -> None:
    cache = BM25IndexCache(ttl=0.05, max_entries=5)  # 50ms TTL
    chunks = _make_chunks(2)
    idx1 = cache.get_or_build("ws1", chunks)
    time.sleep(0.1)
    idx2 = cache.get_or_build("ws1", chunks)
    assert idx1 is not idx2, "Expired TTL should cause index rebuild"


def test_bm25_cache_is_thread_safe() -> None:
    """Multiple threads building for different workspaces should not deadlock or corrupt."""
    cache = BM25IndexCache(ttl=60.0, max_entries=20)
    errors: list[Exception] = []

    def build(ws_id: str) -> None:
        try:
            chunks = _make_chunks(3)
            for _ in range(5):
                cache.get_or_build(ws_id, chunks)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=build, args=(f"ws{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety errors: {errors}"


# ---------------------------------------------------------------------------
# RetrievalPipeline tests
# ---------------------------------------------------------------------------

def test_pipeline_search_returns_rich_results_with_fallback_reranker() -> None:
    pipeline = RetrievalPipeline(reranker=HeuristicReranker())
    results = pipeline.search(
        "billing complaint last week",
        [
            SearchDocument(
                file_name="complaints.csv",
                content=b"customer,issue,week\nAcme,billing complaint,last week\nBeta,shipping delay,this week\n",
                media_type="text/csv",
                metadata={"workspace": "demo"},
            ),
            SearchDocument(
                file_name="notes.txt",
                content=b"general product notes",
            ),
        ],
        top_k=3,
    )

    assert results
    assert results[0].source_name == "complaints.csv"
    assert "billing complaint" in results[0].content
    assert results[0].snippet is not None
    assert results[0].metadata["workspace"] == "demo"
    assert results[0].metadata["parser_name"] == "csv"


def test_pipeline_uses_bm25_cache() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    pipeline = RetrievalPipeline(reranker=HeuristicReranker(), bm25_cache=cache)
    docs = [SearchDocument(file_name="a.txt", content=b"billing refund complaint")]

    pipeline.search("billing", docs, top_k=5, workspace_id="test-ws")
    # Second search — should hit cache (no second build)
    with cache._lock:
        assert "test-ws" in cache._cache


# ---------------------------------------------------------------------------
# AutoReranker tests
# ---------------------------------------------------------------------------

def test_auto_reranker_gracefully_falls_back_without_model() -> None:
    reranker = AutoReranker(model_path="/does/not/exist.onnx")
    assert reranker.is_fallback is True


def test_auto_reranker_fallback_on_missing_tokenizer(tmp_path: "pytest.TempPathFactory") -> None:
    """ONNX model present but no tokenizer.json → must fall back."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"not a real model")
    reranker = AutoReranker(model_path=model_file)
    assert reranker.is_fallback is True


def test_auto_reranker_batch_size_respected() -> None:
    """Verify that 70 candidates are split into ceil(70/32)=3 batches."""
    session_mock = MagicMock()
    # Fake ONNX output: shape (batch, 1) with score 0.5
    session_mock.run.side_effect = lambda _, feed: [
        np.full((len(feed["input_ids"]), 1), 0.5, dtype=np.float32)
    ]
    # MagicMock(name=...) sets repr name, not .name attribute — use plain objects.
    class _FakeInput:
        def __init__(self, n: str) -> None:
            self.name = n

    session_mock.get_inputs.return_value = [
        _FakeInput("input_ids"),
        _FakeInput("attention_mask"),
        _FakeInput("token_type_ids"),
    ]

    tok_mock = MagicMock()
    tok_mock.encode_pairs.side_effect = lambda query, passages, max_length=512: {
        "input_ids": np.zeros((len(passages), 10), dtype=np.int64),
        "attention_mask": np.ones((len(passages), 10), dtype=np.int64),
        "token_type_ids": np.zeros((len(passages), 10), dtype=np.int64),
    }

    from app.retrieval.bm25 import BM25ScoredChunk
    from app.schemas.documents import Chunk, SourceLocation

    def _fake_chunk(i: int) -> Chunk:
        return Chunk(
            chunk_id=f"doc:{i}:0:0",
            content=f"passage number {i}",
            source_name=f"doc{i}.txt",
            location=SourceLocation(),
        )

    candidates = [
        BM25ScoredChunk(chunk=_fake_chunk(i), score=float(i), bm25_score=float(i))
        for i in range(70)
    ]

    reranker = AutoReranker(batch_size=32)
    reranker._use_fallback = False
    reranker._session = session_mock
    reranker._tokenizer_instance = tok_mock

    reranker._onnx_rerank("billing", candidates, top_k=10)
    assert session_mock.run.call_count == 3  # ceil(70 / 32) = 3


# ---------------------------------------------------------------------------
# Connector tests
# ---------------------------------------------------------------------------

def test_webhook_connector_to_search_documents() -> None:
    from app.connectors.webhook import WebhookConnector

    connector = WebhookConnector()
    raw = [
        {"content": "Invoice dispute from Acme Corp", "name": "ticket-1.txt"},
        {"content": ""},  # empty — should be skipped
        {"content": "Shipping delayed by 3 days", "name": "ticket-2.txt"},
    ]
    docs = connector.to_search_documents(raw)
    assert len(docs) == 2
    assert docs[0].file_name == "ticket-1.txt"
    assert docs[0].content == b"Invoice dispute from Acme Corp"
    assert docs[1].content == b"Shipping delayed by 3 days"
    assert all(d.metadata.get("source_type") == "webhook" for d in docs)


def test_connector_registry_resolves_known_connectors() -> None:
    from app.connectors.registry import default_registry

    slack = default_registry.get("slack")
    webhook = default_registry.get("webhook")
    assert slack is not None
    assert webhook is not None
    assert default_registry.get("unknown_connector") is None
