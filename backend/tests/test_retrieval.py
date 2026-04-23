from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import sparse

from app.config import Settings
from app.retrieval.bm25 import _STEMMER_AVAILABLE, BM25Index, BM25ScoredChunk, BM25Tokenizer
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.chunker import chunk_document, chunk_documents, chunk_documents_iter
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_patterns import build_query_patterns
from app.retrieval.reranker import AutoReranker, HeuristicReranker
from app.retrieval.retriever import Bm25Retriever
from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    MrlProjection,
    ProjectionConfig,
    SparseProjection,
    SpladeProjection,
    load_best_projection,
    load_projection,
    save_projection,
)
from app.retrieval.trie import Pattern, QueryTrie
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


def test_chunker_tracks_exact_offsets_and_preserves_source_whitespace() -> None:
    content = "alpha   beta\ngamma delta"
    document = ParsedDocument(content=content, source_name="notes.txt")

    chunks = chunk_document(document, max_tokens=2, overlap=1)

    assert chunks[0].text == "alpha   beta"
    assert chunks[0].metadata["source_start"] == 0
    assert chunks[0].metadata["source_end"] == len("alpha   beta")
    assert chunks[1].text == "beta\ngamma"
    assert chunks[1].metadata["source_start"] == content.index("beta")
    assert chunks[1].metadata["source_end"] == content.index("gamma") + len("gamma")


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
        ParsedDocument(
            content=f"document about topic {i} billing refund",
            source_name=f"doc{i}.txt",
        )
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
# Retriever protocol tests
# ---------------------------------------------------------------------------

def test_retriever_protocol_bm25_default_unchanged() -> None:
    chunks = _make_chunks(4)
    direct = BM25Index(chunks).search("billing refund", top_k=3)
    via_retriever = Bm25Retriever().search(
        "billing refund",
        chunks,
        top_k=3,
        workspace_id="test-ws",
    )
    assert [hit.chunk.chunk_id for hit in via_retriever] == [
        hit.chunk.chunk_id for hit in direct
    ]
    assert [hit.score for hit in via_retriever] == [hit.score for hit in direct]


def test_bm25_retriever_rust_mode_falls_back_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = _make_chunks(4)
    direct = BM25Index(chunks).search("billing refund", top_k=3)
    monkeypatch.setattr("app.retrieval.retriever._get_rust_bm25_class", lambda: None)

    via_retriever = Bm25Retriever(rust_mode="rust").search(
        "billing refund",
        chunks,
        top_k=3,
        workspace_id="test-ws",
    )
    assert [hit.chunk.chunk_id for hit in via_retriever] == [
        hit.chunk.chunk_id for hit in direct
    ]


def test_bm25_retriever_real_rust_matches_python_ids_when_extension_installed() -> None:
    module = pytest.importorskip("waver_core")
    if not hasattr(module, "RustBm25Index"):
        pytest.skip("Rust BM25 class not exported")

    chunks = _sps_chunks([
        ("billing refund issue for customer", "hit.txt"),
        ("weather forecast tomorrow", "weather.txt"),
        ("shipping delay at warehouse", "ship.txt"),
    ])
    python_hits = Bm25Retriever(rust_mode="python").search(
        "billing refund",
        chunks,
        top_k=3,
        workspace_id="ws-rust-bm25",
    )
    rust_hits = Bm25Retriever(rust_mode="rust").search(
        "billing refund",
        chunks,
        top_k=3,
        workspace_id="ws-rust-bm25",
    )

    assert [hit.chunk.chunk_id for hit in rust_hits] == [hit.chunk.chunk_id for hit in python_hits]


def test_retriever_flag_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        Settings(waver_retriever="unknown")


# ---------------------------------------------------------------------------
# SpsRetriever tests — BM25 + semantic projection fusion
# ---------------------------------------------------------------------------

def _sps_chunks(contents: list[tuple[str, str]]):
    """Build one chunk per (content, source_name) pair."""
    chunks = []
    for content, name in contents:
        chunks.extend(
            chunk_document(ParsedDocument(content=content, source_name=name), max_tokens=20)
        )
    return chunks


def test_sps_retriever_recalls_synonym() -> None:
    from app.retrieval.sps import SpsRetriever

    chunks = _sps_chunks([
        ("customer charged duplicate", "charge.txt"),
        ("weather forecast tomorrow", "weather.txt"),
        ("refund issued successfully", "refund.txt"),
    ])
    retriever = SpsRetriever(
        projection=DeterministicSemanticProjection(),
        cache=None,
        alpha=0.4,
    )

    # "billed twice" has zero lexical overlap with any chunk — BM25 alone returns nothing.
    bm25_only = BM25Index(chunks).search("billed twice", top_k=3)
    assert not bm25_only

    hits = retriever.search("billed twice", chunks, top_k=3, workspace_id="ws-syn")
    assert hits, "SPS should surface the synonym chunk"
    assert hits[0].chunk.source_name == "charge.txt"


def test_sps_retriever_preserves_bm25_on_exact_match() -> None:
    from app.retrieval.sps import SpsRetriever

    chunks = _sps_chunks([
        ("billing refund issue for customer", "hit.txt"),
        ("unrelated weather report", "weather.txt"),
        ("shipping delay at warehouse", "ship.txt"),
    ])
    retriever = SpsRetriever(
        projection=DeterministicSemanticProjection(),
        cache=None,
        alpha=0.7,
    )
    hits = retriever.search("billing refund", chunks, top_k=3, workspace_id="ws-exact")
    assert hits[0].chunk.source_name == "hit.txt"


def test_sps_retriever_uses_embedding_cache() -> None:
    from app.retrieval.sps import SpsRetriever

    class _CountingProjection:
        def __init__(self) -> None:
            self._inner = DeterministicSemanticProjection()
            self.encode_chunks_calls = 0

        def encode_query(self, query):
            return self._inner.encode_query(query)

        def encode_chunks(self, chunks):
            self.encode_chunks_calls += 1
            return self._inner.encode_chunks(chunks)

        def score(self, qv, cv):
            return self._inner.score(qv, cv)

    chunks = _sps_chunks([
        ("customer charged duplicate", "a.txt"),
        ("refund issued", "b.txt"),
    ])
    proj = _CountingProjection()
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    retriever = SpsRetriever(projection=proj, cache=cache, alpha=0.5)  # type: ignore[arg-type]

    retriever.search("billed twice", chunks, top_k=2, workspace_id="ws-cache")
    retriever.search("double charge", chunks, top_k=2, workspace_id="ws-cache")

    assert proj.encode_chunks_calls == 1, (
        f"encode_chunks should be cached across queries; called {proj.encode_chunks_calls}x"
    )


def test_sps_retriever_empty_inputs() -> None:
    from app.retrieval.sps import SpsRetriever

    retriever = SpsRetriever(projection=DeterministicSemanticProjection(), cache=None)
    assert retriever.search("", _sps_chunks([("x", "a.txt")]), top_k=3, workspace_id="ws") == []
    assert retriever.search("query", [], top_k=3, workspace_id="ws") == []


def test_splade_projection_falls_back_when_artifact_missing(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = ProjectionConfig(hash_features=64, dim=16)
    projection = SpladeProjection(
        model_path=tmp_path / "missing.onnx",
        config=config,
        fallback=DeterministicSemanticProjection(config),
    )

    query_vec = projection.encode_query("billing refund")
    chunk_vecs = projection.encode_chunks(_sps_chunks([("duplicate charge", "ticket.txt")]))

    assert projection.using_fallback is True
    assert projection.fallback_reason == "splade_artifact_unavailable"
    assert query_vec.shape == (config.dim,)
    assert chunk_vecs.shape == (1, config.dim)


def test_splade_projection_records_runtime_fallback_when_artifact_present(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = ProjectionConfig(hash_features=64, dim=16)
    model_path = tmp_path / "splade" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"placeholder")
    projection = SpladeProjection(
        model_path=model_path,
        config=config,
        fallback=DeterministicSemanticProjection(config),
    )

    _ = projection.encode_query("billing refund")

    assert projection.using_fallback is True
    assert projection.fallback_reason == "splade_runtime_failed"


def test_load_best_projection_prefers_splade_artifact(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = ProjectionConfig(hash_features=64, dim=16)
    model_dir = tmp_path / "models"
    splade_model = model_dir / "splade" / "model.onnx"
    splade_model.parent.mkdir(parents=True, exist_ok=True)
    splade_model.write_bytes(b"placeholder")

    projection = load_best_projection(
        model_dir=model_dir,
        projection_model_path=None,
        config=config,
    )

    assert isinstance(projection, SpladeProjection)


def test_mrl_projection_records_runtime_fallback_when_artifact_present(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    config = ProjectionConfig(hash_features=64, dim=16)
    model_path = tmp_path / "mrl" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"placeholder")
    projection = MrlProjection(
        model_path=model_path,
        config=config,
        fallback=DeterministicSemanticProjection(config),
    )

    _ = projection.encode_query("billing refund")

    assert projection.using_fallback is True
    assert projection.fallback_reason == "mrl_runtime_failed"


# ---------------------------------------------------------------------------
# Negation / polarity feature tests
# ---------------------------------------------------------------------------

def test_negation_augmenter_injects_synthetic_tokens() -> None:
    from app.retrieval.sparse_projection import _augment_with_polarity

    out = _augment_with_polarity("the refund was denied by policy")
    assert "DENIED_refund" in out
    assert "DENIED_policy" in out
    assert "refund was denied by policy" in out  # original preserved


def test_negation_augmenter_noop_without_matches() -> None:
    from app.retrieval.sparse_projection import _augment_with_polarity

    text = "the cat sat on the mat"
    assert _augment_with_polarity(text) == text


def test_negation_separates_opposite_polarity() -> None:
    proj = DeterministicSemanticProjection()
    denied = proj.encode_query("the refund was denied")
    approved = proj.encode_query("the refund was approved")
    rejected = proj.encode_query("the refund was rejected")

    cos_opposite = float(denied @ approved)
    cos_same = float(denied @ rejected)

    assert cos_same > cos_opposite, (
        f"Expected denied~rejected ({cos_same:.3f}) > denied~approved ({cos_opposite:.3f})"
    )


def test_sentence_scorer_selects_most_relevant_sentence() -> None:
    from app.retrieval.sentence_scorer import find_best_sentence

    chunk_text = (
        "The weather report mentioned rain for tomorrow. "
        "The second refund was denied due to policy violations. "
        "Shipping updates will follow later this week."
    )
    proj = DeterministicSemanticProjection()
    result = find_best_sentence("refund denied", chunk_text, proj)
    assert result is not None
    start, end = result
    snippet = chunk_text[start:end]
    assert "denied" in snippet.lower()
    assert "refund" in snippet.lower()


def test_sentence_scorer_noop_for_single_sentence() -> None:
    from app.retrieval.sentence_scorer import find_best_sentence

    proj = DeterministicSemanticProjection()
    assert find_best_sentence("anything", "one short sentence here", proj) is None


def test_sentence_scorer_preserves_offsets() -> None:
    from app.retrieval.sentence_scorer import find_best_sentence

    chunk_text = "First unrelated line.\nRefund was denied for policy reason.\nThird line."
    proj = DeterministicSemanticProjection()
    result = find_best_sentence("refund denied", chunk_text, proj)
    assert result is not None
    start, end = result
    assert chunk_text[start:end] == "Refund was denied for policy reason."


def test_exact_phrase_retriever_finds_literal_substring() -> None:
    from app.retrieval.exact_phrase import ExactPhraseRetriever

    chunks = _sps_chunks([
        ("system reported error code 503 gateway timeout", "log1.txt"),
        ("unrelated weather forecast", "weather.txt"),
    ])
    retriever = ExactPhraseRetriever()
    hits = retriever.search(
        "error code 503", chunks, top_k=5, workspace_id="ws-phrase"
    )
    assert hits
    assert hits[0].chunk.source_name == "log1.txt"
    assert "phrase" in hits[0].channels


def test_exact_phrase_retriever_real_rust_matches_python_when_extension_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = pytest.importorskip("waver_core")
    if not hasattr(module, "phrase_search"):
        pytest.skip("Rust phrase search is not exported")

    from app.retrieval.exact_phrase import ExactPhraseRetriever

    chunks = _sps_chunks([
        ("system reported error code 503 gateway timeout", "log1.txt"),
        ("system reported error code 404", "log2.txt"),
        ("unrelated weather forecast", "weather.txt"),
    ])
    rust_hits = ExactPhraseRetriever().search(
        "error code 503",
        chunks,
        top_k=5,
        workspace_id="ws-phrase-rust",
    )

    monkeypatch.setattr("app.retrieval.exact_phrase.get_attr", lambda name: None)
    python_hits = ExactPhraseRetriever().search(
        "error code 503",
        chunks,
        top_k=5,
        workspace_id="ws-phrase-rust",
    )

    assert [hit.chunk.chunk_id for hit in rust_hits] == [hit.chunk.chunk_id for hit in python_hits]


def test_multihead_rrf_union_combines_heads() -> None:
    from app.retrieval.exact_phrase import ExactPhraseRetriever
    from app.retrieval.retriever import Bm25Retriever, MultiHeadRetriever
    from app.retrieval.sps import SpsRetriever

    chunks = _sps_chunks([
        ("customer charged duplicate", "synonym.txt"),
        ("error code 503 gateway timeout", "phrase.txt"),
        ("billing refund issue", "lexical.txt"),
        ("unrelated shipping note", "noise.txt"),
    ])
    retriever = MultiHeadRetriever(
        heads=[
            ("sps", SpsRetriever(projection=DeterministicSemanticProjection(), cache=None)),
            ("bm25", Bm25Retriever()),
            ("phrase", ExactPhraseRetriever()),
        ],
    )
    hits = retriever.search(
        "billed twice", chunks, top_k=4, workspace_id="ws-multi"
    )
    assert hits
    # Synonym chunk must be surfaced — pure BM25 would miss it.
    ids = [h.chunk.source_name for h in hits]
    assert "synonym.txt" in ids
    # Each hit has channels populated.
    assert all(h.channels for h in hits)


def _hit_signature(hits: list[BM25ScoredChunk]) -> list[tuple[object, ...]]:
    return [
        (
            hit.chunk.chunk_id,
            round(hit.score, 12),
            tuple(hit.channels),
            tuple((k, round(v, 12)) for k, v in sorted(hit.channel_scores.items())),
            None if hit.bm25_score is None else round(hit.bm25_score, 12),
        )
        for hit in hits
    ]


def test_multihead_rrf_rust_flag_falls_back_to_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.retrieval.exact_phrase import ExactPhraseRetriever
    from app.retrieval.retriever import Bm25Retriever, MultiHeadRetriever
    from app.retrieval.sps import SpsRetriever

    monkeypatch.setattr("app.retrieval.retriever._get_rust_rrf_callable", lambda: None)

    chunks = _sps_chunks([
        ("customer charged duplicate", "synonym.txt"),
        ("error code 503 gateway timeout", "phrase.txt"),
        ("billing refund issue", "lexical.txt"),
        ("unrelated shipping note", "noise.txt"),
    ])
    heads = [
        ("sps", SpsRetriever(projection=DeterministicSemanticProjection(), cache=None)),
        ("bm25", Bm25Retriever()),
        ("phrase", ExactPhraseRetriever()),
    ]
    baseline = MultiHeadRetriever(heads=heads).search(
        "billed twice", chunks, top_k=4, workspace_id="ws-rust-fallback"
    )
    rust_flag = MultiHeadRetriever(heads=heads, use_rust_rrf=True).search(
        "billed twice", chunks, top_k=4, workspace_id="ws-rust-fallback"
    )

    assert _hit_signature(rust_flag) == _hit_signature(baseline)


def test_multihead_rrf_uses_rust_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.retrieval.retriever import Bm25Retriever, MultiHeadRetriever

    calls: list[tuple[object, ...]] = []

    def _fake_rust_rrf(payload, top_k, rrf_k):
        calls.append((payload, top_k, rrf_k))
        chunk_id = payload[0][1][0][0]
        return [
            (
                chunk_id,
                42.0,
                ["rust"],
                {"rust": 7.0},
                None,
            )
        ]

    monkeypatch.setattr("app.retrieval.retriever._get_rust_rrf_callable", lambda: _fake_rust_rrf)

    chunks = _sps_chunks([
        ("billing refund issue", "lexical.txt"),
        ("customer charged duplicate", "synonym.txt"),
    ])
    retriever = MultiHeadRetriever(
        heads=[("bm25", Bm25Retriever())],
        use_rust_rrf=True,
    )
    hits = retriever.search(
        "billing refund", chunks, top_k=2, workspace_id="ws-rust-primary"
    )

    assert calls
    assert hits
    assert hits[0].score == 42.0
    assert hits[0].channels == ["rust"]


def test_multihead_rrf_shadow_logs_mismatch(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from app.retrieval.retriever import Bm25Retriever, MultiHeadRetriever

    def _fake_rust_rrf(payload, top_k, rrf_k):
        chunk_id = payload[0][1][0][0]
        return [
            (
                chunk_id,
                999.0,
                ["rust"],
                {"rust": 999.0},
                None,
            )
        ]

    monkeypatch.setattr("app.retrieval.retriever._get_rust_rrf_callable", lambda: _fake_rust_rrf)

    chunks = _sps_chunks([
        ("billing refund issue", "lexical.txt"),
        ("customer charged duplicate", "synonym.txt"),
    ])
    retriever = MultiHeadRetriever(
        heads=[("bm25", Bm25Retriever())],
        shadow_compare=True,
    )
    with caplog.at_level("WARNING", logger="app.retrieval.retriever"):
        hits = retriever.search(
            "billing refund", chunks, top_k=2, workspace_id="ws-rust-shadow"
        )

    assert hits
    assert "RRF parity mismatch" in caplog.text


def test_multihead_rrf_real_rust_matches_python_when_extension_installed() -> None:
    pytest.importorskip("waver_core")
    from app.retrieval.exact_phrase import ExactPhraseRetriever
    from app.retrieval.retriever import Bm25Retriever, MultiHeadRetriever
    from app.retrieval.sps import SpsRetriever

    chunks = _sps_chunks([
        ("customer charged duplicate", "synonym.txt"),
        ("error code 503 gateway timeout", "phrase.txt"),
        ("billing refund issue", "lexical.txt"),
        ("unrelated shipping note", "noise.txt"),
    ])
    heads = [
        ("sps", SpsRetriever(projection=DeterministicSemanticProjection(), cache=None)),
        ("bm25", Bm25Retriever()),
        ("phrase", ExactPhraseRetriever()),
    ]
    python_hits = MultiHeadRetriever(heads=heads).search(
        "billed twice", chunks, top_k=4, workspace_id="ws-rust-real"
    )
    rust_hits = MultiHeadRetriever(heads=heads, use_rust_rrf=True).search(
        "billed twice", chunks, top_k=4, workspace_id="ws-rust-real"
    )

    assert _hit_signature(rust_hits) == _hit_signature(python_hits)


def test_stage0_prefilter_real_rust_matches_python_when_extension_installed() -> None:
    module = pytest.importorskip("waver_core")
    if not hasattr(module, "prefilter_windows"):
        pytest.skip("Rust prefilter is not exported")

    from app.retrieval.source_executor import SourceWindow
    from app.retrieval.span_executor import _prefilter_windows, _python_prefilter_windows

    windows = [
        SourceWindow(
            window_id=f"w-{index}",
            record_id=f"r-{index}",
            source_id=f"s-{index}",
            source_name=f"doc-{index}.txt",
            source_type="raw",
            source_origin="raw",
            parser_name="plaintext",
            text=text,
        )
        for index, text in enumerate(
            [
                "heartbeat heartbeat heartbeat",
                "fatal timeout marker 9f2 observed in prod",
                "gateway timeout with error code 503",
                "shipping notice and unrelated update",
            ],
            start=1,
        )
    ]

    rust_hits = _prefilter_windows("fatal timeout marker 9f2", windows, limit=2)
    python_hits = _python_prefilter_windows("fatal timeout marker 9f2", windows, limit=2)

    assert [window.window_id for window in rust_hits] == [
        window.window_id for window in python_hits
    ]


def test_multihead_empty_chunks_returns_empty() -> None:
    from app.retrieval.retriever import MultiHeadRetriever

    retriever = MultiHeadRetriever(heads=[])
    assert retriever.search("x", [], top_k=5, workspace_id="ws") == []


def test_budget_hint_thorough_increases_candidate_limit() -> None:
    from app.retrieval.planner import build_query_plan

    baseline = build_query_plan(query="q", top_k=10, window_count=500)
    thorough = build_query_plan(
        query="q", top_k=10, window_count=500, budget_hint="thorough"
    )
    assert thorough.candidate_limit > baseline.candidate_limit
    assert thorough.rerank_limit >= baseline.rerank_limit


def test_budget_hint_fast_shrinks_candidate_limit() -> None:
    from app.retrieval.planner import build_query_plan

    baseline = build_query_plan(query="q", top_k=10, window_count=500)
    fast = build_query_plan(
        query="q", top_k=10, window_count=500, budget_hint="fast"
    )
    assert fast.candidate_limit < baseline.candidate_limit
    assert fast.rerank_limit < baseline.rerank_limit


def test_byte_budget_promotes_plan_to_huge_tier() -> None:
    from app.retrieval.planner import build_query_plan

    plan = build_query_plan(
        query="q",
        top_k=5,
        window_count=10,
        bytes_loaded=40 * 1024 * 1024,
    )

    assert plan.window_tier == "tiny"
    assert plan.byte_tier == "huge"
    assert plan.tier == "huge"
    assert plan.scan_budget_bytes <= 32 * 1024 * 1024


def test_negation_preserves_unrelated_text_similarity() -> None:
    proj = DeterministicSemanticProjection()
    a = proj.encode_query("the weather is sunny today")
    b = proj.encode_query("pizza delivery route optimization")
    cos = float(a @ b)
    # Unrelated text should be near-orthogonal (low cosine), augmenter must not
    # spuriously push them together.
    assert cos < 0.3


# ---------------------------------------------------------------------------
# Query trie tests
# ---------------------------------------------------------------------------

def _single_chunk(content: str):
    return chunk_document(ParsedDocument(content=content, source_name="doc.txt"), max_tokens=50)[0]


def test_trie_literal_token_hit() -> None:
    chunk = _single_chunk("Customer reported a Billing problem.")
    trie = QueryTrie([Pattern(id=0, body="billing", weight=1.0, kind="token")])

    hits = list(trie.scan(chunk))

    assert len(hits) == 1
    assert hits[0].chunk_id == chunk.chunk_id
    assert chunk.text[hits[0].start : hits[0].end] == "Billing"


def test_trie_phrase_scores_higher_than_tokens() -> None:
    trie = QueryTrie(build_query_patterns("billing complaint"))
    phrase_chunk = _single_chunk("The billing complaint arrived yesterday.")
    token_chunk = _single_chunk("The billing note followed a separate complaint.")

    assert trie.score_chunk(phrase_chunk) > trie.score_chunk(token_chunk)


def test_trie_overlapping_matches_merged() -> None:
    chunk = _single_chunk("billing complaint")
    trie = QueryTrie(build_query_patterns("billing complaint"))

    assert trie.spans_for_chunk(chunk) == [(0, len("billing complaint"))]


def test_trie_span_offsets_valid() -> None:
    chunk = _single_chunk("Acme opened a Billing Complaint yesterday.")
    patterns = build_query_patterns("billing complaint")
    trie = QueryTrie(patterns)
    pattern_by_id = {pattern.id: pattern for pattern in patterns}

    for hit in trie.scan(chunk):
        pattern = pattern_by_id[hit.pattern_id]
        assert chunk.text[hit.start : hit.end].lower() == pattern.body


def test_trie_phrase_requires_boundaries() -> None:
    trie = QueryTrie(build_query_patterns("billing complaint"))
    chunk = _single_chunk("rebilling complaintx")

    assert list(trie.scan(chunk)) == []
    assert trie.score_chunk(chunk) == 0.0


def test_trie_pattern_cap_enforced() -> None:
    patterns = build_query_patterns(" ".join(f"token{i}" for i in range(20)), max_patterns=5)

    assert len(patterns) == 5


# ---------------------------------------------------------------------------
# Sparse projection tests
# ---------------------------------------------------------------------------

def test_projection_loads_or_falls_back(tmp_path) -> None:
    config = ProjectionConfig(hash_features=16, dim=4)
    projection = load_projection(tmp_path / "missing.npz", config)

    assert isinstance(projection, DeterministicSemanticProjection)
    chunk_vecs = projection.encode_chunks([_single_chunk("billing refund")])
    scores = projection.score(projection.encode_query("billing"), chunk_vecs)
    assert scores[0] > 0.0


def test_deterministic_projection_matches_basic_aliases(tmp_path) -> None:
    projection = load_projection(
        tmp_path / "missing.npz",
        ProjectionConfig(hash_features=64, dim=32),
    )

    query_vec = projection.encode_query("duplicate charge")
    chunk_vecs = projection.encode_chunks(
        [_single_chunk("customer billed twice"), _single_chunk("shipping delay")]
    )
    scores = projection.score(query_vec, chunk_vecs)

    assert scores[0] > scores[1]


def test_projection_cosine_shape(tmp_path) -> None:
    config = ProjectionConfig(hash_features=16, dim=16)
    path = tmp_path / "projection.npz"
    save_projection(path, sparse.eye(config.hash_features, config.dim, format="csr"), config)
    projection = SparseProjection.load(path)

    query_vec = projection.encode_query("billing refund")
    chunk_vecs = projection.encode_chunks(
        [_single_chunk("billing refund request"), _single_chunk("shipping delay")]
    )
    scores = projection.score(query_vec, chunk_vecs)

    assert query_vec.shape == (config.dim,)
    assert chunk_vecs.shape == (2, config.dim)
    assert scores.shape == (2,)
    assert np.all(scores >= 0)


def test_projection_teacher_correlation() -> None:
    config = ProjectionConfig(hash_features=64, dim=64)
    projection = SparseProjection(
        sparse.eye(config.hash_features, config.dim, format="csr"),
        config,
    )
    query_vec = projection.encode_query("billing refund")
    chunk_vecs = projection.encode_chunks(
        [_single_chunk("billing refund request"), _single_chunk("shipping warehouse delay")]
    )
    scores = projection.score(query_vec, chunk_vecs)

    assert scores[0] > scores[1]


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


def test_bm25_cache_snapshot_is_stable_and_thread_safe() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    chunks = _make_chunks(3)

    assert cache.snapshot() == {}
    cache.get_or_build("ws1", chunks)
    first = cache.snapshot()
    second = cache.snapshot()

    assert first == second
    assert first["aliases"]["ws1"] in first["entries"]
    assert first["entries"][first["aliases"]["ws1"]]["chunk_count"] == len(chunks)


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
    with cache._lock:
        assert "ws1" not in cache._workspace_aliases
    idx2 = cache.get_or_build("ws1", chunks)
    assert idx1 is idx2, "Workspace invalidation should not evict reusable payload cache entries"


def test_bm25_cache_evicts_lru() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=3)
    cache.get_or_build("ws1", _make_chunks(1))
    cache.get_or_build("ws2", _make_chunks(2))
    cache.get_or_build("ws3", _make_chunks(3))

    # Access ws1 to mark it recently used
    cache.get_or_build("ws1", _make_chunks(1))

    # Adding ws4 should evict ws2 (LRU)
    cache.get_or_build("ws4", _make_chunks(4))

    with cache._lock:
        aliases = dict(cache._workspace_aliases)
        assert "ws2" not in aliases
        assert aliases["ws1"] in cache._cache
        assert aliases["ws4"] in cache._cache


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
                content=(
                    b"customer,issue,week\n"
                    b"Acme,billing complaint,last week\n"
                    b"Beta,shipping delay,this week\n"
                ),
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


def test_pipeline_returns_primary_span_with_exact_source_offsets() -> None:
    text = "Intro line.\nAcme filed a Billing Complaint after duplicate billing."
    pipeline = RetrievalPipeline(reranker=HeuristicReranker())

    results = pipeline.search(
        "billing complaint",
        [SearchDocument(file_name="notes.txt", content=text)],
        top_k=1,
    )

    span = results[0].primary_span
    assert span is not None
    assert span["text"] == "Billing Complaint"
    assert text[span["source_start"] : span["source_end"]] == "Billing Complaint"
    assert span["highlights"][0]["text"] == "Billing Complaint"
    assert results[0].matched_spans[0] == span


def test_pipeline_primary_span_carries_csv_row_location() -> None:
    pipeline = RetrievalPipeline(reranker=HeuristicReranker())

    results = pipeline.search(
        "billing complaint",
        [
            SearchDocument(
                file_name="complaints.csv",
                content=b"customer,issue\nAcme,billing complaint\nBeta,shipping delay\n",
                media_type="text/csv",
            )
        ],
        top_k=1,
    )

    span = results[0].primary_span
    assert span is not None
    assert span["text"] == "billing complaint"
    assert span["location"]["row_number"] == 2
    assert results[0].content[span["source_start"] : span["source_end"]] == "billing complaint"


def test_pipeline_returns_context_span_when_candidate_has_no_exact_match() -> None:
    class StaticRetriever:
        def search(self, query, chunks, top_k, workspace_id, use_cache=True):  # noqa: ANN001
            return [BM25ScoredChunk(chunk=list(chunks)[0], score=0.42)]

    pipeline = RetrievalPipeline(
        retriever=StaticRetriever(),
        reranker=HeuristicReranker(),
    )

    results = pipeline.search(
        "zebra",
        [SearchDocument(file_name="notes.txt", content="alpha beta gamma")],
        top_k=1,
    )

    span = results[0].primary_span
    assert span is not None
    assert span["text"] == "alpha beta gamma"
    assert span["highlights"] == []
    assert results[0].matched_spans == [span]


def test_pipeline_uses_bm25_cache() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    pipeline = RetrievalPipeline(reranker=HeuristicReranker(), bm25_cache=cache)
    docs = [SearchDocument(file_name="a.txt", content=b"billing refund complaint")]

    pipeline.search("billing", docs, top_k=5, workspace_id="test-ws")
    # Second search — should hit cache (no second build)
    with cache._lock:
        assert cache._workspace_aliases["test-ws"] in cache._cache


def test_pipeline_can_skip_bm25_cache_for_ephemeral_sources() -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    pipeline = RetrievalPipeline(reranker=HeuristicReranker(), bm25_cache=cache)
    docs = [SearchDocument(file_name="raw.txt", content=b"billing refund complaint")]

    pipeline.search("billing", docs, top_k=5, workspace_id="test-ws", use_cache=False)

    with cache._lock:
        assert cache._cache == {}


# ---------------------------------------------------------------------------
# AutoReranker tests
# ---------------------------------------------------------------------------

def test_auto_reranker_gracefully_falls_back_without_model() -> None:
    reranker = AutoReranker(model_path="/does/not/exist.onnx")
    assert reranker.is_fallback is True


def test_auto_reranker_fallback_on_missing_tokenizer(tmp_path: pytest.TempPathFactory) -> None:
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
