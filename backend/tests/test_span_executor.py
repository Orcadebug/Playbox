from __future__ import annotations

import numpy as np

from app.parsers.base import ParserDetector, build_default_parser_registry
from app.parsers.plaintext import PlainTextParser
from app.retrieval.planner import QueryPlan
from app.retrieval.reranker import HeuristicReranker
from app.retrieval.source_executor import build_source_windows_from_documents
from app.retrieval.span_executor import SpanExecutor
from app.retrieval.sparse_projection import DeterministicSemanticProjection, ProjectionConfig
from app.schemas.search import SearchDocument


def _executor() -> SpanExecutor:
    detector = ParserDetector(
        registry=build_default_parser_registry(),
        default_parser=PlainTextParser(),
    )
    return SpanExecutor(
        parser_detector=detector,
        reranker=HeuristicReranker(),
        projection=DeterministicSemanticProjection(ProjectionConfig(hash_features=64, dim=16)),
    )


def _isolated_executor(
    *,
    enabled_channels: tuple[str, ...],
    rerank_enabled: bool = False,
) -> SpanExecutor:
    detector = ParserDetector(
        registry=build_default_parser_registry(),
        default_parser=PlainTextParser(),
    )
    return SpanExecutor(
        parser_detector=detector,
        reranker=HeuristicReranker(),
        projection=DeterministicSemanticProjection(ProjectionConfig(hash_features=64, dim=16)),
        enabled_channels=enabled_channels,
        rerank_enabled=rerank_enabled,
    )


def test_span_executor_returns_exact_phase_results_with_channels() -> None:
    executor = _executor()
    documents = [
        SearchDocument(
            file_name="notes.txt",
            content="Acme flagged a billing complaint after duplicate charges.",
            media_type="text/plain",
            metadata={"source_id": "raw-1", "source_origin": "raw", "source_type": "raw"},
        )
    ]

    outcome = executor.execute(
        query="billing complaint",
        documents=documents,
        top_k=3,
    )

    assert outcome.exact_results
    top = outcome.exact_results[0]
    assert top.metadata["phase"] == "exact"
    assert "exact" in top.metadata["channels"]
    assert top.channels == top.metadata["channels"]
    assert top.channel_scores["exact"] > 0
    assert top.primary_span is not None
    assert top.primary_span["highlights"]
    assert top.primary_span["offset_basis"] == "source"


def test_span_executor_can_isolate_exact_channel_without_rerank() -> None:
    executor = _isolated_executor(enabled_channels=("exact",), rerank_enabled=False)
    documents = [
        SearchDocument(
            file_name="notes.txt",
            content="Acme logged literal marker orchid relay for review.",
            media_type="text/plain",
            metadata={"source_id": "raw-1", "source_origin": "raw", "source_type": "raw"},
        )
    ]

    outcome = executor.execute(query="orchid relay", documents=documents, top_k=3)

    assert outcome.execution["active_channels"] == ["exact"]
    assert outcome.execution["phase_counts"]["reranked_results"] == 0
    assert outcome.final_results == outcome.proxy_results
    assert "exact" in outcome.final_results[0].channels
    assert "semantic" not in outcome.final_results[0].channels


def test_span_executor_semantic_only_does_not_backfill_exact_highlights() -> None:
    executor = _isolated_executor(enabled_channels=("semantic",), rerank_enabled=False)
    documents = [
        SearchDocument(
            file_name="ticket.txt",
            content="Customer wrote avatar crop issue; agent called it image trim.",
            media_type="text/plain",
            metadata={"source_id": "raw-semantic", "source_origin": "raw", "source_type": "raw"},
        )
    ]

    outcome = executor.execute(query="avatar crop issue", documents=documents, top_k=3)

    assert outcome.execution["active_channels"] == ["semantic"]
    assert outcome.proxy_results
    span = outcome.proxy_results[0].primary_span
    assert span is not None
    assert span["highlights"] == []
    assert outcome.proxy_results[0].spans is None


def test_span_executor_non_exact_candidates_use_context_span() -> None:
    executor = _executor()
    documents = [
        SearchDocument(
            file_name="issues.csv",
            content=b"customer,issue\nAcme,shipping delay\nBeta,warehouse hold\n",
            media_type="text/csv",
            metadata={"source_id": "stored-1", "source_origin": "stored", "source_type": "upload"},
        )
    ]

    outcome = executor.execute(
        query="nonexistent phrase",
        documents=documents,
        top_k=2,
    )

    assert outcome.proxy_results
    span = outcome.proxy_results[0].primary_span
    assert span is not None
    assert span["highlights"] == []
    assert "structure" in outcome.proxy_results[0].metadata["channels"]


def test_span_executor_semantic_fallback_finds_fresh_raw_without_projection() -> None:
    executor = _executor()
    documents = [
        SearchDocument(
            file_name="ticket.txt",
            content="Customer was billed twice and asked support for help.",
            media_type="text/plain",
            metadata={"source_id": "raw-semantic", "source_origin": "raw", "source_type": "raw"},
        )
    ]

    outcome = executor.execute(
        query="duplicate charge",
        documents=documents,
        top_k=3,
    )

    assert outcome.proxy_results
    top = outcome.proxy_results[0]
    assert "semantic" in top.channels
    assert top.channel_scores["semantic"] > 0
    assert top.primary_span is not None
    assert top.primary_span["highlights"] == []


def test_span_offset_basis_is_source_for_plaintext_with_leading_space() -> None:
    executor = _executor()
    content = "  alpha line\n  Billing Complaint arrived"
    documents = [
        SearchDocument(
            file_name="notes.txt",
            content=content,
            media_type="text/plain",
            metadata={"source_id": "plain", "source_origin": "raw", "source_type": "raw"},
        )
    ]

    outcome = executor.execute(query="billing complaint", documents=documents, top_k=1)

    span = outcome.final_results[0].primary_span
    assert span is not None
    assert span["offset_basis"] == "source"
    assert content[span["source_start"] : span["source_end"]] == "Billing Complaint"


def test_transformed_parsers_mark_span_offsets_as_parsed() -> None:
    executor = _executor()
    documents = [
        SearchDocument(
            file_name="rows.csv",
            content=b"customer,issue\nAcme,billing complaint\n",
            media_type="text/csv",
            metadata={"source_id": "csv", "source_origin": "raw", "source_type": "raw"},
        ),
        SearchDocument(
            file_name="events.ndjson",
            content=b'{"ticket":"A","issue":"billing complaint"}\n',
            media_type="application/json",
            metadata={"source_id": "ndjson", "source_origin": "raw", "source_type": "raw"},
        ),
        SearchDocument(
            file_name="items.json",
            content=b'[{"ticket":"B","issue":"billing complaint"}]',
            media_type="application/json",
            metadata={"source_id": "json", "source_origin": "raw", "source_type": "raw"},
        ),
    ]

    outcome = executor.execute(query="billing complaint", documents=documents, top_k=5)
    spans_by_source = {
        result.metadata["source_id"]: result.primary_span for result in outcome.final_results
    }

    assert spans_by_source["csv"]["offset_basis"] == "parsed"
    assert spans_by_source["ndjson"]["offset_basis"] == "parsed"
    assert spans_by_source["json"]["offset_basis"] == "parsed"
    for result in outcome.final_results:
        span = result.primary_span
        assert span is not None
        assert (
            result.content[span["source_start"] : span["source_end"]].lower() == "billing complaint"
        )


def test_span_executor_respects_rerank_budget_and_partial_flags() -> None:
    executor = _executor()
    documents = [
        SearchDocument(
            file_name=f"note-{idx}.txt",
            content=f"Document {idx} has billing issue and refund workflow details",
            media_type="text/plain",
            metadata={
                "source_id": f"src-{idx}",
                "source_origin": "stored",
                "source_type": "upload",
            },
        )
        for idx in range(1_500)
    ]

    outcome = executor.execute(
        query="billing issue",
        documents=documents,
        top_k=5,
        cacheable=True,
    )

    assert outcome.execution["shortlisted_candidates"] <= outcome.execution["rerank_limit"]
    assert len(outcome.reranked_results) <= 5
    assert outcome.execution["partial"] is True
    assert outcome.execution["window_count"] >= outcome.execution["scanned_windows"]
    assert outcome.final_results[0].metadata["retrieval_partial"] is True
    assert outcome.final_results[0].metadata["completion_reason"] == "partial_scan_limit"


def test_span_executor_applies_mrl_pruning_before_rerank(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    class FakeMrlProjection:
        def encode_query(self, query: str):  # type: ignore[no-untyped-def]
            return query

        def encode_chunks(self, chunks):  # type: ignore[no-untyped-def]
            return list(chunks)

        def score(self, query_vec, chunk_vecs):  # type: ignore[no-untyped-def]
            del query_vec
            return [float(len(chunk.text.split())) for chunk in chunk_vecs]

    def fake_build_query_plan(**kwargs):  # type: ignore[no-untyped-def]
        return QueryPlan(
            tier="huge",
            window_tier="huge",
            byte_tier="tiny",
            channels=("exact", "semantic", "structure"),
            window_count=10,
            bytes_loaded=int(kwargs.get("bytes_loaded", 0)),
            scan_budget_bytes=1024,
            scan_limit=10,
            candidate_limit=10,
            rerank_limit=3,
            partial=False,
            metadata_prefilter=True,
            use_cache=False,
        )

    monkeypatch.setattr("app.retrieval.span_executor.build_query_plan", fake_build_query_plan)
    executor = SpanExecutor(
        parser_detector=ParserDetector(
            registry=build_default_parser_registry(),
            default_parser=PlainTextParser(),
        ),
        reranker=HeuristicReranker(),
        projection=DeterministicSemanticProjection(ProjectionConfig(hash_features=64, dim=16)),
        mrl_projection=FakeMrlProjection(),
        rerank_enabled=False,
    )
    documents = [
        SearchDocument(
            file_name=f"note-{idx}.txt",
            content=("billing issue " + "detail " * idx + "refund workflow").strip(),
            media_type="text/plain",
            metadata={
                "source_id": f"src-{idx}",
                "source_origin": "stored",
                "source_type": "upload",
            },
        )
        for idx in range(1, 11)
    ]

    outcome = executor.execute(query="billing issue", documents=documents, top_k=3)

    assert outcome.execution["mrl_applied"] is True
    assert outcome.execution["shortlisted_candidates"] == 3
    assert outcome.execution["mrl_input_count"] == 10
    assert outcome.execution["mrl_output_count"] == 3


def test_span_executor_applies_stage0_before_candidate_scoring(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class RecordingProjection:
        def __init__(self) -> None:
            self.chunk_count = 0

        def encode_query(self, query: str):  # type: ignore[no-untyped-def]
            return np.ones(4, dtype=np.float32)

        def encode_chunks(self, chunks):  # type: ignore[no-untyped-def]
            self.chunk_count = len(chunks)
            return np.ones((len(chunks), 4), dtype=np.float32)

        def score(self, query_vec, chunk_vecs):  # type: ignore[no-untyped-def]
            del query_vec
            return np.ones(len(chunk_vecs), dtype=np.float32)

    def fake_build_query_plan(**kwargs):  # type: ignore[no-untyped-def]
        return QueryPlan(
            tier="huge",
            window_tier="huge",
            byte_tier="tiny",
            channels=("exact", "semantic", "structure"),
            window_count=int(kwargs["window_count"]),
            bytes_loaded=int(kwargs.get("bytes_loaded", 0)),
            scan_budget_bytes=1024,
            scan_limit=2,
            candidate_limit=10,
            rerank_limit=5,
            partial=True,
            metadata_prefilter=True,
            use_cache=False,
        )

    def fake_prefilter(query, windows, *, limit):  # type: ignore[no-untyped-def]
        del query
        return list(windows[:limit]), "rust"

    monkeypatch.setattr("app.retrieval.span_executor.build_query_plan", fake_build_query_plan)
    monkeypatch.setattr(
        "app.retrieval.span_executor._prefilter_windows_with_backend",
        fake_prefilter,
    )
    projection = RecordingProjection()
    executor = SpanExecutor(
        parser_detector=ParserDetector(
            registry=build_default_parser_registry(),
            default_parser=PlainTextParser(),
        ),
        reranker=HeuristicReranker(),
        projection=projection,
        rerank_enabled=False,
    )
    documents = [
        SearchDocument(
            file_name=f"note-{idx}.txt",
            content=f"billing issue document {idx}",
            media_type="text/plain",
            metadata={
                "source_id": f"src-{idx}",
                "source_origin": "stored",
                "source_type": "upload",
            },
        )
        for idx in range(5)
    ]

    outcome = executor.execute(query="billing issue", documents=documents, top_k=2)

    assert projection.chunk_count == 2
    assert outcome.execution["stage0_applied"] is True
    assert outcome.execution["stage0_backend"] == "rust"
    assert outcome.execution["stage0_candidates"] == 2


def test_span_executor_caps_mrl_input_and_rerank_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class RecordingMrlProjection:
        def __init__(self) -> None:
            self.input_count = 0

        def encode_query(self, query: str):  # type: ignore[no-untyped-def]
            return np.ones(16, dtype=np.float32)

        def encode_chunks(self, chunks):  # type: ignore[no-untyped-def]
            self.input_count = len(chunks)
            return np.ones((len(chunks), 16), dtype=np.float32)

        def score(self, query_vec, chunk_vecs):  # type: ignore[no-untyped-def]
            del query_vec
            return np.arange(len(chunk_vecs), dtype=np.float32)

    class RecordingReranker:
        def __init__(self) -> None:
            self.input_count = 0

        def rerank(self, query, candidates, top_k):  # type: ignore[no-untyped-def]
            del query
            self.input_count = len(candidates)
            return list(candidates)[:top_k]

    def fake_build_query_plan(**kwargs):  # type: ignore[no-untyped-def]
        return QueryPlan(
            tier="huge",
            window_tier="huge",
            byte_tier="tiny",
            channels=("exact", "semantic", "structure"),
            window_count=int(kwargs["window_count"]),
            bytes_loaded=int(kwargs.get("bytes_loaded", 0)),
            scan_budget_bytes=1024,
            scan_limit=int(kwargs["window_count"]),
            candidate_limit=5_000,
            rerank_limit=50,
            partial=False,
            metadata_prefilter=True,
            use_cache=False,
        )

    monkeypatch.setattr("app.retrieval.span_executor.build_query_plan", fake_build_query_plan)
    mrl_projection = RecordingMrlProjection()
    reranker = RecordingReranker()
    executor = SpanExecutor(
        parser_detector=ParserDetector(
            registry=build_default_parser_registry(),
            default_parser=PlainTextParser(),
        ),
        reranker=reranker,  # type: ignore[arg-type]
        projection=DeterministicSemanticProjection(ProjectionConfig(hash_features=64, dim=16)),
        mrl_projection=mrl_projection,  # type: ignore[arg-type]
    )
    documents = [
        SearchDocument(
            file_name=f"note-{idx}.txt",
            content=f"billing issue refund workflow {idx}",
            media_type="text/plain",
            metadata={
                "source_id": f"src-{idx}",
                "source_origin": "stored",
                "source_type": "upload",
            },
        )
        for idx in range(5_100)
    ]

    outcome = executor.execute(query="billing issue", documents=documents, top_k=3)

    assert mrl_projection.input_count == 5_000
    assert reranker.input_count == 50
    assert outcome.execution["mrl_input_count"] == 5_000
    assert outcome.execution["mrl_output_count"] == 50


def test_source_windows_keep_neighbor_ids_for_span_expansion() -> None:
    documents = [
        SearchDocument(
            file_name="plain.txt",
            content="line one\nline two\nline three",
            media_type="text/plain",
            metadata={"source_id": "plain-1", "source_origin": "raw", "source_type": "raw"},
        )
    ]

    windows = build_source_windows_from_documents(documents)
    assert len(windows) == 3
    assert windows[1].neighboring_window_ids == [windows[0].window_id, windows[2].window_id]
