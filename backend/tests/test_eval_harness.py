from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.parsers.base import ParserDetector, build_default_parser_registry
from app.parsers.plaintext import PlainTextParser
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.eval_harness import (
    ColdCorpusGate,
    DeterministicEvalProjection,
    EvalCase,
    EvalRunConfig,
    ExpectedHit,
    build_run_config,
    evaluate_case,
    load_eval_cases,
    run_eval,
)
from app.retrieval.reranker import HeuristicReranker
from app.retrieval.span_executor import SpanExecutor
from app.retrieval.sparse_projection import NullProjection, ProjectionConfig


@pytest.fixture
def fixtures_dir() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "app"
        / "retrieval"
        / "eval_fixtures"
    )


def _executor() -> SpanExecutor:
    detector = ParserDetector(
        registry=build_default_parser_registry(),
        default_parser=PlainTextParser(),
    )
    return SpanExecutor(
        parser_detector=detector,
        reranker=HeuristicReranker(),
        projection=DeterministicEvalProjection(),
    )


def test_load_eval_cases_filters_by_profile(fixtures_dir: Path) -> None:
    smoke_cases = load_eval_cases(fixtures_dir, "smoke")
    regression_cases = load_eval_cases(fixtures_dir, "regression")

    assert smoke_cases
    assert regression_cases
    assert len(regression_cases) >= len(smoke_cases)
    assert all("smoke" in case.profiles for case in smoke_cases)


def test_load_eval_cases_rejects_invalid_schema(tmp_path: Path) -> None:
    bad_payload = {
        "pack_id": "bad",
        "cases": [
            {
                "query": "missing id",
                "profiles": ["smoke"],
                "documents": [{"source_id": "x", "content": "hello"}],
            }
        ],
    }
    (tmp_path / "bad.json").write_text(json.dumps(bad_payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_eval_cases(tmp_path, "smoke")


def test_evaluate_case_scores_exact_and_semantic_hits() -> None:
    executor = _executor()

    exact_case = EvalCase(
        id="exact-hit",
        query="duplicate charge refund",
        top_k=3,
        profiles=("smoke",),
        documents=[
            {
                "source_id": "ticket-1",
                "source_name": "ticket-1.txt",
                "source_origin": "raw",
                "source_type": "ticket",
                "content": "Customer reported duplicate charge and refund request.",
            }
        ],
        expected=[
            ExpectedHit(
                source_id="ticket-1",
                must_include_text="duplicate charge",
                match_type="exact",
            )
        ],
    )

    exact_score = evaluate_case(exact_case, executor)
    assert exact_score.source_recall == 1.0
    assert exact_score.span_recall == 1.0
    assert exact_score.span_recall_at_5 == 1.0
    assert exact_score.primary_span_accuracy == 1.0
    assert exact_score.top_1_span_accuracy == 1.0
    assert exact_score.exact_highlight_rate == 1.0
    assert exact_score.exact_hit_precision == 1.0
    assert exact_score.time_to_first_useful_ms > 0
    assert exact_score.time_to_final_ms > 0

    semantic_case = EvalCase(
        id="semantic-hit",
        query="refund request from mexico customer",
        top_k=3,
        profiles=("semantic",),
        documents=[
            {
                "source_id": "ticket-es",
                "source_name": "ticket-es.txt",
                "source_origin": "connector",
                "source_type": "ticket",
                "content": "Cliente de Mexico solicita reembolso por factura duplicada.",
            }
        ],
        expected=[
            ExpectedHit(
                source_id="ticket-es",
                must_include_text="reembolso",
                match_type="semantic_or_context",
            )
        ],
    )

    semantic_score = evaluate_case(semantic_case, executor)
    assert semantic_score.source_recall == 1.0
    assert semantic_score.semantic_context_success == 1.0
    assert semantic_score.semantic_hit_recall == 1.0


def test_cold_corpus_gate_keeps_cache_and_projection_untouched(tmp_path: Path) -> None:
    executor = _executor()
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    case = EvalCase(
        id="cold-hit",
        query="cold marker",
        top_k=3,
        profiles=("smoke",),
        documents=[
            {
                "source_id": "cold-raw",
                "source_name": "cold.txt",
                "source_origin": "raw",
                "source_type": "ticket",
                "content": "cold marker appears in a transient raw ticket",
            }
        ],
        expected=[
            ExpectedHit(
                source_id="cold-raw",
                must_include_text="cold marker",
                match_type="exact",
            )
        ],
    )

    score = evaluate_case(
        case,
        executor,
        cold_gate=ColdCorpusGate(
            bm25_cache=cache,
            projection_model_path=tmp_path / "missing-projection.npz",
        ),
    )

    assert score.cold_gate_failures == []
    assert cache.snapshot() == {}


def test_evaluate_case_partial_scan_can_recover_with_prefiltering() -> None:
    executor = _executor()
    case = EvalCase(
        id="partial-miss",
        query="fatal timeout marker 9f2",
        top_k=5,
        profiles=("latency",),
        generated_lines={
            "source_id": "log-large",
            "source_name": "large.log",
            "source_origin": "raw",
            "source_type": "log",
            "line_count": 1500,
            "target_line": 1405,
            "target_text": "fatal timeout marker 9f2",
            "noise_prefix": "heartbeat",
        },
        expected=[
            ExpectedHit(
                source_id="log-large",
                must_include_text="fatal timeout marker 9f2",
                match_type="exact",
            )
        ],
    )

    score = evaluate_case(case, executor)

    assert score.execution["partial"] is True
    assert score.execution["stage0_applied"] is True
    assert score.partial_miss_rate == 0.0
    assert score.span_recall == 1.0


def test_run_eval_applies_gate_failures(fixtures_dir: Path) -> None:
    config = build_run_config("smoke", fixtures_dir, semantic_mode="deterministic")
    config.gate_thresholds_min["source_recall_at_k"] = 1.1

    scorecard = run_eval(config)

    assert scorecard.status == "failed"
    assert scorecard.exit_code == 2
    assert scorecard.failed_gates
    assert "persistence_dependence_ratio" in scorecard.metrics


def test_semantic_profile_skips_when_real_projection_missing(
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.retrieval.eval_harness as harness

    def fake_resolve_projection(mode: str):
        return NullProjection(ProjectionConfig(hash_features=16, dim=4)), "projection missing"

    monkeypatch.setattr(harness, "_resolve_projection", fake_resolve_projection)

    config = EvalRunConfig(
        profile="semantic",
        fixtures_dir=fixtures_dir,
        semantic_mode="real",
        gate_metrics=True,
    )

    scorecard = run_eval(config)

    assert scorecard.status == "skipped"
    assert scorecard.exit_code == 0
    assert scorecard.skipped_reason == "projection missing"
