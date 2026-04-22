from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Literal

from app.parsers.base import ParserDetector, build_default_parser_registry
from app.parsers.plaintext import PlainTextParser
from app.retrieval.bm25 import BM25ScoredChunk
from app.retrieval.eval_harness import (
    EvalCase,
    EvalRunConfig,
    EvalScorecard,
    _aggregate_metrics,
    _document_from_payload,
    _evaluate_gates,
    _resolve_projection,
    evaluate_case,
)
from app.retrieval.planner import build_query_plan
from app.retrieval.reranker import HeuristicReranker
from app.retrieval.span_executor import SpanExecutor
from app.retrieval.sparse_projection import NullProjection
from app.schemas.documents import Chunk, SourceLocation

LayerName = Literal["exact_only", "semantic_only", "rerank_only", "planner_only"]


def _load_cases_from_files(
    fixtures_dir: Path,
    names: tuple[str, ...],
    profile: str,
) -> list[EvalCase]:
    import json

    cases: list[EvalCase] = []
    for name in names:
        path = fixtures_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError(f"Fixture {path} must define a 'cases' list")
        for raw_case in raw_cases:
            case = EvalCase.from_dict(dict(raw_case))
            if profile in case.profiles:
                cases.append(case)
    if not cases:
        raise ValueError(f"No layer eval cases found in {fixtures_dir} for {profile}")
    return cases


def _executor(
    projection: object,
    *,
    enabled_channels: tuple[str, ...],
    rerank_enabled: bool,
) -> SpanExecutor:
    detector = ParserDetector(
        registry=build_default_parser_registry(),
        default_parser=PlainTextParser(),
    )
    return SpanExecutor(
        parser_detector=detector,
        reranker=HeuristicReranker(),
        projection=projection,
        enabled_channels=enabled_channels,
        rerank_enabled=rerank_enabled,
    )


def _scorecard(
    *,
    config: EvalRunConfig,
    metrics: dict[str, float],
    failed_gates: list[str],
    case_scores: list[dict],
    started_at: float,
    skipped_reason: str | None = None,
) -> EvalScorecard:
    status: Literal["passed", "failed", "skipped"] = "passed"
    if skipped_reason is not None:
        status = "skipped"
    elif failed_gates:
        status = "failed"
    return EvalScorecard(
        profile=config.profile,
        semantic_mode=config.semantic_mode,
        status=status,
        skipped_reason=skipped_reason,
        metrics=metrics,
        failed_gates=failed_gates,
        cases_evaluated=len(case_scores),
        duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
        case_scores=case_scores,
    )


def _run_channel_layer(
    layer: LayerName,
    config: EvalRunConfig,
    *,
    files: tuple[str, ...],
    enabled_channels: tuple[str, ...],
    rerank_enabled: bool,
    min_gates: dict[str, float],
) -> EvalScorecard:
    started_at = perf_counter()
    cases = _load_cases_from_files(config.fixtures_dir, files, config.profile)
    projection, warning = _resolve_projection(config.semantic_mode)
    if config.semantic_mode == "real" and isinstance(projection, NullProjection):
        return _scorecard(
            config=config,
            metrics={},
            failed_gates=[],
            case_scores=[],
            started_at=started_at,
            skipped_reason=warning,
        )

    executor = _executor(
        projection,
        enabled_channels=enabled_channels,
        rerank_enabled=rerank_enabled,
    )
    scores = [
        evaluate_case(case, executor, top_k_override=config.top_k_override)
        for case in cases
    ]
    metrics = _aggregate_metrics(scores)
    layer_config = EvalRunConfig(
        profile=config.profile,
        fixtures_dir=config.fixtures_dir,
        semantic_mode=config.semantic_mode,
        top_k_override=config.top_k_override,
        gate_thresholds_min=min_gates,
        gate_thresholds_max={},
        gate_metrics=True,
    )
    failed = _evaluate_gates(layer_config, metrics)
    metrics["layer"] = 1.0
    metrics[f"{layer}_cases"] = float(len(scores))
    return _scorecard(
        config=config,
        metrics=metrics,
        failed_gates=failed,
        case_scores=[asdict(score) for score in scores],
        started_at=started_at,
    )


def _chunk(source_id: str, text: str, score: float) -> BM25ScoredChunk:
    return BM25ScoredChunk(
        chunk=Chunk(
            chunk_id=source_id,
            content=text,
            source_name=f"{source_id}.txt",
            metadata={"source_id": source_id},
            location=SourceLocation(),
            token_count=max(1, len(text.split())),
        ),
        score=score,
        bm25_score=score,
    )


def _run_rerank_only(config: EvalRunConfig) -> EvalScorecard:
    started_at = perf_counter()
    cases = _load_cases_from_files(
        config.fixtures_dir,
        ("mixed_exact_semantic.json", "exact_lookups.json"),
        config.profile,
    )
    reranker = HeuristicReranker()
    baseline_hits = 0
    reranked_hits = 0
    case_scores: list[dict] = []

    for index, case in enumerate(cases[:25]):
        expected = case.expected[0]
        docs = [_document_from_payload(doc, idx) for idx, doc in enumerate(case.documents)]
        gold_doc = next(
            (doc for doc in docs if doc.metadata.get("source_id") == expected.source_id),
            docs[0],
        )
        gold_text = (
            gold_doc.content
            if isinstance(gold_doc.content, str)
            else gold_doc.data.decode()
        )
        distractor = _chunk(
            f"oracle-distractor-{index}",
            "shipping address preferences and warehouse labels only",
            0.1,
        )
        gold = _chunk(expected.source_id, gold_text, 1.0)
        shortlist = [distractor, gold]
        baseline_hits += int(shortlist[0].chunk.metadata.get("source_id") == expected.source_id)
        reranked = reranker.rerank(case.query, shortlist, top_k=1)
        reranked_hits += int(reranked[0].chunk.metadata.get("source_id") == expected.source_id)
        case_scores.append(
            {
                "case_id": case.id,
                "baseline_top_1": shortlist[0].chunk.metadata.get("source_id"),
                "reranked_top_1": reranked[0].chunk.metadata.get("source_id"),
            }
        )

    total = max(1, len(case_scores))
    baseline_accuracy = baseline_hits / total
    reranked_accuracy = reranked_hits / total
    lift = reranked_accuracy - baseline_accuracy
    metrics = {
        "baseline_top_1_accuracy": round(baseline_accuracy, 4),
        "reranked_top_1_accuracy": round(reranked_accuracy, 4),
        "primary_span_accuracy_lift": round(lift, 4),
    }
    failed = []
    if lift < 0.15:
        failed.append(f"primary_span_accuracy_lift={lift:.4f} < min 0.1500")
    return _scorecard(
        config=config,
        metrics=metrics,
        failed_gates=failed,
        case_scores=case_scores,
        started_at=started_at,
    )


def _run_planner_only(config: EvalRunConfig) -> EvalScorecard:
    started_at = perf_counter()
    checks = [
        (
            "tiny_complete",
            build_query_plan(
                query="billing complaint",
                top_k=5,
                window_count=50,
                source_origins={"raw"},
                cacheable=False,
            ),
            lambda plan: plan.tier == "tiny" and not plan.partial and plan.scan_limit == 50,
        ),
        (
            "medium_partial",
            build_query_plan(
                query="refund timeout",
                top_k=5,
                window_count=2_200,
                source_origins={"stored"},
                cacheable=True,
            ),
            lambda plan: plan.tier == "medium" and plan.partial and plan.scan_limit < 2_200,
        ),
        (
            "huge_partial",
            build_query_plan(
                query='regex("503|timeout")',
                top_k=5,
                window_count=18_000,
                source_origins={"stored", "connector"},
                cacheable=True,
            ),
            lambda plan: plan.tier == "huge" and plan.partial and not plan.use_cache,
        ),
    ]
    case_scores = [
        {"case_id": name, "passed": bool(assertion(plan)), "plan": plan.to_dict()}
        for name, plan, assertion in checks
    ]
    failed = [
        f"{item['case_id']} planner assertion failed"
        for item in case_scores
        if not item["passed"]
    ]
    metrics = {
        "planner_assertion_rate": round(
            sum(1 for item in case_scores if item["passed"]) / len(case_scores),
            4,
        )
    }
    return _scorecard(
        config=config,
        metrics=metrics,
        failed_gates=failed,
        case_scores=case_scores,
        started_at=started_at,
    )


def run_layer_eval(layer: str, config: EvalRunConfig) -> EvalScorecard:
    if layer == "exact_only":
        return _run_channel_layer(
            "exact_only",
            config,
            files=("exact_lookups.json", "phrase_lookups.json"),
            enabled_channels=("exact",),
            rerank_enabled=False,
            min_gates={"exact_highlight_rate": 0.95, "top_1_span_accuracy": 0.90},
        )
    if layer == "semantic_only":
        return _run_channel_layer(
            "semantic_only",
            config,
            files=("semantic_paraphrase.json",),
            enabled_channels=("semantic",),
            rerank_enabled=False,
            min_gates={"span_recall_at_5": 0.70},
        )
    if layer == "rerank_only":
        return _run_rerank_only(config)
    if layer == "planner_only":
        return _run_planner_only(config)
    raise ValueError(f"Unsupported layer eval: {layer}")


__all__ = ["LayerName", "run_layer_eval"]
