from __future__ import annotations

import json
import re
from pathlib import Path

from app.retrieval.eval_harness import DeterministicEvalProjection, build_run_config
from app.retrieval.eval_layers import run_layer_eval
from app.retrieval.sparse_projection import NullProjection, ProjectionConfig


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "eval" / "tickets"


def _fixture_payloads() -> list[tuple[Path, dict]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(_fixtures_dir().glob("*.json"))
    ]


def test_ticket_eval_fixture_count_and_schema() -> None:
    payloads = _fixture_payloads()
    counts = {path.name: len(payload["cases"]) for path, payload in payloads}

    assert counts == {
        "exact_lookups.json": 25,
        "freshness.json": 25,
        "mixed_exact_semantic.json": 25,
        "negatives.json": 25,
        "phrase_lookups.json": 25,
        "semantic_paraphrase.json": 25,
    }
    assert sum(counts.values()) == 150
    assert all(payload["authorship"] == "ai-generated" for _, payload in payloads)
    assert all(
        "regression" in case["profiles"]
        for _, payload in payloads
        for case in payload["cases"]
    )
    assert all(
        case["disallowed_sources"]
        for path, payload in payloads
        if path.name == "negatives.json"
        for case in payload["cases"]
    )


def test_ticket_eval_expected_text_is_verbatim() -> None:
    for _, payload in _fixture_payloads():
        for case in payload["cases"]:
            documents = {
                document["source_id"]: document["content"]
                for document in case["documents"]
            }
            for expected in case.get("expected", []):
                needle = expected.get("must_include_text")
                if needle:
                    assert needle in documents[expected["source_id"]]


def test_semantic_fixture_queries_do_not_use_eval_aliases() -> None:
    aliases = set(DeterministicEvalProjection._ALIASES)
    semantic = json.loads(
        (_fixtures_dir() / "semantic_paraphrase.json").read_text(encoding="utf-8")
    )

    for case in semantic["cases"]:
        tokens = set(re.findall(r"\w+", case["query"].lower()))
        assert tokens.isdisjoint(aliases)


def test_exact_layer_gate_passes() -> None:
    config = build_run_config("smoke", _fixtures_dir(), semantic_mode="deterministic")

    scorecard = run_layer_eval("exact_only", config)

    assert scorecard.status == "passed"
    assert scorecard.metrics["exact_highlight_rate"] >= 0.95
    assert scorecard.metrics["top_1_span_accuracy"] >= 0.90


def test_semantic_layer_gate_passes() -> None:
    config = build_run_config("smoke", _fixtures_dir(), semantic_mode="deterministic")

    scorecard = run_layer_eval("semantic_only", config)

    assert scorecard.status == "passed"
    assert scorecard.metrics["span_recall_at_5"] >= 0.70


def test_rerank_and_planner_layers_pass() -> None:
    config = build_run_config("smoke", _fixtures_dir(), semantic_mode="deterministic")

    rerank = run_layer_eval("rerank_only", config)
    planner = run_layer_eval("planner_only", config)

    assert rerank.status == "passed"
    assert rerank.metrics["primary_span_accuracy_lift"] >= 0.15
    assert planner.status == "passed"
    assert planner.metrics["planner_assertion_rate"] == 1.0


def test_semantic_layer_real_projection_missing_skips(monkeypatch) -> None:
    import app.retrieval.eval_layers as layers

    def fake_resolve_projection(mode: str):
        return NullProjection(ProjectionConfig(hash_features=16, dim=4)), "projection missing"

    monkeypatch.setattr(layers, "_resolve_projection", fake_resolve_projection)
    config = build_run_config("smoke", _fixtures_dir(), semantic_mode="real")

    scorecard = run_layer_eval("semantic_only", config)

    assert scorecard.status == "skipped"
    assert scorecard.skipped_reason == "projection missing"
