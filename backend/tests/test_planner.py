from __future__ import annotations

from app.retrieval.planner import build_query_plan


def test_planner_uses_tiny_tier_for_small_window_sets() -> None:
    plan = build_query_plan(
        query="billing complaint",
        top_k=10,
        window_count=120,
        source_origins={"stored"},
        cacheable=True,
    )

    assert plan.tier == "tiny"
    assert plan.scan_limit == 120
    assert plan.partial is False
    assert plan.use_cache is True


def test_planner_uses_medium_tier_with_partial_when_limit_exceeded() -> None:
    plan = build_query_plan(
        query="refund timeout",
        top_k=8,
        window_count=2_200,
        source_origins={"stored"},
        cacheable=True,
    )

    assert plan.tier == "medium"
    assert plan.scan_limit < 2_200
    assert plan.partial is True
    assert plan.metadata_prefilter is False
    assert plan.candidate_limit >= 320
    assert plan.rerank_limit >= 200


def test_planner_uses_huge_tier_and_disables_cache_for_large_or_live() -> None:
    plan = build_query_plan(
        query='regex("timeout|503") billing dispute',
        top_k=5,
        window_count=18_000,
        source_origins={"stored", "connector"},
        cacheable=True,
    )

    assert plan.tier == "huge"
    assert plan.metadata_prefilter is True
    assert plan.partial is True
    assert plan.use_cache is False
    assert plan.scan_limit < plan.window_count
