from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from app.config import get_settings

TierName = Literal["tiny", "medium", "huge"]


@dataclass(slots=True)
class QueryPlan:
    tier: TierName
    window_tier: TierName
    byte_tier: TierName
    channels: tuple[str, ...]
    window_count: int
    bytes_loaded: int
    scan_budget_bytes: int
    scan_limit: int
    candidate_limit: int
    rerank_limit: int
    partial: bool
    metadata_prefilter: bool
    use_cache: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        return payload


def build_query_plan(
    *,
    query: str,
    top_k: int,
    window_count: int,
    bytes_loaded: int = 0,
    source_origins: set[str] | None = None,
    cacheable: bool = False,
    budget_hint: Literal["auto", "fast", "thorough"] = "auto",
) -> QueryPlan:
    del query  # Reserved for future query-shape heuristics.
    origins = source_origins or set()
    k = max(1, top_k)
    settings = get_settings()

    if window_count <= 400:
        window_tier: TierName = "tiny"
        scan_limit = window_count
        candidate_limit = max(200, k * 30)
        rerank_limit = max(120, k * 20)
        metadata_prefilter = False
    elif window_count <= 4_000:
        window_tier = "medium"
        scan_limit = min(window_count, max(1_200, k * 120))
        candidate_limit = min(max(320, k * 40), 1_200)
        rerank_limit = min(max(200, k * 25), 320)
        metadata_prefilter = False
    else:
        window_tier = "huge"
        scan_limit = min(window_count, max(1_500, k * 90))
        candidate_limit = min(max(300, k * 35), 900)
        rerank_limit = min(max(200, k * 20), 260)
        metadata_prefilter = True

    if bytes_loaded <= settings.waver_tiny_max_bytes:
        byte_tier: TierName = "tiny"
    elif bytes_loaded <= settings.waver_medium_max_bytes:
        byte_tier = "medium"
    else:
        byte_tier = "huge"

    tier_order = {"tiny": 0, "medium": 1, "huge": 2}
    tier: TierName = max(window_tier, byte_tier, key=lambda item: tier_order[item])

    if tier != window_tier:
        if tier == "medium":
            scan_limit = min(window_count, max(scan_limit, max(1_200, k * 120)))
            candidate_limit = min(max(candidate_limit, max(320, k * 40)), 1_200)
            rerank_limit = min(max(rerank_limit, max(200, k * 25)), 320)
            metadata_prefilter = False
        else:
            scan_limit = min(window_count, max(1_500, k * 90))
            candidate_limit = min(max(300, k * 35), 900)
            rerank_limit = min(max(200, k * 20), 260)
            metadata_prefilter = True

    if "connector" in origins and tier != "tiny":
        scan_limit = min(window_count, int(scan_limit * 0.9))

    if budget_hint == "thorough":
        candidate_limit = int(candidate_limit * 2)
        rerank_limit = int(rerank_limit * 1.5)
    elif budget_hint == "fast":
        candidate_limit = max(k * 5, candidate_limit // 2)
        rerank_limit = max(k * 3, rerank_limit // 2)

    partial = window_count > scan_limit
    use_cache = cacheable and tier != "huge" and origins.issubset({"stored"})
    if tier == "tiny":
        scan_budget_bytes = min(bytes_loaded, settings.waver_tiny_max_bytes)
    else:
        scan_budget_bytes = min(bytes_loaded, settings.waver_medium_max_bytes)

    return QueryPlan(
        tier=tier,
        window_tier=window_tier,
        byte_tier=byte_tier,
        channels=("exact", "semantic", "structure"),
        window_count=window_count,
        bytes_loaded=bytes_loaded,
        scan_budget_bytes=scan_budget_bytes,
        scan_limit=scan_limit,
        candidate_limit=candidate_limit,
        rerank_limit=rerank_limit,
        partial=partial,
        metadata_prefilter=metadata_prefilter,
        use_cache=use_cache,
    )


__all__ = ["QueryPlan", "build_query_plan"]
