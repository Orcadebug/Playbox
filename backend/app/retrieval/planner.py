from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(slots=True)
class QueryPlan:
    tier: Literal["tiny", "medium", "huge"]
    channels: tuple[str, ...]
    window_count: int
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
    source_origins: set[str] | None = None,
    cacheable: bool = False,
) -> QueryPlan:
    del query  # Reserved for future query-shape heuristics.
    origins = source_origins or set()
    k = max(1, top_k)

    if window_count <= 400:
        tier: Literal["tiny", "medium", "huge"] = "tiny"
        scan_limit = window_count
        candidate_limit = max(200, k * 30)
        rerank_limit = max(120, k * 20)
        metadata_prefilter = False
    elif window_count <= 4_000:
        tier = "medium"
        scan_limit = min(window_count, max(1_200, k * 120))
        candidate_limit = min(max(320, k * 40), 1_200)
        rerank_limit = min(max(200, k * 25), 320)
        metadata_prefilter = False
    else:
        tier = "huge"
        scan_limit = min(window_count, max(1_500, k * 90))
        candidate_limit = min(max(300, k * 35), 900)
        rerank_limit = min(max(200, k * 20), 260)
        metadata_prefilter = True

    if "connector" in origins and tier != "tiny":
        scan_limit = min(window_count, int(scan_limit * 0.9))

    partial = window_count > scan_limit
    use_cache = cacheable and tier != "huge" and origins.issubset({"stored"})

    return QueryPlan(
        tier=tier,
        channels=("exact", "semantic", "structure"),
        window_count=window_count,
        scan_limit=scan_limit,
        candidate_limit=candidate_limit,
        rerank_limit=rerank_limit,
        partial=partial,
        metadata_prefilter=metadata_prefilter,
        use_cache=use_cache,
    )


__all__ = ["QueryPlan", "build_query_plan"]
