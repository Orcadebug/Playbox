from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Base
from app.runtime import readiness_status
from app.services.search import SearchService

_CASES = [
    {
        "query": "duplicate billing refund",
        "content": "Ticket 9182: Acme reports duplicate billing and requests a refund.",
    },
    {
        "query": "timeout during checkout",
        "content": "API payload: checkout attempts return timeout after payment confirmation.",
    },
    {
        "query": "refund denied",
        "content": "Support note: refund denied because the invoice was already voided.",
    },
    {
        "query": "launch blocker",
        "content": "Incident log: launch blocker is missing webhook signature validation.",
    },
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


async def _run(iterations: int) -> dict[str, Any]:
    settings = get_settings()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    timings: list[float] = []
    model_modes: set[str] = set()
    async with session_factory() as session:
        service = SearchService(session)
        for _ in range(2):  # warm sessions and model singletons
            case = _CASES[0]
            await service.search(
                query=case["query"],
                top_k=3,
                source_ids=None,
                workspace_id="latency-gate",
                include_stored_sources=False,
                raw_sources=[{"name": "warmup.txt", "content": case["content"]}],
                answer_mode="off",
                budget_hint="fast",
            )
        for index in range(iterations):
            case = _CASES[index % len(_CASES)]
            started = time.perf_counter()
            response = await service.search(
                query=case["query"],
                top_k=3,
                source_ids=None,
                workspace_id="latency-gate",
                include_stored_sources=False,
                raw_sources=[
                    {
                        "name": f"latency-{index}.txt",
                        "content": case["content"],
                        "media_type": "text/plain",
                    }
                ],
                answer_mode="off",
                budget_hint="fast",
            )
            timings.append((time.perf_counter() - started) * 1000.0)
            model_modes.add(str(response["execution"].get("model_mode")))
    await engine.dispose()

    p95 = _percentile(timings, 95)
    return {
        "profile": "small_raw_p95_200ms",
        "iterations": iterations,
        "threshold_ms": settings.waver_latency_gate_p95_ms,
        "passed": p95 <= settings.waver_latency_gate_p95_ms,
        "latency_ms": {
            "p50": round(statistics.median(timings), 3),
            "p95": round(p95, 3),
            "p99": round(_percentile(timings, 99), 3),
            "max": round(max(timings), 3),
        },
        "payload": {
            "max_bytes": settings.waver_latency_gate_max_bytes,
            "case_count": len(_CASES),
        },
        "model_modes": sorted(model_modes),
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "readiness": readiness_status(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Waver small raw payload latency gate.")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = asyncio.run(_run(args.iterations))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
