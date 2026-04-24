from __future__ import annotations

import pytest

from scripts.run_latency_gate import _run


@pytest.mark.asyncio
async def test_latency_gate_outputs_ci_schema() -> None:
    result = await _run(iterations=1)

    assert result["profile"] == "small_raw_p95_200ms"
    assert {"p50", "p95", "p99", "max"} <= set(result["latency_ms"])
    assert "machine" in result
    assert "readiness" in result
