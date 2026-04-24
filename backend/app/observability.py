from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import Request, Response

_log = logging.getLogger("waver.events")


@dataclass(slots=True)
class MetricsRegistry:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    histograms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.histograms[name].append(value)

    def render_prometheus(self) -> str:
        lines: list[str] = []
        for name, value in sorted(self.counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, values in sorted(self.histograms.items()):
            lines.append(f"# TYPE {name} summary")
            if values:
                lines.append(f"{name}_count {len(values)}")
                lines.append(f"{name}_sum {sum(values):.6f}")
                lines.append(f"{name}_max {max(values):.6f}")
        return "\n".join(lines) + "\n"


metrics = MetricsRegistry()


def new_request_id() -> str:
    return str(uuid4())


def log_event(event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    _log.info(json.dumps(payload, sort_keys=True, default=str))


async def request_observability_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("x-request-id") or new_request_id()
    request.state.request_id = request_id
    started = time.perf_counter()
    metrics.inc("waver_http_requests_total")
    try:
        response = await call_next(request)
    except Exception:
        metrics.inc("waver_http_errors_total")
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metrics.observe("waver_http_request_latency_ms", elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    log_event(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        elapsed_ms=round(elapsed_ms, 3),
        content_length=request.headers.get("content-length"),
    )
    return response
