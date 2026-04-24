from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.auth import AuthContext, quota_limiter
from app.config import get_settings
from app.errors import api_error


@dataclass(slots=True)
class LimitContext:
    raw_bytes: int
    raw_sources: int
    connectors: int
    top_k: int

    def to_execution(self) -> dict[str, int]:
        return {
            "raw_bytes": self.raw_bytes,
            "raw_sources": self.raw_sources,
            "connectors": self.connectors,
            "top_k": self.top_k,
        }


def raw_content_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(str(value).encode("utf-8"))


def enforce_search_limits(
    *,
    top_k: int,
    raw_sources: list[dict[str, Any]],
    connector_configs: list[dict[str, Any]],
    auth: AuthContext,
) -> LimitContext:
    settings = get_settings()
    raw_bytes = sum(raw_content_size(source.get("content")) for source in raw_sources)
    raw_count = len(raw_sources)
    connector_count = len(connector_configs)

    if top_k > settings.waver_max_top_k:
        raise api_error(
            422,
            "top_k_limit_exceeded",
            "top_k exceeds the beta limit",
            details={"limit": settings.waver_max_top_k, "received": top_k},
        )
    if raw_count > settings.waver_max_raw_sources:
        raise api_error(
            422,
            "raw_source_limit_exceeded",
            "Too many raw sources in one request",
            details={"limit": settings.waver_max_raw_sources, "received": raw_count},
        )
    if connector_count > settings.waver_max_connectors:
        raise api_error(
            422,
            "connector_limit_exceeded",
            "Too many connector configs in one request",
            details={"limit": settings.waver_max_connectors, "received": connector_count},
        )
    if raw_bytes > settings.waver_max_raw_bytes:
        raise api_error(
            413,
            "raw_payload_too_large",
            "Raw payload exceeds the beta byte limit",
            details={"limit": settings.waver_max_raw_bytes, "received": raw_bytes},
        )
    if auth.authenticated:
        quota_limiter.check_bytes(auth.api_key_id or "", raw_bytes, auth.bytes_per_minute)
    return LimitContext(
        raw_bytes=raw_bytes,
        raw_sources=raw_count,
        connectors=connector_count,
        top_k=top_k,
    )
