from __future__ import annotations

import hashlib
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.errors import api_error
from app.models import ApiKey

_TOKEN_PREFIX = "wav_"


@dataclass(slots=True)
class AuthContext:
    workspace_id: str
    api_key_id: str | None = None
    authenticated: bool = False
    requests_per_minute: int = 0
    bytes_per_minute: int = 0

    def to_execution(self) -> dict[str, object]:
        return {
            "workspace_id": self.workspace_id,
            "api_key_id": self.api_key_id,
            "authenticated": self.authenticated,
        }


class InMemoryQuotaLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._bytes: dict[str, deque[tuple[float, int]]] = defaultdict(deque)

    def check_request(self, key_id: str, limit: int) -> None:
        now = time.monotonic()
        bucket = self._requests[key_id]
        self._trim_times(bucket, now)
        if len(bucket) >= limit:
            raise api_error(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "quota_requests_per_minute_exceeded",
                "Request quota exceeded for this API key",
                details={"limit": limit, "window_seconds": 60},
            )
        bucket.append(now)

    def check_bytes(self, key_id: str, byte_count: int, limit: int) -> None:
        now = time.monotonic()
        bucket = self._bytes[key_id]
        while bucket and now - bucket[0][0] > 60.0:
            bucket.popleft()
        used = sum(value for _, value in bucket)
        if used + byte_count > limit:
            raise api_error(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "quota_bytes_per_minute_exceeded",
                "Byte quota exceeded for this API key",
                details={"limit": limit, "used": used, "requested": byte_count},
            )
        bucket.append((now, byte_count))

    @staticmethod
    def _trim_times(bucket: deque[float], now: float) -> None:
        while bucket and now - bucket[0] > 60.0:
            bucket.popleft()


quota_limiter = InMemoryQuotaLimiter()


def hash_api_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return f"{_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"


async def get_auth_context(
    session: Annotated[AsyncSession, Depends(get_session)],
    authorization: Annotated[str | None, Header()] = None,
) -> AuthContext:
    settings = get_settings()
    if not authorization:
        if settings.waver_api_keys_required or settings.waver_production_mode:
            raise api_error(
                status.HTTP_401_UNAUTHORIZED,
                "missing_api_key",
                "Missing Authorization bearer token",
            )
        return AuthContext(workspace_id=settings.default_workspace)

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.startswith(_TOKEN_PREFIX):
        raise api_error(status.HTTP_401_UNAUTHORIZED, "invalid_api_key", "Invalid API key format")

    row = await session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_api_key(token)))
    if row is None or row.status != "active":
        raise api_error(status.HTTP_401_UNAUTHORIZED, "invalid_api_key", "Invalid API key")

    quota_limiter.check_request(row.id, row.requests_per_minute)
    row.last_used_at = datetime.now(UTC)
    await session.commit()
    return AuthContext(
        workspace_id=row.workspace_id,
        api_key_id=row.id,
        authenticated=True,
        requests_per_minute=row.requests_per_minute,
        bytes_per_minute=row.bytes_per_minute,
    )
