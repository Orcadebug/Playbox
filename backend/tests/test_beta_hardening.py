from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth import generate_api_key, get_auth_context, hash_api_key
from app.limits import enforce_search_limits
from app.models import ApiKey, Base
from app.runtime import readiness_status, validate_production_requirements


@pytest.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as db_session:
        yield db_session
    await engine.dispose()


async def test_api_key_auth_uses_hashed_storage_and_workspace(session: AsyncSession) -> None:
    token = generate_api_key()
    session.add(
        ApiKey(
            workspace_id="acme",
            key_hash=hash_api_key(token),
            requests_per_minute=5,
            bytes_per_minute=1024,
        )
    )
    await session.commit()

    context = await get_auth_context(session, f"Bearer {token}")

    assert context.authenticated is True
    assert context.workspace_id == "acme"
    assert context.bytes_per_minute == 1024


async def test_api_key_quota_rejects_second_request(session: AsyncSession) -> None:
    token = generate_api_key()
    session.add(
        ApiKey(
            workspace_id="acme",
            key_hash=hash_api_key(token),
            requests_per_minute=1,
            bytes_per_minute=1024,
        )
    )
    await session.commit()

    await get_auth_context(session, f"Bearer {token}")
    with pytest.raises(HTTPException) as exc:
        await get_auth_context(session, f"Bearer {token}")

    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "quota_requests_per_minute_exceeded"


def test_request_limits_return_stable_error_codes() -> None:
    tokenless = type(
        "Auth",
        (),
        {
            "authenticated": False,
            "api_key_id": None,
            "bytes_per_minute": 0,
        },
    )()

    with pytest.raises(HTTPException) as exc:
        enforce_search_limits(
            top_k=10_000,
            raw_sources=[],
            connector_configs=[],
            auth=tokenless,  # type: ignore[arg-type]
        )

    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "top_k_limit_exceeded"


def test_readiness_reports_artifacts_and_rust_exports() -> None:
    status = readiness_status()

    assert "artifacts" in status
    assert "rust_exports" in status
    assert "projection" in status["artifacts"]


def test_production_mode_fails_when_artifacts_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class FakeSettings:
        waver_production_mode = True

    monkeypatch.setattr(
        "app.runtime.readiness_status",
        lambda settings: {
            "ready": False,
            "missing": ["projection"],
            "missing_rust_exports": [],
        },
    )
    monkeypatch.setattr("app.runtime.get_settings", lambda: FakeSettings())

    with pytest.raises(RuntimeError):
        validate_production_requirements()
