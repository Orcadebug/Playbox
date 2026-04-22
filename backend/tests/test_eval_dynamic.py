from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base, Document, Source
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.eval_dynamic import EvalDynamicConfig, run_dynamic_eval


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as db_session:
        yield db_session

    await engine.dispose()


async def _row_count(session: AsyncSession, model: type[Base]) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


def _fixture_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "eval"
        / "dynamic"
        / "mutations.json"
    )


async def test_dynamic_eval_runs_mutations_without_persistence(session: AsyncSession) -> None:
    cache = BM25IndexCache(ttl=60.0, max_entries=5)

    scorecard = await run_dynamic_eval(
        EvalDynamicConfig(fixture_path=_fixture_path(), bm25_cache=cache),
        session=session,
    )

    assert scorecard.status == "passed"
    assert scorecard.failed_gates == []
    assert scorecard.metrics["queries_evaluated"] == 6.0
    assert cache.snapshot() == {}
    assert await _row_count(session, Source) == 0
    assert await _row_count(session, Document) == 0
    found_by_case = {
        item["case_id"]: item["found_sources"] for item in scorecard.case_scores
    }
    assert found_by_case["added-gamma"] == ["dyn-3"]
    assert found_by_case["deleted-alpha"] == []
    assert found_by_case["mutated-beta"] == ["dyn-2"]
