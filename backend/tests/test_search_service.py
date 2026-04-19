from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base, Document, Source
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import HeuristicReranker
from app.services.search import SearchService


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as db_session:
        yield db_session

    await engine.dispose()


def _service(session: AsyncSession) -> SearchService:
    service = SearchService(session)
    service.pipeline = RetrievalPipeline(reranker=HeuristicReranker())
    return service


async def _row_count(session: AsyncSession, model: type[Base]) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def test_raw_source_search_does_not_persist_rows(session: AsyncSession) -> None:
    service = _service(session)
    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    service.pipeline = RetrievalPipeline(reranker=HeuristicReranker(), bm25_cache=cache)

    response = await service.search(
        query="billing refund",
        top_k=3,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        raw_sources=[
            {
                "id": "raw-note",
                "name": "support note",
                "content": "Acme asked for a billing refund after a duplicate charge.",
                "media_type": "text/plain",
                "metadata": {"customer": "Acme"},
            }
        ],
    )

    assert response["results"]
    assert response["results"][0]["source_origin"] == "raw"
    assert response["results"][0]["metadata"]["source_id"] == "raw-note"
    assert response["source_errors"] == []
    assert response["answer"] is None
    assert response["answer_error"] is None
    with cache._lock:
        assert cache._cache == {}
    assert await _row_count(session, Source) == 0
    assert await _row_count(session, Document) == 0


async def test_webhook_connector_search_is_transient_when_live_connectors_disabled(
    session: AsyncSession,
) -> None:
    service = _service(session)

    response = await service.search(
        query="invoice dispute",
        top_k=3,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        connector_configs=[
            {
                "connector_id": "webhook",
                "documents": [
                    {
                        "name": "ticket-7.txt",
                        "content": "Acme opened an invoice dispute yesterday.",
                        "api_key": "must-not-leak",
                        "attachments": [{"api_key": "must-not-leak-nested", "name": "safe"}],
                    }
                ],
            }
        ],
    )

    assert response["results"]
    result = response["results"][0]
    assert result["source_origin"] == "connector"
    assert result["metadata"]["connector_id"] == "webhook"
    assert "api_key" not in result["metadata"]
    assert result["metadata"]["attachments"] == [{"name": "safe"}]
    assert response["source_errors"] == []
    assert await _row_count(session, Source) == 0
    assert await _row_count(session, Document) == 0


async def test_unknown_and_disabled_connectors_return_source_errors(
    session: AsyncSession,
) -> None:
    service = _service(session)

    response = await service.search(
        query="anything",
        top_k=3,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        connector_configs=[
            {"connector_id": "unknown_connector"},
            {"connector_id": "slack", "bot_token": "xoxb-secret"},
        ],
    )

    assert response["results"] == []
    assert response["source_errors"] == [
        {
            "source_type": "connector",
            "connector_id": "unknown_connector",
            "source_name": None,
            "message": "Unknown connector",
        },
        {
            "source_type": "connector",
            "connector_id": "slack",
            "source_name": None,
            "message": "Live connector is disabled",
        },
    ]


async def test_mixed_search_with_connector_config_skips_cache(session: AsyncSession) -> None:
    source = Source(
        workspace_id="default",
        source_type="upload",
        name="stored.txt",
        media_type="text/plain",
        parser_name="plaintext",
        source_metadata={"document_count": 1},
    )
    session.add(source)
    await session.flush()
    session.add(
        Document(
            source_id=source.id,
            title="stored.txt",
            content="stored billing refund note",
            order_index=0,
            document_metadata={},
        )
    )
    await session.commit()

    cache = BM25IndexCache(ttl=60.0, max_entries=5)
    service = _service(session)
    service.pipeline = RetrievalPipeline(reranker=HeuristicReranker(), bm25_cache=cache)

    response = await service.search(
        query="billing",
        top_k=1,
        source_ids=None,
        workspace_id="default",
        connector_configs=[{"connector_id": "unknown_connector"}],
    )

    assert response["results"]
    assert response["source_errors"]
    with cache._lock:
        assert cache._cache == {}


async def test_answer_generation_only_runs_when_requested(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.answer.generator import AnswerGenerator

    async def fail_generate(self, query, results):  # noqa: ANN001
        raise AssertionError("AnswerGenerator should not run by default")

    monkeypatch.setattr(AnswerGenerator, "generate", fail_generate)
    service = _service(session)

    response = await service.search(
        query="billing",
        top_k=1,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        raw_sources=[{"name": "note.txt", "content": "billing refund"}],
    )

    assert response["answer"] is None
    assert response["answer_error"] is None


async def test_answer_mode_llm_returns_generator_result(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.answer.generator import AnswerGenerator

    async def fake_generate(self, query, results):  # noqa: ANN001
        return {"markdown": "Billing refund found. [1]", "confidence": "medium"}, None

    monkeypatch.setattr(AnswerGenerator, "generate", fake_generate)
    service = _service(session)

    response = await service.search(
        query="billing",
        top_k=1,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        answer_mode="llm",
        raw_sources=[{"name": "note.txt", "content": "billing refund"}],
    )

    assert response["answer"] == {
        "markdown": "Billing refund found. [1]",
        "confidence": "medium",
    }
    assert response["answer_error"] is None


async def test_answer_mode_llm_preserves_results_on_answer_error(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.answer.generator import AnswerGenerator

    async def fake_generate(self, query, results):  # noqa: ANN001
        return None, "Answer generation unavailable"

    monkeypatch.setattr(AnswerGenerator, "generate", fake_generate)
    service = _service(session)

    response = await service.search(
        query="billing",
        top_k=1,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        answer_mode="llm",
        raw_sources=[{"name": "note.txt", "content": "billing refund"}],
    )

    assert response["results"]
    assert response["answer"] is None
    assert response["answer_error"] == "Answer generation unavailable"
