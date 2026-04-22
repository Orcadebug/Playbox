from __future__ import annotations

import json
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


async def _collect_stream_events(
    service: SearchService,
    *,
    query: str,
    top_k: int = 3,
    answer_mode: str = "off",
    raw_sources: list[dict] | None = None,
    connector_configs: list[dict] | None = None,
) -> list[dict]:
    events: list[dict] = []
    async for chunk in service.stream_search(
        query=query,
        top_k=top_k,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        answer_mode=answer_mode,
        raw_sources=raw_sources or [],
        connector_configs=connector_configs or [],
    ):
        if not chunk.startswith("data: "):
            continue
        payload = json.loads(chunk[6:].strip())
        if isinstance(payload, dict):
            events.append(payload)
    return events


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


async def test_structured_raw_json_searches_immediately_without_persistence(
    session: AsyncSession,
) -> None:
    service = _service(session)

    response = await service.search(
        query="duplicate charge",
        top_k=3,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        raw_sources=[
            {
                "id": "raw-json-ticket",
                "name": "live-ticket.json",
                "content": {
                    "ticket": "A-7",
                    "customer": "Acme",
                    "issue": "Customer billed twice yesterday",
                },
            }
        ],
    )

    assert response["results"]
    result = response["results"][0]
    assert result["source_origin"] == "raw"
    assert result["metadata"]["media_type"] == "application/json"
    assert "semantic" in result["channels"]
    assert result["channel_scores"]["semantic"] > 0
    assert result["primary_span"]["offset_basis"] == "parsed"
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


async def test_structured_webhook_payload_uses_semantic_fallback(session: AsyncSession) -> None:
    service = _service(session)

    response = await service.search(
        query="duplicate charge",
        top_k=3,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        connector_configs=[
            {
                "connector_id": "webhook",
                "documents": [
                    {
                        "name": "ticket-structured.json",
                        "content": {
                            "customer": "Acme",
                            "issue": "Customer billed twice",
                        },
                    }
                ],
            }
        ],
    )

    assert response["results"]
    result = response["results"][0]
    assert result["source_origin"] == "connector"
    assert result["metadata"]["media_type"] == "application/json"
    assert "semantic" in result["channels"]
    assert result["channel_scores"]["semantic"] > 0
    assert await _row_count(session, Source) == 0
    assert await _row_count(session, Document) == 0


async def test_mixed_exact_and_semantic_channels_return_one_ranked_experience(
    session: AsyncSession,
) -> None:
    service = _service(session)

    response = await service.search(
        query="duplicate charge",
        top_k=5,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        raw_sources=[
            {
                "id": "exact-note",
                "name": "exact.txt",
                "content": "Customer reported a duplicate charge on the invoice.",
            },
            {
                "id": "semantic-note",
                "name": "semantic.txt",
                "content": "Customer was billed twice and asked for help.",
            },
        ],
    )

    assert len(response["results"]) >= 2
    channels_by_source = {
        result["source_id"]: set(result["channels"])
        for result in response["results"]
    }
    assert "exact" in channels_by_source["exact-note"]
    assert "semantic" in channels_by_source["semantic-note"]
    assert all("channel_scores" in result for result in response["results"])


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


async def test_search_response_includes_execution_metadata(session: AsyncSession) -> None:
    service = _service(session)

    response = await service.search(
        query="billing",
        top_k=2,
        source_ids=None,
        workspace_id="default",
        include_stored_sources=False,
        raw_sources=[{"name": "note.txt", "content": "billing refund"}],
    )

    assert "execution" in response
    assert isinstance(response["execution"], dict)
    assert "scan_limit" in response["execution"]
    assert "candidate_count" in response["execution"]
    assert "elapsed_ms" in response["execution"]
    assert "bytes_loaded" in response["execution"]
    assert "skipped_windows" in response["execution"]
    assert "completion_reason" in response["execution"]
    assert "phase_counts" in response["execution"]


async def test_stream_search_emits_progressive_events_in_order(session: AsyncSession) -> None:
    service = _service(session)

    events = await _collect_stream_events(
        service,
        query="billing refund",
        raw_sources=[{"name": "note.txt", "content": "billing refund duplicate charge"}],
    )

    assert events
    assert events[0]["event"] == "sources_loaded"
    assert events[-1]["event"] == "done"
    event_types = [event["event"] for event in events]
    assert any(
        event_type in {"exact_results", "proxy_results", "reranked_results"}
        for event_type in event_types
    )
    done = events[-1]
    assert done["results"]
    assert done["source_errors"] == []
    assert "execution" in done


async def test_stream_search_answer_mode_emits_answer_deltas(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.answer.generator import AnswerGenerator

    async def fake_stream_generate(self, query, results):  # noqa: ANN001
        for token in ("Billing ", "answer ", "stream"):
            yield token

    monkeypatch.setattr(AnswerGenerator, "stream_generate", fake_stream_generate)
    service = _service(session)

    events = await _collect_stream_events(
        service,
        query="billing refund",
        answer_mode="llm",
        raw_sources=[{"name": "note.txt", "content": "billing refund duplicate charge"}],
    )

    delta_events = [event for event in events if event.get("event") == "answer_delta"]
    assert len(delta_events) == 3
    done = events[-1]
    assert done["event"] == "done"
    assert done["answer"] == {
        "markdown": "Billing answer stream",
        "confidence": "low",
    }
