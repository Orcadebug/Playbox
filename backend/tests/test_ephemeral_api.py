from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app


async def _fake_get_session() -> AsyncIterator[object]:
    yield object()


def test_ephemeral_search_route_forwards_transient_payloads(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    async def fake_init_db() -> None:
        return None

    class FakeSearchService:
        def __init__(self, session: object) -> None:
            self.session = session

        async def search(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            return {
                "query": kwargs["query"],
                "answer": None,
                "answer_error": None,
                "results": [],
                "sources": [],
                "source_errors": [],
                "execution": {"stage0_applied": False, "mrl_applied": False},
            }

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    monkeypatch.setattr("app.api.ephemeral.SearchService", FakeSearchService)
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/search/ephemeral",
                json={
                    "query": "billing refund",
                    "top_k": 3,
                    "answer_mode": "off",
                    "budget_hint": "fast",
                    "raw_sources": [
                        {
                            "id": "raw-1",
                            "name": "note.txt",
                            "content": "duplicate charge and refund request",
                        }
                    ],
                    "connector_configs": [{"connector_id": "webhook", "documents": []}],
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert captured["workspace_id"] == "ephemeral"
    assert captured["source_ids"] is None
    assert captured["include_stored_sources"] is False
    assert captured["budget_hint"] == "fast"
    assert response.json()["query"] == "billing refund"


def test_ephemeral_stream_route_emits_sse(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    async def fake_init_db() -> None:
        return None

    class FakeSearchService:
        def __init__(self, session: object) -> None:
            self.session = session

        async def stream_search(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            yield 'data: {"event":"done","results":[]}\n\n'

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    monkeypatch.setattr("app.api.ephemeral.SearchService", FakeSearchService)
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/search/ephemeral/stream",
                json={
                    "query": "billing refund",
                    "top_k": 2,
                    "raw_sources": [],
                    "connector_configs": [],
                    "answer_mode": "off",
                    "budget_hint": "auto",
                },
            ) as response:
                body = "".join(response.iter_text())
                headers = dict(response.headers)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert headers["content-type"].startswith("text/event-stream")
    assert 'data: {"event":"done","results":[]}' in body
    assert captured["workspace_id"] == "ephemeral"
    assert captured["include_stored_sources"] is False
