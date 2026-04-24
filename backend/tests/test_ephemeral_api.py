from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi.testclient import TestClient

from app.db import get_session
from app.main import app


async def _fake_get_session() -> AsyncIterator[object]:
    yield object()


@dataclass
class _FakeSettings:
    waver_ghost_enabled: bool = True


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
    assert response.json()["execution"]["ghost_proxy"]["enabled"] is True


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


def test_ephemeral_route_ingests_stream_before_search(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []
    searched = False

    async def fake_init_db() -> None:
        return None

    class FakeGhostProxy:
        def ingest(self, payload_id: str, chunks: list[str]) -> None:
            assert not searched
            calls.append((payload_id, chunks))

        def query(self, payload_id: str, query: str):  # noqa: ANN001
            from waver_ghost import GhostVerdict

            return GhostVerdict(maybe_hit=True, bloom_overlap=1, cms_score=1)

    class FakeSearchService:
        def __init__(self, session: object) -> None:
            self.session = session

        async def search(self, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal searched
            searched = True
            return {
                "query": kwargs["query"],
                "answer": None,
                "answer_error": None,
                "results": [],
                "sources": [],
                "source_errors": [],
                "execution": {},
            }

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    monkeypatch.setattr("app.api.ephemeral.SearchService", FakeSearchService)
    monkeypatch.setattr("app.api.ephemeral._ghost_proxy", FakeGhostProxy())
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/search/ephemeral",
                json={
                    "query": "billing refund",
                    "raw_sources": [
                        {
                            "id": "raw-1",
                            "name": "note.txt",
                            "content": "duplicate billing refund",
                        }
                    ],
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert searched is True
    assert calls
    assert any(payload_id == "raw-1" for payload_id, _ in calls)


def test_ephemeral_route_respects_ghost_disabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, list[str]]] = []

    async def fake_init_db() -> None:
        return None

    class FakeGhostProxy:
        def ingest(self, payload_id: str, chunks: list[str]) -> None:
            calls.append((payload_id, chunks))

        def query(self, payload_id: str, query: str):  # noqa: ANN001
            raise AssertionError("ghost query should be skipped")

    class FakeSearchService:
        def __init__(self, session: object) -> None:
            self.session = session

        async def search(self, **kwargs):  # type: ignore[no-untyped-def]
            return {
                "query": kwargs["query"],
                "answer": None,
                "answer_error": None,
                "results": [],
                "sources": [],
                "source_errors": [],
                "execution": {},
            }

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    monkeypatch.setattr("app.api.ephemeral.SearchService", FakeSearchService)
    monkeypatch.setattr("app.api.ephemeral._ghost_proxy", FakeGhostProxy())
    monkeypatch.setattr(
        "app.api.ephemeral.get_settings",
        lambda: _FakeSettings(waver_ghost_enabled=False),
    )
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/search/ephemeral",
                json={"query": "billing", "raw_sources": [{"name": "n", "content": "billing"}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert calls == []
    assert response.json()["execution"]["ghost_proxy"] == {"enabled": False}


def test_ephemeral_route_rejects_malformed_streamed_json(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def fake_init_db() -> None:
        return None

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/search/ephemeral",
                content=b'{"query":',
                headers={"content-type": "application/json"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422


def test_ephemeral_raw_stream_emits_progress_and_search_events(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    captured: dict[str, object] = {}

    async def fake_init_db() -> None:
        return None

    class FakeSearchService:
        def __init__(self, session: object) -> None:
            self.session = session

        async def stream_search(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            yield 'data: {"event":"done","results":[],"execution":{}}\n\n'

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    monkeypatch.setattr("app.api.ephemeral.SearchService", FakeSearchService)
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/search/ephemeral/raw-stream",
                content=b"Acme has duplicate billing refund issue\n",
                headers={
                    "x-waver-query": "billing refund",
                    "x-waver-top-k": "2",
                    "x-waver-source-name": "ticket.txt",
                    "content-type": "text/plain",
                },
            ) as response:
                body = "".join(response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert '"event": "ingest_progress"' in body
    assert '"event":"done"' in body
    assert captured["query"] == "billing refund"
    assert captured["top_k"] == 2
    assert captured["raw_sources"][0]["content"].startswith("Acme has")


def test_live_search_alias_emits_first_hit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def fake_init_db() -> None:
        return None

    class FakeSearchService:
        def __init__(self, session: object) -> None:
            self.session = session

        async def stream_search(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            yield (
                'data: {"event":"exact_results","results":[{"source_name":"ticket.txt",'
                '"primary_span":{"text":"duplicate billing refund"}}],"execution":{}}\n\n'
            )
            yield 'data: {"event":"done","results":[],"execution":{}}\n\n'

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    monkeypatch.setattr("app.api.ephemeral.SearchService", FakeSearchService)
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/live-search",
                content=b"duplicate billing refund\n",
                headers={
                    "x-waver-query": "billing refund",
                    "content-type": "text/plain",
                },
            ) as response:
                body = "".join(response.iter_text())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert '"event": "first_hit"' in body
    assert '"event":"exact_results"' in body


def test_ephemeral_rejects_raw_payload_over_limit(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    async def fake_init_db() -> None:
        return None

    class TinySettings:
        waver_ghost_enabled = False
        waver_max_raw_bytes = 4

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    monkeypatch.setattr("app.api.ephemeral.get_settings", lambda: TinySettings())
    app.dependency_overrides[get_session] = _fake_get_session

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/search/ephemeral",
                json={"query": "billing", "raw_sources": [{"name": "n", "content": "x" * 200000}]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "raw_payload_too_large"
