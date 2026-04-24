from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.auth import generate_api_key, hash_api_key
from app.db import get_session
from app.main import app
from app.models import ApiKey, Base, Corpus, Document, Source


@pytest.fixture
async def session_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'corpora.db'}", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async def override_session() -> AsyncIterator[AsyncSession]:
        async with factory() as session:
            yield session

    async def fake_init_db() -> None:
        return None

    monkeypatch.setattr("app.main.init_db", fake_init_db)
    app.dependency_overrides[get_session] = override_session
    try:
        yield factory
    finally:
        app.dependency_overrides.clear()
        await engine.dispose()


async def _count(session_factory: async_sessionmaker[AsyncSession], model: type[Base]) -> int:
    async with session_factory() as session:
        return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def _expire_corpus(
    session_factory: async_sessionmaker[AsyncSession],
    corpus_id: str,
) -> None:
    async with session_factory() as session:
        corpus = await session.get(Corpus, corpus_id)
        assert corpus is not None
        corpus.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        await session.commit()


async def _create_key(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    workspace_id: str,
) -> str:
    token = generate_api_key()
    async with session_factory() as session:
        session.add(
            ApiKey(
                workspace_id=workspace_id,
                key_hash=hash_api_key(token),
                requests_per_minute=60,
                bytes_per_minute=10_000_000,
            )
        )
        await session.commit()
    return token


async def test_session_corpus_upload_and_repeated_search(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with TestClient(app) as client:
        created = client.post("/v1/corpora", json={"name": "agent-session"})
        assert created.status_code == 200
        corpus = created.json()["corpus"]
        assert corpus["retention"] == "session"
        assert corpus["ttl_seconds"] == 86_400

        uploaded = client.post(
            f"/v1/corpora/{corpus['id']}/sources",
            data={
                "raw_text": "Ticket 48291: customer was charged twice and will cancel today.",
                "raw_text_name": "support_tickets.ndjson",
            },
        )
        assert uploaded.status_code == 200
        uploaded_payload = uploaded.json()
        assert uploaded_payload["corpus"]["source_count"] == 1
        assert uploaded_payload["sources"][0]["corpus_id"] == corpus["id"]

        for _ in range(2):
            searched = client.post(
                f"/v1/corpora/{corpus['id']}/search",
                json={"query": "duplicate charge cancel account", "top_k": 3},
            )
            assert searched.status_code == 200
            body = searched.json()
            assert body["results"]
            assert body["execution"]["corpus"]["id"] == corpus["id"]
            assert body["results"][0]["source_id"] == uploaded_payload["sources"][0]["id"]

    assert await _count(session_factory, Corpus) == 1
    assert await _count(session_factory, Source) == 1
    assert await _count(session_factory, Document) == 1


async def test_persistent_corpus_is_rejected(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    del session_factory
    with TestClient(app) as client:
        response = client.post(
            "/v1/corpora",
            json={"name": "kb", "retention": "persistent"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "persistent_kb_not_enabled"


async def test_expired_corpora_are_hidden_and_not_searchable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with TestClient(app) as client:
        created = client.post("/v1/corpora", json={"name": "short-lived"})
        corpus_id = created.json()["corpus"]["id"]
        client.post(
            f"/v1/corpora/{corpus_id}/sources",
            data={"raw_text": "refund requested", "raw_text_name": "ticket.txt"},
        )

        await _expire_corpus(session_factory, corpus_id)

        listed = client.get("/v1/corpora")
        assert listed.status_code == 200
        assert all(item["id"] != corpus_id for item in listed.json()["corpora"])

        searched = client.post(
            f"/v1/corpora/{corpus_id}/search",
            json={"query": "refund"},
        )
        assert searched.status_code in {404, 410}


async def test_delete_corpus_removes_sources_and_documents(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    with TestClient(app) as client:
        created = client.post("/v1/corpora", json={"name": "delete-me"})
        corpus_id = created.json()["corpus"]["id"]
        client.post(
            f"/v1/corpora/{corpus_id}/sources",
            data={"raw_text": "billing error", "raw_text_name": "ticket.txt"},
        )
        deleted = client.delete(f"/v1/corpora/{corpus_id}")

    assert deleted.status_code == 204
    assert await _count(session_factory, Corpus) == 0
    assert await _count(session_factory, Source) == 0
    assert await _count(session_factory, Document) == 0


async def test_corpus_search_is_workspace_scoped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    acme_token = await _create_key(session_factory, workspace_id="acme")
    other_token = await _create_key(session_factory, workspace_id="other")

    with TestClient(app) as client:
        created = client.post(
            "/v1/corpora",
            headers={"authorization": f"Bearer {acme_token}"},
            json={"name": "acme-session"},
        )
        assert created.status_code == 200
        corpus_id = created.json()["corpus"]["id"]

        blocked = client.post(
            f"/v1/corpora/{corpus_id}/search",
            headers={"authorization": f"Bearer {other_token}"},
            json={"query": "billing"},
        )

    assert blocked.status_code == 404
    assert blocked.json()["detail"]["code"] == "corpus_not_found"
