from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import AuthContext, get_auth_context
from app.config import get_settings
from app.db import get_session
from app.limits import enforce_search_limits
from app.models import Corpus
from app.services.corpora import CorpusService
from app.services.search import SearchService
from app.services.sources import SourceService

router = APIRouter()

_ALLOWED_EXTENSIONS = {".csv", ".json", ".txt", ".md", ".pdf", ".html", ".htm", ".log", ".ndjson"}


class CreateCorpusRequest(BaseModel):
    name: str = Field(default="agent-session", min_length=1, max_length=255)
    retention: Literal["session", "persistent"] = "session"
    ttl_seconds: int | None = None


class CorpusSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, le=100, ge=1)
    stream: bool = False
    answer_mode: Literal["off", "llm"] = "off"
    budget_hint: Literal["auto", "fast", "thorough"] = "auto"


def _workspace(auth: AuthContext) -> str:
    return auth.workspace_id


def _validate_upload(file_name: str, byte_count: int) -> None:
    settings = get_settings()
    if byte_count > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File '{file_name}' exceeds {max_mb} MB limit",
        )
    ext = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if ext and ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{ext}' not supported",
        )


def _attach_corpus(response: dict, corpus: Corpus) -> dict:
    response.setdefault("execution", {})["corpus"] = CorpusService.serialize(corpus)
    return response


async def _stream_with_corpus(events: AsyncIterator[str], corpus: Corpus) -> AsyncIterator[str]:
    corpus_payload = CorpusService.serialize(corpus)
    first_hit_emitted = False
    async for chunk in events:
        if not chunk.startswith("data: "):
            yield chunk
            continue
        try:
            payload = json.loads(chunk[6:].strip())
        except json.JSONDecodeError:
            yield chunk
            continue
        if isinstance(payload, dict):
            payload.setdefault("execution", {})["corpus"] = corpus_payload
            results = payload.get("results")
            if not first_hit_emitted and isinstance(results, list) and results:
                first_hit_emitted = True
                first_hit = {
                    "event": "first_hit",
                    "result": results[0],
                    "execution": {"corpus": corpus_payload},
                }
                yield f"data: {json.dumps(first_hit)}\n\n"
            yield f"data: {json.dumps(payload)}\n\n"
        else:
            yield chunk


@router.post("")
async def create_corpus(
    payload: CreateCorpusRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> dict:
    corpus = await CorpusService(session).create_corpus(
        workspace_id=_workspace(auth),
        name=payload.name,
        retention=payload.retention,
        ttl_seconds=payload.ttl_seconds,
    )
    return {"corpus": corpus}


@router.get("")
async def list_corpora(
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> dict:
    return {"corpora": await CorpusService(session).list_corpora(workspace_id=_workspace(auth))}


@router.post("/{corpus_id}/sources")
async def upload_corpus_sources(
    corpus_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    files: list[UploadFile] | None = File(default=None),
    raw_text: str | None = Form(default=None),
    raw_text_name: str | None = Form(default=None),
    query: str | None = Form(default=None),
    top_k: int = Form(default=10),
    stream: bool = Form(default=False),
    answer_mode: Literal["off", "llm"] = Form(default="off"),
    budget_hint: Literal["auto", "fast", "thorough"] = Form(default="auto"),
):
    if not files and not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one file or raw_text",
        )

    workspace_id = _workspace(auth)
    corpus_service = CorpusService(session)
    corpus = await corpus_service.get_active_corpus(corpus_id=corpus_id, workspace_id=workspace_id)
    source_service = SourceService(session)
    created_sources: list[dict] = []

    for upload in files or []:
        payload = await upload.read()
        file_name = upload.filename or "upload.bin"
        _validate_upload(file_name, len(payload))
        created_sources.append(
            await source_service.create_from_bytes(
                file_name=file_name,
                content=payload,
                media_type=upload.content_type,
                workspace_id=workspace_id,
                corpus_id=corpus.id,
            )
        )

    if raw_text:
        created_sources.append(
            await source_service.create_from_text(
                name=raw_text_name or "Pasted text",
                text=raw_text,
                workspace_id=workspace_id,
                corpus_id=corpus.id,
            )
        )

    await corpus_service.refresh_counts(corpus_id=corpus.id, workspace_id=workspace_id)
    corpus = await corpus_service.get_active_corpus(corpus_id=corpus.id, workspace_id=workspace_id)

    if not query:
        return {"corpus": CorpusService.serialize(corpus), "sources": created_sources}

    search_payload = CorpusSearchRequest(
        query=query,
        top_k=top_k,
        stream=stream,
        answer_mode=answer_mode,
        budget_hint=budget_hint,
    )
    return await _search_corpus(
        corpus=corpus,
        payload=search_payload,
        session=session,
        auth=auth,
    )


@router.post("/{corpus_id}/search")
async def search_corpus(
    corpus_id: str,
    payload: CorpusSearchRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
):
    corpus = await CorpusService(session).get_active_corpus(
        corpus_id=corpus_id,
        workspace_id=_workspace(auth),
    )
    return await _search_corpus(corpus=corpus, payload=payload, session=session, auth=auth)


@router.delete("/{corpus_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_corpus(
    corpus_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> None:
    deleted = await CorpusService(session).delete_corpus(
        corpus_id=corpus_id,
        workspace_id=_workspace(auth),
    )
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Corpus not found")


async def _search_corpus(
    *,
    corpus: Corpus,
    payload: CorpusSearchRequest,
    session: AsyncSession,
    auth: AuthContext,
):
    source_ids = [source.id for source in corpus.sources]
    limits = enforce_search_limits(
        top_k=payload.top_k,
        raw_sources=[],
        connector_configs=[],
        auth=auth,
    )
    service = SearchService(session)
    if payload.stream:
        events = service.stream_search(
            query=payload.query,
            top_k=payload.top_k,
            source_ids=source_ids,
            workspace_id=corpus.workspace_id,
            raw_sources=[],
            connector_configs=[],
            include_stored_sources=True,
            answer_mode=payload.answer_mode,
            budget_hint=payload.budget_hint,
            auth_context=auth,
            limit_context=limits,
        )
        return StreamingResponse(
            _stream_with_corpus(events, corpus),
            media_type="text/event-stream",
        )
    response = await service.search(
        query=payload.query,
        top_k=payload.top_k,
        source_ids=source_ids,
        workspace_id=corpus.workspace_id,
        raw_sources=[],
        connector_configs=[],
        include_stored_sources=True,
        answer_mode=payload.answer_mode,
        budget_hint=payload.budget_hint,
        auth_context=auth,
        limit_context=limits,
    )
    return _attach_corpus(response, corpus)
