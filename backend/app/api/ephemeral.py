from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.search import ConnectorConfigPayload, RawSourcePayload
from app.auth import AuthContext, get_auth_context, quota_limiter
from app.config import get_settings
from app.db import get_session
from app.errors import api_error
from app.limits import enforce_search_limits
from app.services.search import SearchService
from waver_ghost import GhostProxy, GhostVerdict

router = APIRouter()
_ghost_proxy = GhostProxy()


async def _single_sse_event(payload_json: str) -> AsyncIterator[str]:
    yield f"data: {payload_json}\n\n"


class EphemeralSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, le=100, ge=1)
    raw_sources: list[RawSourcePayload] = Field(default_factory=list)
    connector_configs: list[ConnectorConfigPayload] = Field(default_factory=list)
    answer_mode: Literal["off", "llm"] = "off"
    budget_hint: Literal["auto", "fast", "thorough"] = "auto"


def _content_to_text(content: Any) -> str:
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="ignore")
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    if content is None:
        return ""
    return str(content)


async def _read_streaming_payload(request: Request) -> tuple[EphemeralSearchRequest, str]:
    request_payload_id = f"ephemeral-request:{uuid4()}"
    chunks: list[bytes] = []
    settings = get_settings()
    ghost_enabled = settings.waver_ghost_enabled
    max_raw_bytes = getattr(settings, "waver_max_raw_bytes", 1 * 1024 * 1024)
    bytes_loaded = 0

    async for chunk in request.stream():
        if not chunk:
            continue
        bytes_loaded += len(chunk)
        if bytes_loaded > max_raw_bytes + 64 * 1024:
            raise api_error(
                413,
                "raw_payload_too_large",
                "Request body exceeds the beta byte limit",
                details={"limit": max_raw_bytes},
            )
        chunks.append(chunk)
        if ghost_enabled:
            text = chunk.decode("utf-8", errors="ignore")
            if text.strip():
                _ghost_proxy.ingest(request_payload_id, [text])

    body = b"".join(chunks)
    try:
        payload = EphemeralSearchRequest.model_validate_json(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Invalid ephemeral search payload") from exc

    if ghost_enabled:
        for index, raw_source in enumerate(payload.raw_sources):
            payload_id = str(raw_source.id or f"raw:{index}:{raw_source.name}")
            text = _content_to_text(raw_source.content)
            if text.strip():
                _ghost_proxy.ingest(payload_id, [text])

    return payload, request_payload_id


def _ghost_payload_ids(payload: EphemeralSearchRequest, request_payload_id: str) -> list[str]:
    payload_ids = [request_payload_id]
    payload_ids.extend(
        str(raw_source.id or f"raw:{index}:{raw_source.name}")
        for index, raw_source in enumerate(payload.raw_sources)
    )
    return payload_ids


def _query_ghost(payload: EphemeralSearchRequest, request_payload_id: str) -> dict[str, Any]:
    if not get_settings().waver_ghost_enabled:
        return {"enabled": False}

    verdicts: list[GhostVerdict] = [
        _ghost_proxy.query(payload_id, payload.query)
        for payload_id in _ghost_payload_ids(payload, request_payload_id)
    ]
    return {
        "enabled": True,
        "maybe_hit": any(verdict.maybe_hit for verdict in verdicts),
        "bloom_overlap": sum(verdict.bloom_overlap for verdict in verdicts),
        "cms_score": sum(verdict.cms_score for verdict in verdicts),
        "payload_count": len(verdicts),
    }


@router.post("")
async def ephemeral_search(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> dict:
    payload, request_payload_id = await _read_streaming_payload(request)
    ghost = _query_ghost(payload, request_payload_id)
    raw_sources = [item.model_dump() for item in payload.raw_sources]
    connector_configs = [item.model_dump(exclude_none=True) for item in payload.connector_configs]
    limits = enforce_search_limits(
        top_k=payload.top_k,
        raw_sources=raw_sources,
        connector_configs=connector_configs,
        auth=auth,
    )
    service = SearchService(session)
    response = await service.search(
        query=payload.query,
        top_k=payload.top_k,
        source_ids=None,
        workspace_id=auth.workspace_id if auth.authenticated else "ephemeral",
        raw_sources=raw_sources,
        connector_configs=connector_configs,
        include_stored_sources=False,
        answer_mode=payload.answer_mode,
        budget_hint=payload.budget_hint,
        request_id=request.state.request_id,
        auth_context=auth,
        limit_context=limits,
    )
    response.setdefault("execution", {})["ghost_proxy"] = ghost
    return response


@router.post("/stream")
async def ephemeral_search_stream(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> StreamingResponse:
    payload, request_payload_id = await _read_streaming_payload(request)
    ghost = _query_ghost(payload, request_payload_id)
    raw_sources = [item.model_dump() for item in payload.raw_sources]
    connector_configs = [item.model_dump(exclude_none=True) for item in payload.connector_configs]
    limits = enforce_search_limits(
        top_k=payload.top_k,
        raw_sources=raw_sources,
        connector_configs=connector_configs,
        auth=auth,
    )
    service = SearchService(session)

    async def event_stream() -> AsyncIterator[str]:
        async for chunk in service.stream_search(
            query=payload.query,
            top_k=payload.top_k,
            source_ids=None,
            workspace_id=auth.workspace_id if auth.authenticated else "ephemeral",
            raw_sources=raw_sources,
            connector_configs=connector_configs,
            include_stored_sources=False,
            answer_mode=payload.answer_mode,
            budget_hint=payload.budget_hint,
            request_id=request.state.request_id,
            auth_context=auth,
            limit_context=limits,
        ):
            if '"event": "done"' in chunk:
                try:
                    prefix, raw_payload = chunk.split("data: ", 1)
                    del prefix
                    data = json.loads(raw_payload)
                    data.setdefault("execution", {})["ghost_proxy"] = ghost
                    yield f"data: {json.dumps(data)}\n\n"
                except Exception:
                    yield chunk
            else:
                yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/raw-stream")
async def ephemeral_raw_stream(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    x_waver_query: Annotated[str | None, Header()] = None,
    x_waver_top_k: Annotated[int, Header()] = 10,
    x_waver_source_name: Annotated[str | None, Header()] = None,
    content_type: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    query = (x_waver_query or request.query_params.get("query") or "").strip()
    if not query:
        raise api_error(
            422,
            "missing_query",
            "Provide query via X-Waver-Query header or query parameter",
        )
    if x_waver_top_k < 1:
        raise api_error(
            422,
            "invalid_top_k",
            "X-Waver-Top-K must be >= 1",
        )

    service = SearchService(session)
    settings = get_settings()
    request_payload_id = f"raw-stream:{uuid4()}"
    source_name = x_waver_source_name or "raw-stream.txt"
    media_type = content_type or "text/plain"
    chunks: list[bytes] = []
    bytes_loaded = 0
    windows_seen = 0
    progress_events: list[dict[str, object]] = []

    async for chunk in request.stream():
        if not chunk:
            continue
        bytes_loaded += len(chunk)
        if bytes_loaded > settings.waver_max_raw_bytes:
            payload = {
                "event": "error",
                "code": "raw_payload_too_large",
                "message": "Raw stream exceeds the beta byte limit",
                "limit": settings.waver_max_raw_bytes,
            }
            payload_json = json.dumps(payload)
            return StreamingResponse(
                _single_sse_event(payload_json),
                media_type="text/event-stream",
            )
        chunks.append(chunk)
        text = chunk.decode("utf-8", errors="ignore")
        if text.strip() and settings.waver_ghost_enabled:
            _ghost_proxy.ingest(request_payload_id, [text])
        windows_seen += max(1, text.count("\n")) if text.strip() else 0
        progress_events.append(
            {
                "event": "ingest_progress",
                "request_id": request.state.request_id,
                "bytes_loaded": bytes_loaded,
                "windows_seen": windows_seen,
            }
        )

    if auth.authenticated:
        quota_limiter.check_bytes(auth.api_key_id or "", bytes_loaded, auth.bytes_per_minute)
    content = b"".join(chunks).decode("utf-8", errors="replace")
    raw_sources = [
        {
            "id": request_payload_id,
            "name": source_name,
            "content": content,
            "media_type": media_type,
            "metadata": {"streamed": True},
        }
    ]
    limits = enforce_search_limits(
        top_k=x_waver_top_k,
        raw_sources=raw_sources,
        connector_configs=[],
        auth=AuthContext(workspace_id=auth.workspace_id),
    )

    async def event_stream() -> AsyncIterator[str]:
        for progress in progress_events:
            yield f"data: {json.dumps(progress)}\n\n"
        first_hit_emitted = False
        async for item in service.stream_search(
            query=query,
            top_k=x_waver_top_k,
            source_ids=None,
            workspace_id=auth.workspace_id if auth.authenticated else "ephemeral",
            raw_sources=raw_sources,
            connector_configs=[],
            include_stored_sources=False,
            answer_mode="off",
            budget_hint="fast",
            request_id=request.state.request_id,
            auth_context=auth,
            limit_context=limits,
        ):
            if not first_hit_emitted and item.startswith("data: "):
                payload = None
                try:
                    payload = json.loads(item[6:].strip())
                except Exception:
                    payload = None
                results = payload.get("results") if isinstance(payload, dict) else None
                if isinstance(results, list) and results:
                    first_hit_emitted = True
                    first_hit = {
                        "event": "first_hit",
                        "request_id": request.state.request_id,
                        "query": query,
                        "result": results[0],
                    }
                    yield f"data: {json.dumps(first_hit)}\n\n"
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")
