from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import AuthContext, get_auth_context
from app.db import get_session
from app.limits import enforce_search_limits
from app.services.search import SearchService

router = APIRouter()


class RawSourcePayload(BaseModel):
    id: str | None = None
    name: str
    content: Any
    media_type: str | None = None
    source_type: str = "raw"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConnectorConfigPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    connector_id: str
    documents: list[dict[str, Any]] | None = None
    bot_token: str | None = None
    channels: list[str] | None = None
    limit: int | None = Field(default=None, ge=1, le=1000)
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, le=100, ge=1)
    source_ids: list[str] | None = None
    workspace_id: str = "default"
    raw_sources: list[RawSourcePayload] = Field(default_factory=list)
    connector_configs: list[ConnectorConfigPayload] = Field(default_factory=list)
    include_stored_sources: bool = True
    answer_mode: Literal["off", "llm"] = "off"
    budget_hint: Literal["auto", "fast", "thorough"] = "auto"


@router.post("")
async def search(
    request: Request,
    payload: Annotated[SearchRequest, Body()],
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> dict:
    raw_sources = [item.model_dump() for item in payload.raw_sources]
    connector_configs = [item.model_dump(exclude_none=True) for item in payload.connector_configs]
    limits = enforce_search_limits(
        top_k=payload.top_k,
        raw_sources=raw_sources,
        connector_configs=connector_configs,
        auth=auth,
    )
    service = SearchService(session)
    return await service.search(
        query=payload.query,
        top_k=payload.top_k,
        source_ids=payload.source_ids,
        workspace_id=auth.workspace_id if auth.authenticated else payload.workspace_id,
        raw_sources=raw_sources,
        connector_configs=connector_configs,
        include_stored_sources=payload.include_stored_sources,
        answer_mode=payload.answer_mode,
        budget_hint=payload.budget_hint,
        request_id=request.state.request_id,
        auth_context=auth,
        limit_context=limits,
    )


@router.post("/stream")
async def search_stream(
    request: Request,
    payload: Annotated[SearchRequest, Body()],
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
) -> StreamingResponse:
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
            source_ids=payload.source_ids,
            workspace_id=auth.workspace_id if auth.authenticated else payload.workspace_id,
            raw_sources=raw_sources,
            connector_configs=connector_configs,
            include_stored_sources=payload.include_stored_sources,
            answer_mode=payload.answer_mode,
            budget_hint=payload.budget_hint,
            request_id=request.state.request_id,
            auth_context=auth,
            limit_context=limits,
        ):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")
