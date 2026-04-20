from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
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


@router.post("")
async def search(
    payload: Annotated[SearchRequest, Body()],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    service = SearchService(session)
    return await service.search(
        query=payload.query,
        top_k=payload.top_k,
        source_ids=payload.source_ids,
        workspace_id=payload.workspace_id,
        raw_sources=[item.model_dump() for item in payload.raw_sources],
        connector_configs=[
            item.model_dump(exclude_none=True) for item in payload.connector_configs
        ],
        include_stored_sources=payload.include_stored_sources,
        answer_mode=payload.answer_mode,
    )


@router.post("/stream")
async def search_stream(
    payload: Annotated[SearchRequest, Body()],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> StreamingResponse:
    service = SearchService(session)

    async def event_stream() -> AsyncIterator[str]:
        async for chunk in service.stream_search(
            query=payload.query,
            top_k=payload.top_k,
            source_ids=payload.source_ids,
            workspace_id=payload.workspace_id,
            raw_sources=[item.model_dump() for item in payload.raw_sources],
            connector_configs=[
                item.model_dump(exclude_none=True) for item in payload.connector_configs
            ],
            include_stored_sources=payload.include_stored_sources,
            answer_mode=payload.answer_mode,
        ):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")
