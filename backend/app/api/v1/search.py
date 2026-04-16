from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.services.search import SearchService

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, le=100, ge=1)
    source_ids: list[str] | None = None
    workspace_id: str = "default"
    connector_configs: list[dict[str, Any]] | None = None


@router.post("")
async def search(
    payload: SearchRequest = Body(...),
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = SearchService(session)
    return await service.search(
        query=payload.query,
        top_k=payload.top_k,
        source_ids=payload.source_ids,
        workspace_id=payload.workspace_id,
        connector_configs=payload.connector_configs,
    )


@router.post("/stream")
async def search_stream(
    payload: SearchRequest = Body(...),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    service = SearchService(session)

    async def event_stream() -> AsyncIterator[str]:
        async for chunk in service.stream_search(
            query=payload.query,
            top_k=payload.top_k,
            source_ids=payload.source_ids,
            workspace_id=payload.workspace_id,
            connector_configs=payload.connector_configs,
        ):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")
