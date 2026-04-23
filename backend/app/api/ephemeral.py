from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.search import ConnectorConfigPayload, RawSourcePayload
from app.db import get_session
from app.services.search import SearchService

router = APIRouter()


class EphemeralSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, le=100, ge=1)
    raw_sources: list[RawSourcePayload] = Field(default_factory=list)
    connector_configs: list[ConnectorConfigPayload] = Field(default_factory=list)
    answer_mode: Literal["off", "llm"] = "off"
    budget_hint: Literal["auto", "fast", "thorough"] = "auto"


@router.post("")
async def ephemeral_search(
    payload: Annotated[EphemeralSearchRequest, Body()],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    service = SearchService(session)
    return await service.search(
        query=payload.query,
        top_k=payload.top_k,
        source_ids=None,
        workspace_id="ephemeral",
        raw_sources=[item.model_dump() for item in payload.raw_sources],
        connector_configs=[
            item.model_dump(exclude_none=True) for item in payload.connector_configs
        ],
        include_stored_sources=False,
        answer_mode=payload.answer_mode,
        budget_hint=payload.budget_hint,
    )


@router.post("/stream")
async def ephemeral_search_stream(
    payload: Annotated[EphemeralSearchRequest, Body()],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> StreamingResponse:
    service = SearchService(session)

    async def event_stream() -> AsyncIterator[str]:
        async for chunk in service.stream_search(
            query=payload.query,
            top_k=payload.top_k,
            source_ids=None,
            workspace_id="ephemeral",
            raw_sources=[item.model_dump() for item in payload.raw_sources],
            connector_configs=[
                item.model_dump(exclude_none=True) for item in payload.connector_configs
            ],
            include_stored_sources=False,
            answer_mode=payload.answer_mode,
            budget_hint=payload.budget_hint,
        ):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")
