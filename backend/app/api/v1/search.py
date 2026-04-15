from collections.abc import AsyncIterator
import json

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
    skip_answer: bool = False


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
        skip_answer=payload.skip_answer,
    )


@router.post("/stream")
async def search_stream(
    payload: SearchRequest = Body(...),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    service = SearchService(session)

    async def event_stream() -> AsyncIterator[str]:
        result = await service.search(
            query=payload.query,
            top_k=payload.top_k,
            source_ids=payload.source_ids,
            workspace_id=payload.workspace_id,
            skip_answer=False,
        )
        answer = result.get("answer") or {}
        text = answer.get("markdown", "")
        yield f"event: meta\ndata: {json.dumps({k: v for k, v in result.items() if k != 'answer'})}\n\n"
        if text:
            for token in text.split():
                yield f"event: token\ndata: {json.dumps({'token': token + ' '})}\n\n"
        yield f"event: done\ndata: {json.dumps(result)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

