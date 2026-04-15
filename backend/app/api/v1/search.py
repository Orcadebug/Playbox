from collections.abc import AsyncIterator
import json

from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.services.search import SearchService

router = APIRouter()


@router.post("")
async def search(
    payload: dict = Body(...),
    session: AsyncSession = Depends(get_session),
) -> dict:
    service = SearchService(session)
    return await service.search(
        query=payload["query"],
        top_k=payload.get("top_k", 10),
        source_ids=payload.get("source_ids"),
        workspace_id=payload.get("workspace_id", "default"),
        skip_answer=payload.get("skip_answer", False),
    )


@router.post("/stream")
async def search_stream(
    payload: dict = Body(...),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    service = SearchService(session)

    async def event_stream() -> AsyncIterator[str]:
        result = await service.search(
            query=payload["query"],
            top_k=payload.get("top_k", 10),
            source_ids=payload.get("source_ids"),
            workspace_id=payload.get("workspace_id", "default"),
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

