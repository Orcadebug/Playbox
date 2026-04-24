from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.ephemeral import ephemeral_raw_stream
from app.auth import AuthContext, get_auth_context
from app.db import get_session

router = APIRouter()


@router.post("")
async def live_search(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    auth: Annotated[AuthContext, Depends(get_auth_context)],
    x_waver_query: Annotated[str | None, Header()] = None,
    x_waver_top_k: Annotated[int, Header()] = 10,
    x_waver_source_name: Annotated[str | None, Header()] = None,
    content_type: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    return await ephemeral_raw_stream(
        request=request,
        session=session,
        auth=auth,
        x_waver_query=x_waver_query,
        x_waver_top_k=x_waver_top_k,
        x_waver_source_name=x_waver_source_name,
        content_type=content_type,
    )
