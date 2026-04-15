from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.services.sources import SourceService

router = APIRouter()


@router.get("")
async def list_sources(
    workspace_id: str = Query(default="default"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    return {"sources": await SourceService(session).list_sources(workspace_id=workspace_id)}


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: str,
    session: AsyncSession = Depends(get_session),
) -> None:
    deleted = await SourceService(session).delete_source(source_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source not found")

