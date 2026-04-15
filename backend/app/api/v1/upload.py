from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.services.sources import SourceService

router = APIRouter()


@router.post("")
async def upload_sources(
    files: list[UploadFile] | None = File(default=None),
    raw_text: str | None = Form(default=None),
    raw_text_name: str | None = Form(default=None),
    workspace_id: str = Form(default="default"),
    session: AsyncSession = Depends(get_session),
) -> dict:
    if not files and not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one file or raw_text",
        )

    service = SourceService(session)
    created_sources: list[dict] = []

    for upload in files or []:
        payload = await upload.read()
        created = await service.create_from_bytes(
            file_name=upload.filename or "upload.bin",
            content=payload,
            media_type=upload.content_type,
            workspace_id=workspace_id,
        )
        created_sources.append(created)

    if raw_text:
        created = await service.create_from_text(
            name=raw_text_name or "Pasted text",
            text=raw_text,
            workspace_id=workspace_id,
        )
        created_sources.append(created)

    return {"sources": created_sources}

