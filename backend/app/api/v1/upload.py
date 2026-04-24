from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import AuthContext, get_auth_context
from app.config import get_settings
from app.db import get_session
from app.services.sources import SourceService

router = APIRouter()

_ALLOWED_EXTENSIONS = {".csv", ".json", ".txt", ".md", ".pdf", ".html", ".htm", ".log", ".ndjson"}


@router.post("")
async def upload_sources(
    files: list[UploadFile] | None = File(default=None),
    raw_text: str | None = Form(default=None),
    raw_text_name: str | None = Form(default=None),
    workspace_id: str = Form(default="default"),
    session: AsyncSession = Depends(get_session),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    if not files and not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one file or raw_text",
        )

    settings = get_settings()
    service = SourceService(session)
    created_sources: list[dict] = []

    for upload in files or []:
        payload = await upload.read()

        if len(payload) > settings.max_upload_bytes:
            max_mb = settings.max_upload_bytes // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{upload.filename}' exceeds {max_mb} MB limit",
            )

        file_name = upload.filename or "upload.bin"
        ext = "." + file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if ext and ext not in _ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"File type '{ext}' not supported",
            )

        created = await service.create_from_bytes(
            file_name=file_name,
            content=payload,
            media_type=upload.content_type,
            workspace_id=auth.workspace_id if auth.authenticated else workspace_id,
        )
        created_sources.append(created)

    if raw_text:
        created = await service.create_from_text(
            name=raw_text_name or "Pasted text",
            text=raw_text,
            workspace_id=auth.workspace_id if auth.authenticated else workspace_id,
        )
        created_sources.append(created)

    return {"sources": created_sources}
