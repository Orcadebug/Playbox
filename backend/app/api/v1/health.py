from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.runtime import readiness_status

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(session: Annotated[AsyncSession, Depends(get_session)]) -> dict:
    status = readiness_status()
    db_ok = True
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False
    return {
        "status": "ready" if status["ready"] and db_ok else "degraded",
        "database": {"ok": db_ok},
        **status,
    }
