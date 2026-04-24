from fastapi import APIRouter

from app.api.v1.corpora import router as corpora_router
from app.api.v1.health import router as health_router
from app.api.v1.search import router as search_router
from app.api.v1.sources import router as sources_router
from app.api.v1.upload import router as upload_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(corpora_router, prefix="/corpora", tags=["corpora"])
api_router.include_router(upload_router, prefix="/upload", tags=["upload"])
api_router.include_router(search_router, prefix="/search", tags=["search"])
api_router.include_router(sources_router, prefix="/sources", tags=["sources"])
