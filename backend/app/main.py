from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.ephemeral import router as ephemeral_router
from app.api.live_search import router as live_search_router
from app.api.router import api_router
from app.api.v1.corpora import router as corpora_router
from app.config import get_settings
from app.db import init_db
from app.observability import metrics, request_observability_middleware
from app.runtime import validate_production_requirements

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    validate_production_requirements()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Waver-Query",
        "X-Waver-Top-K",
        "X-Waver-Source-Name",
    ],
)


app.middleware("http")(request_observability_middleware)


@app.middleware("http")
async def security_headers(request: Request, call_next) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.get("/healthz")
async def root_healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(metrics.render_prometheus(), media_type="text/plain; version=0.0.4")


app.include_router(api_router, prefix=settings.api_prefix)
app.include_router(ephemeral_router, prefix="/v1/search/ephemeral", tags=["ephemeral-search"])
app.include_router(live_search_router, prefix="/v1/live-search", tags=["live-search"])
app.include_router(corpora_router, prefix="/v1/corpora", tags=["corpora"])
