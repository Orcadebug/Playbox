from collections.abc import AsyncIterator

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.models import Base

settings = get_settings()

engine = create_async_engine(settings.database_url, future=True, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
        await connection.run_sync(_ensure_runtime_columns)


def _ensure_runtime_columns(sync_connection) -> None:  # type: ignore[no-untyped-def]
    inspector = inspect(sync_connection)
    table_names = set(inspector.get_table_names())
    if "sources" not in table_names:
        return
    source_columns = {column["name"] for column in inspector.get_columns("sources")}
    if "corpus_id" not in source_columns:
        sync_connection.exec_driver_sql("ALTER TABLE sources ADD COLUMN corpus_id VARCHAR(36)")
        sync_connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_sources_corpus_id ON sources (corpus_id)"
        )
