from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Waver"
    api_prefix: str = "/api/v1"
    debug: bool = False
    database_url: str = "sqlite+aiosqlite:///./waver.db"
    openrouter_api_key: str | None = None
    anthropic_api_key: str | None = None
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    default_workspace: str = "default"
    model_dir: str = "models"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32
    reranker_max_length: int = 512
    reranker_quantized: bool = True
    reranker_max_candidates: int = 200
    bm25_cache_ttl: float = 300.0
    bm25_cache_max_entries: int = 10
    bm25_use_stemming: bool = True
    bm25_use_stopwords: bool = True
    confidence_threshold: float = 0.85
    streaming_chunk_threshold: int = 50
    connector_fetch_timeout: float = 10.0
    connector_max_documents: int = 200
    enable_live_connectors: bool = False
    max_upload_bytes: int = 50 * 1024 * 1024  # 50 MB
    allowed_media_types: set[str] = Field(
        default_factory=lambda: {
            "text/plain",
            "text/markdown",
            "text/csv",
            "application/json",
            "application/pdf",
            "text/html",
        }
    )

    @classmethod
    def parse_env_var(cls, field_name: str, raw_value: Any) -> Any:
        if field_name == "cors_origins" and isinstance(raw_value, str):
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        return raw_value


@lru_cache
def get_settings() -> Settings:
    return Settings()

