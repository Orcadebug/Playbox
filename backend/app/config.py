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

    @classmethod
    def parse_env_var(cls, field_name: str, raw_value: Any) -> Any:
        if field_name == "cors_origins" and isinstance(raw_value, str):
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        return raw_value


@lru_cache
def get_settings() -> Settings:
    return Settings()

