from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

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
    waver_retriever: Literal["bm25", "sps", "cortical"] = "sps"
    waver_sps_alpha: float = 0.6
    waver_sps_candidate_multiplier: int = 3
    waver_sps_negation: bool = True
    waver_adaptive_budget: bool = True
    waver_budget_spread_threshold: float = 0.15
    waver_budget_expansion_factor: int = 3
    waver_multihead: bool = True
    waver_rrf_k: int = 60
    waver_rust_rrf: bool = False
    waver_rust_rrf_shadow: bool = False
    waver_candidate_cap: int = 2000
    waver_trie_max_patterns: int = 2000
    waver_trie_phrase_ngram: int = 2
    waver_trie_phrase_weight: float = 1.5
    projection_model_path: Path | None = Path("models/projection.npz")
    projection_dim: int = 256
    projection_hash_features: int = 262144
    waver_diffusion_steps: int = 3
    waver_diffusion_beta: float = 0.5
    waver_diffusion_gamma: float = 0.3
    waver_diffusion_delta: float = 0.2
    waver_gating_m: int = 80
    waver_lambda_l: float = 0.35
    waver_lambda_s: float = 0.35
    waver_lambda_c: float = 0.20
    waver_lambda_cost: float = 0.10
    waver_adjacency_max_edges: int = 8
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
