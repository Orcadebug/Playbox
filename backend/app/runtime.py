from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.retrieval.rust_core import get_attr
from app.retrieval.sparse_projection import (
    DeterministicSemanticProjection,
    MrlProjection,
    SparseProjection,
    SpladeProjection,
)

_RUST_EXPORTS = (
    "mrl_encode",
    "prefilter_windows",
    "extract_direct_embeddings",
    "visit_trigrams_avx512",
)


@dataclass(slots=True)
class ArtifactCheck:
    name: str
    path: str
    present: bool
    sha256_prefix: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "present": self.present,
            "sha256_prefix": self.sha256_prefix,
        }


@lru_cache(maxsize=32)
def _sha256_prefix_cached(path_value: str, size: int, mtime_ns: int) -> str | None:
    del size, mtime_ns
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def _sha256_prefix(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    stat = path.stat()
    return _sha256_prefix_cached(str(path), stat.st_size, stat.st_mtime_ns)


def _reranker_model_path(settings: Settings) -> Path:
    model_name = settings.reranker_model.replace("/", "-")
    model_file = "model_quantized.onnx" if settings.reranker_quantized else "model.onnx"
    return Path(settings.model_dir) / model_name / model_file


def artifact_checks(settings: Settings | None = None) -> dict[str, ArtifactCheck]:
    settings = settings or get_settings()
    paths = {
        "projection": Path(settings.projection_model_path)
        if settings.projection_model_path is not None
        else Path("__missing_projection__"),
        "reranker": _reranker_model_path(settings),
        "reranker_tokenizer": _reranker_model_path(settings).parent / "tokenizer.json",
        "mrl": Path(settings.model_dir) / "mrl" / "model.onnx",
        "mrl_tokenizer": Path(settings.model_dir) / "mrl" / "tokenizer.json",
    }
    return {
        name: ArtifactCheck(
            name=name,
            path=str(path),
            present=path.exists(),
            sha256_prefix=_sha256_prefix(path),
        )
        for name, path in paths.items()
    }


def artifact_versions(settings: Settings | None = None) -> dict[str, dict[str, object]]:
    return {name: check.to_dict() for name, check in artifact_checks(settings).items()}


def rust_export_status() -> dict[str, bool]:
    return {name: callable(get_attr(name)) for name in _RUST_EXPORTS}


def model_mode(
    projection: object | None = None,
    mrl_projection: object | None = None,
    reranker: object | None = None,
) -> str:
    real_projection = isinstance(projection, (SparseProjection, SpladeProjection))
    real_mrl = isinstance(mrl_projection, MrlProjection) and not mrl_projection.using_fallback
    real_reranker = reranker is not None and not bool(getattr(reranker, "is_fallback", True))
    if real_projection and real_mrl and real_reranker:
        return "real_neural"
    if real_projection or real_mrl or real_reranker:
        return "mixed"
    if isinstance(projection, DeterministicSemanticProjection):
        return "deterministic_dev"
    return "fallback"


def readiness_status(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or get_settings()
    artifacts = artifact_versions(settings)
    rust = rust_export_status()
    required = ["projection", "reranker", "reranker_tokenizer", "mrl", "mrl_tokenizer"]
    missing = [name for name in required if not artifacts[name]["present"]]
    missing_rust = [name for name, ok in rust.items() if not ok]
    ready = not missing and not missing_rust
    return {
        "ready": ready,
        "production_mode": settings.waver_production_mode,
        "artifacts": artifacts,
        "rust_exports": rust,
        "missing": missing,
        "missing_rust_exports": missing_rust,
    }


def validate_production_requirements() -> None:
    settings = get_settings()
    if not settings.waver_production_mode:
        return
    status = readiness_status(settings)
    if not status["ready"]:
        missing = ", ".join([*status["missing"], *status["missing_rust_exports"]])
        raise RuntimeError(f"WAVER_PRODUCTION_MODE requires real artifacts/exports: {missing}")
