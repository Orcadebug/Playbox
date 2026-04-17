from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from app.schemas.documents import Chunk

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class ProjectionConfig:
    hash_features: int = 262144
    dim: int = 256
    ngram_range: tuple[int, int] = (1, 2)
    version: str = "1"


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        norm = float(np.linalg.norm(values))
        return values / norm if norm > 0 else values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def _config_from_json(raw: Any) -> ProjectionConfig:
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    data = json.loads(str(raw))
    ngram_range = data.get("ngram_range", (1, 2))
    return ProjectionConfig(
        hash_features=int(data["hash_features"]),
        dim=int(data["dim"]),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
        version=str(data.get("version", "1")),
    )


def save_projection(
    path: Path,
    W: np.ndarray | sparse.csr_matrix,
    config: ProjectionConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(asdict(config))
    if sparse.issparse(W):
        csr = W.tocsr().astype(np.float32)
        np.savez(
            path,
            sparse=np.array(True),
            data=csr.data,
            indices=csr.indices,
            indptr=csr.indptr,
            shape=np.array(csr.shape),
            config=np.array(config_json),
        )
    else:
        np.savez(
            path,
            sparse=np.array(False),
            W=np.asarray(W, dtype=np.float32),
            config=np.array(config_json),
        )


class SparseProjection:
    def __init__(
        self,
        W: np.ndarray | sparse.csr_matrix,
        config: ProjectionConfig,
    ) -> None:
        self.W = W.astype(np.float32)
        self.config = config
        self._matrix_is_transposed = self._validate_shape(self.W.shape)
        self._vectorizer = HashingVectorizer(
            n_features=config.hash_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=config.ngram_range,
        )

    @classmethod
    def load(cls, path: Path) -> SparseProjection:
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as payload:
            config = _config_from_json(payload["config"])
            is_sparse = bool(payload["sparse"].item()) if "sparse" in payload else False
            if is_sparse:
                W = sparse.csr_matrix(
                    (
                        payload["data"].astype(np.float32),
                        payload["indices"],
                        payload["indptr"],
                    ),
                    shape=tuple(payload["shape"]),
                )
            else:
                W = payload["W"].astype(np.float32)
        return cls(W, config)

    def encode_query(self, query: str) -> np.ndarray:
        return self._project([query])[0]

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return self._project([chunk.text for chunk in chunks])

    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
        if len(chunk_vecs) == 0:
            return np.zeros(0, dtype=np.float32)
        scores = chunk_vecs @ query_vec
        return np.maximum(scores.astype(np.float32), 0.0)

    def _validate_shape(self, shape: tuple[int, int]) -> bool:
        expected = (self.config.hash_features, self.config.dim)
        transposed = (self.config.dim, self.config.hash_features)
        if shape == expected:
            return False
        if shape == transposed:
            return True
        raise ValueError(f"Projection matrix shape {shape} does not match {expected}")

    def _project(self, texts: Sequence[str]) -> np.ndarray:
        hashed = self._vectorizer.transform(texts).astype(np.float32)
        if self._matrix_is_transposed:
            projected = hashed @ self.W.T
        else:
            projected = hashed @ self.W
        if sparse.issparse(projected):
            projected = projected.toarray()
        return _normalize_rows(np.asarray(projected, dtype=np.float32))


@dataclass(slots=True)
class NullProjection:
    config: ProjectionConfig = field(default_factory=ProjectionConfig)

    def encode_query(self, query: str) -> np.ndarray:
        return np.zeros(self.config.dim, dtype=np.float32)

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        return np.zeros((len(chunks), self.config.dim), dtype=np.float32)

    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
        return np.zeros(len(chunk_vecs), dtype=np.float32)


def load_projection(
    path: Path | None,
    config: ProjectionConfig | None = None,
) -> SparseProjection | NullProjection:
    fallback_config = config or ProjectionConfig()
    if path is None:
        _log.warning("Projection model path is not configured; semantic projection disabled")
        return NullProjection(fallback_config)
    try:
        return SparseProjection.load(path)
    except FileNotFoundError:
        _log.warning("Projection model not found at %s; semantic projection disabled", path)
        return NullProjection(fallback_config)
    except Exception as exc:
        _log.warning(
            "Failed to load projection model at %s: %s; semantic projection disabled",
            path,
            exc,
        )
        return NullProjection(fallback_config)
