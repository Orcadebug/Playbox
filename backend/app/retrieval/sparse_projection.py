from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from app.retrieval.rust_core import get_attr
from app.schemas.documents import Chunk

_log = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

_ALIASES: dict[str, str] = {
    "billed": "charge",
    "billing": "charge",
    "charged": "charge",
    "charges": "charge",
    "charge": "charge",
    "payment": "charge",
    "payments": "charge",
    "invoice": "invoice",
    "invoices": "invoice",
    "factura": "invoice",
    "duplicate": "duplicate",
    "duplicated": "duplicate",
    "duplicada": "duplicate",
    "double": "duplicate",
    "twice": "duplicate",
    "refund": "refund",
    "refunded": "refund",
    "reembolso": "refund",
    "chargeback": "refund",
    "failure": "failure",
    "failed": "failure",
    "failing": "failure",
    "declined": "failure",
    "denied": "failure",
    "error": "failure",
    "timeout": "timeout",
    "timeouts": "timeout",
    "timed": "timeout",
    "latency": "timeout",
    "slow": "timeout",
    "503": "timeout",
    "ticket": "ticket",
    "tickets": "ticket",
    "incidente": "ticket",
    "issue": "ticket",
    "issues": "ticket",
    "complaint": "complaint",
    "complaints": "complaint",
    "complained": "complaint",
    "customer": "customer",
    "client": "customer",
    "cliente": "customer",
    "account": "customer",
    "launch": "launch",
    "release": "launch",
    "blocker": "blocker",
    "blocked": "blocker",
    "risk": "risk",
    "risky": "risk",
    "churn": "churn",
}


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


def _canonical_tokens(text: str) -> list[str]:
    return [_ALIASES.get(token, token) for token in _TOKEN_RE.findall(text.lower())]


def _stable_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % dim


_NEG_VERBS = frozenset({
    "not", "no", "never", "cannot", "cant", "wont", "didnt", "doesnt", "wasnt",
    "werent", "isnt", "arent", "shouldnt", "couldnt", "wouldnt",
})
_DENY_VERBS = frozenset({
    "denied", "rejected", "failed", "refused", "blocked", "unable", "error",
    "errors", "broken", "missing", "invalid",
})
_POS_VERBS = frozenset({
    "approved", "succeeded", "resolved", "fixed", "completed", "accepted",
    "confirmed", "granted", "passed",
})

_NEGATION_ENABLED = os.environ.get("WAVER_SPS_NEGATION", "true").lower() not in (
    "false", "0", "no",
)


def _augment_with_polarity(text: str) -> str:
    """Inject sentence-level polarity markers as synthetic tokens.

    When a sentence contains a negation ("not", "never"), denial verb ("denied",
    "failed"), or positive verb ("approved", "resolved"), stamp every content
    token with a polarity prefix. "refund denied" → adds "DENIED_refund";
    "refund approved" → adds "POS_refund". Opposite-polarity sentences then
    land in disjoint hash buckets instead of colliding on the shared nouns.
    """
    if not _NEGATION_ENABLED:
        return text
    lower_tokens = [t for t in re.findall(r"\w+", text.lower())]
    if not lower_tokens:
        return text

    prefixes: list[str] = []
    token_set = set(lower_tokens)
    if token_set & _NEG_VERBS:
        prefixes.append("NEG_")
    if token_set & _DENY_VERBS:
        prefixes.append("DENIED_")
    if token_set & _POS_VERBS:
        prefixes.append("POS_")
    if not prefixes:
        return text

    polarity_words = _NEG_VERBS | _DENY_VERBS | _POS_VERBS
    content = [
        t for t in lower_tokens
        if t not in polarity_words and len(t) > 2
    ]
    if not content:
        return text

    synthetic = [f"{p}{t}" for p in prefixes for t in content]
    return f"{text} {' '.join(synthetic)}"


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
        augmented_texts = [_augment_with_polarity(t) for t in texts]
        hashed = self._vectorizer.transform(augmented_texts).astype(np.float32)
        if self._matrix_is_transposed:
            projected = hashed @ self.W.T
        else:
            projected = hashed @ self.W
        if sparse.issparse(projected):
            projected = projected.toarray()
        return _normalize_rows(np.asarray(projected, dtype=np.float32))


@dataclass(slots=True)
class DeterministicSemanticProjection:
    """Always-available semantic scorer for fresh sources.

    It is deliberately lightweight: canonical token aliases plus stable hashed
    unigrams/bigrams. A learned projection can improve quality, but this keeps
    semantic recall from depending on a setup step.
    """

    config: ProjectionConfig = field(default_factory=ProjectionConfig)

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed(query)

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return np.vstack([self._embed(chunk.text) for chunk in chunks])

    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
        if len(chunk_vecs) == 0:
            return np.zeros(0, dtype=np.float32)
        scores = chunk_vecs @ query_vec
        return np.maximum(scores.astype(np.float32), 0.0)

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.config.dim, dtype=np.float32)
        augmented = _augment_with_polarity(text)
        tokens = _canonical_tokens(augmented)
        if not tokens:
            return vector

        features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
        features.extend(
            (f"{left} {right}", 1.35) for left, right in zip(tokens, tokens[1:], strict=False)
        )
        # Unique concepts matter more than repeated noise in messy logs/payloads.
        seen: set[str] = set()
        for feature, weight in features:
            if feature in seen:
                continue
            seen.add(feature)
            vector[_stable_index(feature, self.config.dim)] += weight
        return _normalize_rows(vector)


@dataclass(slots=True)
class NullProjection:
    config: ProjectionConfig = field(default_factory=ProjectionConfig)

    def encode_query(self, query: str) -> np.ndarray:
        return np.zeros(self.config.dim, dtype=np.float32)

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        return np.zeros((len(chunks), self.config.dim), dtype=np.float32)

    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
        return np.zeros(len(chunk_vecs), dtype=np.float32)


@dataclass(slots=True)
class SpladeProjection:
    model_path: Path | None
    config: ProjectionConfig = field(default_factory=ProjectionConfig)
    fallback: DeterministicSemanticProjection = field(
        default_factory=DeterministicSemanticProjection
    )
    using_fallback: bool = False
    fallback_reason: str | None = None

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True)[0]

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return self._encode([chunk.text for chunk in chunks], is_query=False)

    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
        if len(chunk_vecs) == 0:
            return np.zeros(0, dtype=np.float32)
        scores = chunk_vecs @ query_vec
        return np.maximum(scores.astype(np.float32), 0.0)

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        splade_encode = get_attr("splade_encode")
        if self.model_path is None or not self.model_path.exists() or not callable(splade_encode):
            self.using_fallback = True
            self.fallback_reason = "splade_artifact_unavailable"
            if is_query:
                return np.vstack([self.fallback.encode_query(text) for text in texts])
            return self.fallback.encode_chunks([_StubChunk(text) for text in texts])
        try:
            rows = splade_encode(
                str(self.model_path),
                list(texts),
                self.config.dim,
                is_query,
            )
            matrix = np.asarray(rows, dtype=np.float32)
            return _normalize_rows(matrix)
        except Exception as exc:
            _log.warning("SPLADE inference unavailable (%s); using deterministic fallback", exc)
            self.using_fallback = True
            self.fallback_reason = "splade_runtime_failed"
            if is_query:
                return np.vstack([self.fallback.encode_query(text) for text in texts])
            return self.fallback.encode_chunks([_StubChunk(text) for text in texts])


@dataclass(slots=True)
class MrlProjection:
    model_path: Path | None
    config: ProjectionConfig = field(default_factory=ProjectionConfig)
    fallback: DeterministicSemanticProjection = field(
        default_factory=DeterministicSemanticProjection
    )
    query_prefix: str = ""
    document_prefix: str = ""
    using_fallback: bool = False
    fallback_reason: str | None = None

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query], is_query=True)[0]

    def encode_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self.config.dim), dtype=np.float32)
        return self._encode([chunk.text for chunk in chunks], is_query=False)

    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
        if len(chunk_vecs) == 0:
            return np.zeros(0, dtype=np.float32)
        scores = chunk_vecs @ query_vec
        return np.maximum(scores.astype(np.float32), 0.0)

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        mrl_encode = get_attr("mrl_encode")
        if self.model_path is None or not self.model_path.exists() or not callable(mrl_encode):
            self.using_fallback = True
            self.fallback_reason = "mrl_artifact_unavailable"
            if is_query:
                return np.vstack([self.fallback.encode_query(text) for text in texts])
            return self.fallback.encode_chunks([_StubChunk(text) for text in texts])
        try:
            prefix = self.query_prefix if is_query else self.document_prefix
            encoded_texts = [_with_prefix(text, prefix) for text in texts]
            rows = mrl_encode(
                str(self.model_path),
                encoded_texts,
                self.config.dim,
                is_query,
            )
            matrix = np.asarray(rows, dtype=np.float32)
            return _normalize_rows(matrix)
        except Exception as exc:
            _log.warning("MRL inference unavailable (%s); using deterministic fallback", exc)
            self.using_fallback = True
            self.fallback_reason = "mrl_runtime_failed"
            if is_query:
                return np.vstack([self.fallback.encode_query(text) for text in texts])
            return self.fallback.encode_chunks([_StubChunk(text) for text in texts])


def _with_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    return text if text.startswith(prefix) else f"{prefix}{text}"


class _StubChunk:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


def load_projection(
    path: Path | None,
    config: ProjectionConfig | None = None,
) -> SparseProjection | DeterministicSemanticProjection:
    fallback_config = config or ProjectionConfig()
    if path is None:
        _log.warning(
            "Projection model path is not configured; using deterministic semantic projection"
        )
        return DeterministicSemanticProjection(fallback_config)
    try:
        return SparseProjection.load(path)
    except FileNotFoundError:
        _log.warning(
            "Projection model not found at %s; using deterministic semantic projection", path
        )
        return DeterministicSemanticProjection(fallback_config)
    except Exception as exc:
        _log.warning(
            "Failed to load projection model at %s: %s; using deterministic semantic projection",
            path,
            exc,
        )
        return DeterministicSemanticProjection(fallback_config)


def load_best_projection(
    *,
    model_dir: Path | None,
    projection_model_path: Path | None,
    config: ProjectionConfig | None = None,
) -> SparseProjection | DeterministicSemanticProjection | SpladeProjection:
    fallback_config = config or ProjectionConfig()
    splade_model_path = model_dir / "splade" / "model.onnx" if model_dir is not None else None
    if splade_model_path is not None and splade_model_path.exists():
        return SpladeProjection(
            model_path=splade_model_path,
            config=fallback_config,
            fallback=DeterministicSemanticProjection(fallback_config),
        )
    return load_projection(projection_model_path, fallback_config)
