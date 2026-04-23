from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter
from typing import Protocol

from app.config import RustBm25Mode
from app.retrieval.bm25 import BM25Index, BM25ScoredChunk, BM25Tokenizer
from app.retrieval.bm25_cache import BM25IndexCache, _chunks_hash
from app.retrieval.rust_core import get_attr
from app.schemas.documents import Chunk

_log = logging.getLogger(__name__)
_RUST_RRF_MODULE = "waver_core"


class Retriever(Protocol):
    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        raise NotImplementedError


@dataclass(slots=True)
class Bm25Retriever:
    cache: BM25IndexCache | None = None
    use_stemming: bool = True
    use_stopwords: bool = True
    rust_mode: RustBm25Mode = "python"

    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        chunk_list = list(chunks)
        if not chunk_list or not query.strip():
            return []

        tokenizer = BM25Tokenizer(
            use_stemming=self.use_stemming,
            use_stopwords=self.use_stopwords,
        )
        corpus_hash = _chunks_hash(chunk_list)
        python_hits: list[BM25ScoredChunk] | None = None
        rust_hits: list[BM25ScoredChunk] | None = None

        if self.cache is not None and use_cache:
            index = self.cache.get_or_build(workspace_id, chunk_list)
        else:
            index = BM25Index(
                chunk_list,
                use_stemming=self.use_stemming,
                use_stopwords=self.use_stopwords,
            )
        if self.rust_mode != "rust":
            python_hits = index.search(query, top_k=top_k)

        if self.rust_mode != "python":
            rust_hits = _search_bm25_rust(
                query,
                chunk_list,
                top_k=top_k,
                tokenizer=tokenizer,
                cache=self.cache if use_cache else None,
                corpus_hash=corpus_hash,
            )

        if self.rust_mode == "rust":
            if rust_hits is not None:
                return rust_hits
            return python_hits if python_hits is not None else index.search(query, top_k=top_k)

        if self.rust_mode == "shadow" and rust_hits is not None and python_hits is not None:
            if _bm25_signature(python_hits) != _bm25_signature(rust_hits):
                _log.warning(
                    "BM25 parity mismatch (python_top=%s rust_top=%s)",
                    [hit.chunk.chunk_id for hit in python_hits[:5]],
                    [hit.chunk.chunk_id for hit in rust_hits[:5]],
                )

        if python_hits is None:
            return index.search(query, top_k=top_k)
        return python_hits


def _bm25_signature(hits: list[BM25ScoredChunk]) -> list[tuple[str, float]]:
    return [(hit.chunk.chunk_id, round(float(hit.score), 6)) for hit in hits]


@lru_cache(maxsize=1)
def _get_rust_bm25_class() -> Callable[..., object] | None:
    value = get_attr("RustBm25Index")
    return value if callable(value) else None


def _build_tokenized_documents(
    chunks: Sequence[Chunk],
    tokenizer: BM25Tokenizer,
) -> list[tuple[str, str]]:
    return [
        (
            chunk.chunk_id,
            " ".join(tokenizer.tokenize(chunk.text)),
        )
        for chunk in chunks
    ]


def _search_bm25_rust(
    query: str,
    chunks: Sequence[Chunk],
    *,
    top_k: int,
    tokenizer: BM25Tokenizer,
    cache: BM25IndexCache | None,
    corpus_hash: str,
) -> list[BM25ScoredChunk] | None:
    rust_bm25 = _get_rust_bm25_class()
    if rust_bm25 is None:
        return None

    tokenized_query = " ".join(tokenizer.tokenize(query))
    if not tokenized_query:
        return []

    rust_index = cache.get_rust_index(corpus_hash) if cache is not None else None
    if rust_index is None:
        try:
            rust_index = rust_bm25(_build_tokenized_documents(chunks, tokenizer))
        except Exception as exc:
            _log.warning("Rust BM25 unavailable (%s) — using Python fallback", exc)
            return None
        if cache is not None:
            cache.set_rust_index(corpus_hash, rust_index)

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    try:
        rows = rust_index.search(tokenized_query, top_k)
    except Exception as exc:
        _log.warning("Rust BM25 search failed (%s) — using Python fallback", exc)
        return None

    hits: list[BM25ScoredChunk] = []
    try:
        for chunk_id, score in rows:
            chunk = chunk_by_id.get(str(chunk_id))
            if chunk is None:
                continue
            hits.append(
                BM25ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    bm25_score=float(score),
                    rerank_score=None,
                )
            )
    except Exception as exc:
        _log.warning("Rust BM25 payload parse failed (%s) — using Python fallback", exc)
        return None
    hits.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
    return hits[:top_k]


@dataclass(slots=True)
class _FusedRrfRow:
    chunk_id: str
    score: float
    channels: list[str]
    channel_scores: dict[str, float]
    bm25_score: float | None


def _rrf_fuse_python(
    head_hits: list[tuple[str, list[BM25ScoredChunk]]],
    *,
    top_k: int,
    rrf_k: int,
) -> list[_FusedRrfRow]:
    fused: dict[str, dict[str, object]] = {}
    for name, hits in head_hits:
        for rank, hit in enumerate(hits):
            cid = hit.chunk.chunk_id
            slot = fused.setdefault(
                cid,
                {
                    "score": 0.0,
                    "channels": [],
                    "channel_scores": {},
                    "bm25_score": hit.bm25_score,
                },
            )
            slot["score"] = float(slot["score"]) + 1.0 / (rrf_k + rank + 1)
            slot["channels"].append(name)  # type: ignore[union-attr]
            slot["channel_scores"][name] = float(hit.score)  # type: ignore[index]
            # Keep the most informative bm25_score (non-None wins).
            if slot["bm25_score"] is None and hit.bm25_score is not None:
                slot["bm25_score"] = hit.bm25_score

    ordered = sorted(fused.items(), key=lambda kv: (-float(kv[1]["score"]), kv[0]))
    return [
        _FusedRrfRow(
            chunk_id=chunk_id,
            score=float(slot["score"]),
            channels=list(slot["channels"]),  # type: ignore[arg-type]
            channel_scores=dict(slot["channel_scores"]),  # type: ignore[arg-type]
            bm25_score=slot["bm25_score"],  # type: ignore[arg-type]
        )
        for chunk_id, slot in ordered[:top_k]
    ]


@lru_cache(maxsize=1)
def _get_rust_rrf_callable() -> Callable[..., object] | None:
    rrf_fuse = get_attr("rrf_fuse")
    if not callable(rrf_fuse):
        _log.warning(
            "Rust RRF unavailable (%s.rrf_fuse missing) — using Python fallback",
            _RUST_RRF_MODULE,
        )
        return None
    return rrf_fuse


def _prepare_rust_payload(
    head_hits: list[tuple[str, list[BM25ScoredChunk]]],
) -> list[tuple[str, list[tuple[str, float, float | None]]]]:
    payload: list[tuple[str, list[tuple[str, float, float | None]]]] = []
    for name, hits in head_hits:
        payload.append(
            (
                name,
                [
                    (
                        hit.chunk.chunk_id,
                        float(hit.score),
                        hit.bm25_score,
                    )
                    for hit in hits
                ],
            )
        )
    return payload


def _rrf_fuse_rust(
    head_hits: list[tuple[str, list[BM25ScoredChunk]]],
    *,
    top_k: int,
    rrf_k: int,
) -> list[_FusedRrfRow] | None:
    rrf_fuse = _get_rust_rrf_callable()
    if rrf_fuse is None:
        return None

    payload = _prepare_rust_payload(head_hits)
    try:
        rows = rrf_fuse(payload, top_k, rrf_k)
    except Exception as exc:
        _log.warning("Rust RRF call failed (%s) — using Python fallback", exc)
        return None

    parsed: list[_FusedRrfRow] = []
    try:
        for row in rows:
            chunk_id, score, channels, channel_scores, bm25_score = row
            parsed.append(
                _FusedRrfRow(
                    chunk_id=str(chunk_id),
                    score=float(score),
                    channels=[str(channel) for channel in channels],
                    channel_scores={
                        str(name): float(value) for name, value in dict(channel_scores).items()
                    },
                    bm25_score=None if bm25_score is None else float(bm25_score),
                )
            )
    except Exception as exc:
        _log.warning("Rust RRF payload parse failed (%s) — using Python fallback", exc)
        return None
    return parsed


def _fused_signature(rows: list[_FusedRrfRow]) -> list[tuple[object, ...]]:
    return [
        (
            row.chunk_id,
            round(float(row.score), 12),
            tuple(row.channels),
            tuple(
                (name, round(float(value), 12))
                for name, value in sorted(row.channel_scores.items())
            ),
            None if row.bm25_score is None else round(float(row.bm25_score), 12),
        )
        for row in rows
    ]


def _materialize_results(
    rows: list[_FusedRrfRow],
    chunk_by_id: dict[str, Chunk],
) -> list[BM25ScoredChunk]:
    results: list[BM25ScoredChunk] = []
    for row in rows:
        chunk = chunk_by_id.get(row.chunk_id)
        if chunk is None:
            continue
        results.append(
            BM25ScoredChunk(
                chunk=chunk,
                score=float(row.score),
                bm25_score=row.bm25_score,
                rerank_score=None,
                channels=list(row.channels),
                channel_scores=dict(row.channel_scores),
            )
        )
    return results


@dataclass(slots=True)
class MultiHeadRetriever:
    """Run multiple retriever heads in parallel and fuse via Reciprocal Rank
    Fusion (RRF). RRF is scale-free, so heads with different score ranges
    (BM25 unbounded, SPS in [0,1], phrase binary) combine cleanly.

    Populates each result's `channels` and `channel_scores` so API consumers
    see which head surfaced each chunk and at what raw score.
    """

    heads: list[tuple[str, Retriever]] = field(default_factory=list)
    rrf_k: int = 60
    use_rust_rrf: bool = False
    shadow_compare: bool = False

    def search(
        self,
        query: str,
        chunks: Sequence[Chunk],
        top_k: int,
        workspace_id: str,
        use_cache: bool = True,
    ) -> list[BM25ScoredChunk]:
        chunk_list = list(chunks)
        if not chunk_list or not self.heads:
            return []

        per_head: list[tuple[str, list[BM25ScoredChunk]]] = []
        for name, head in self.heads:
            per_head.append(
                (
                    name,
                    head.search(
                        query,
                        chunk_list,
                        top_k=top_k,
                        workspace_id=workspace_id,
                        use_cache=use_cache,
                    ),
                )
            )

        chunk_by_id: dict[str, Chunk] = {}
        for _, hits in per_head:
            for hit in hits:
                chunk_by_id.setdefault(hit.chunk.chunk_id, hit.chunk)
        if not chunk_by_id:
            return []

        python_rows: list[_FusedRrfRow] | None = None
        python_ms: float | None = None
        rust_rows: list[_FusedRrfRow] | None = None
        rust_ms: float | None = None

        if self.use_rust_rrf:
            rust_started_at = perf_counter()
            rust_rows = _rrf_fuse_rust(per_head, top_k=top_k, rrf_k=self.rrf_k)
            rust_ms = (perf_counter() - rust_started_at) * 1000.0
            if rust_rows is not None:
                primary_rows = rust_rows
            else:
                python_started_at = perf_counter()
                python_rows = _rrf_fuse_python(per_head, top_k=top_k, rrf_k=self.rrf_k)
                python_ms = (perf_counter() - python_started_at) * 1000.0
                primary_rows = python_rows
        else:
            python_started_at = perf_counter()
            python_rows = _rrf_fuse_python(per_head, top_k=top_k, rrf_k=self.rrf_k)
            python_ms = (perf_counter() - python_started_at) * 1000.0
            primary_rows = python_rows

        if self.shadow_compare:
            if python_rows is None:
                python_started_at = perf_counter()
                python_rows = _rrf_fuse_python(per_head, top_k=top_k, rrf_k=self.rrf_k)
                python_ms = (perf_counter() - python_started_at) * 1000.0
            if rust_rows is None:
                rust_started_at = perf_counter()
                rust_rows = _rrf_fuse_rust(per_head, top_k=top_k, rrf_k=self.rrf_k)
                rust_ms = (perf_counter() - rust_started_at) * 1000.0

            if rust_rows is None:
                _log.debug("Rust RRF shadow compare skipped: extension unavailable")
            elif _fused_signature(python_rows) != _fused_signature(rust_rows):
                _log.warning(
                    "RRF parity mismatch (python_top=%s rust_top=%s)",
                    [row.chunk_id for row in python_rows[:5]],
                    [row.chunk_id for row in rust_rows[:5]],
                )
            elif python_ms is not None and rust_ms is not None:
                _log.info(
                    "RRF parity ok (python_ms=%.3f rust_ms=%.3f)",
                    python_ms,
                    rust_ms,
                )

        return _materialize_results(primary_rows, chunk_by_id)
