"""
Content-addressed BM25 and embedding cache.

Entries are keyed by a stable corpus hash derived from chunk text and offsets.
Workspaces keep only a lightweight alias to their last-seen corpus hash so
stored-source invalidation can forget the alias without forcing cache eviction.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from app.retrieval.bm25 import BM25Index
    from app.schemas.documents import Chunk

_log = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    index: BM25Index
    corpus_hash: str
    created_at: float = field(default_factory=time.monotonic)
    chunk_embeddings: np.ndarray | None = None
    rust_index: object | None = None


def _chunk_fingerprint(chunk: Chunk) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    source_start = metadata.get(
        "source_start",
        metadata.get("char_start", metadata.get("byte_start")),
    )
    source_end = metadata.get(
        "source_end",
        metadata.get("char_end", metadata.get("byte_end")),
    )
    return {
        "chunk_id": chunk.chunk_id,
        "source_name": chunk.source_name,
        "text": chunk.text,
        "parser_name": metadata.get("parser_name"),
        "source_start": source_start,
        "source_end": source_end,
        "token_start": metadata.get("token_start"),
        "token_end": metadata.get("token_end"),
        "page_number": getattr(chunk.location, "page_number", None),
        "row_number": getattr(chunk.location, "row_number", None),
        "line_number": getattr(chunk.location, "line_number", None),
        "section": getattr(chunk.location, "section", None),
    }


def _chunks_hash(chunks: list[Chunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        payload = json.dumps(
            _chunk_fingerprint(chunk),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class BM25IndexCache:
    """
    LRU-limited, TTL-expiring cache keyed by corpus hash.

    Workspace ids are tracked only as aliases to the latest observed corpus hash
    so stored-source invalidation can detach the alias without discarding a
    reusable payload entry immediately.
    """

    def __init__(
        self,
        ttl: float = 300.0,
        max_entries: int = 10,
        use_stemming: bool = True,
        use_stopwords: bool = True,
    ) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._use_stemming = use_stemming
        self._use_stopwords = use_stopwords
        self._cache: dict[str, _CacheEntry] = {}
        self._access_order: list[str] = []
        self._workspace_aliases: dict[str, str] = {}
        self._lock = threading.Lock()

    def resolve_hash(self, chunks: list[Chunk]) -> str:
        return _chunks_hash(chunks)

    def get_or_build(self, workspace_id: str, chunks: list[Chunk]) -> BM25Index:
        from app.retrieval.bm25 import BM25Index

        corpus_hash = self.resolve_hash(chunks)

        with self._lock:
            entry = self._cache.get(corpus_hash)
            now = time.monotonic()
            if entry is not None:
                age = now - entry.created_at
                if age < self._ttl:
                    self._workspace_aliases[workspace_id] = corpus_hash
                    self._touch(corpus_hash)
                    _log.debug("BM25 cache hit for corpus %s (age=%.1fs)", corpus_hash[:12], age)
                    return entry.index

        _log.debug("Building BM25 index for corpus %s (%d chunks)", corpus_hash[:12], len(chunks))
        index = BM25Index(
            chunks,
            use_stemming=self._use_stemming,
            use_stopwords=self._use_stopwords,
        )

        with self._lock:
            self._set(
                corpus_hash,
                _CacheEntry(index=index, corpus_hash=corpus_hash),
            )
            self._workspace_aliases[workspace_id] = corpus_hash
        return index

    def get_embeddings(self, corpus_hash: str):  # type: ignore[no-untyped-def]
        with self._lock:
            entry = self._cache.get(corpus_hash)
            if entry is None:
                return None
            if time.monotonic() - entry.created_at >= self._ttl:
                return None
            self._touch(corpus_hash)
            return entry.chunk_embeddings

    def set_embeddings(self, corpus_hash: str, embeddings) -> None:  # type: ignore[no-untyped-def]
        with self._lock:
            entry = self._cache.get(corpus_hash)
            if entry is None:
                return
            entry.chunk_embeddings = embeddings
            self._touch(corpus_hash)

    def get_rust_index(self, corpus_hash: str):  # type: ignore[no-untyped-def]
        with self._lock:
            entry = self._cache.get(corpus_hash)
            if entry is None:
                return None
            if time.monotonic() - entry.created_at >= self._ttl:
                return None
            self._touch(corpus_hash)
            return entry.rust_index

    def set_rust_index(self, corpus_hash: str, rust_index: object) -> None:
        with self._lock:
            entry = self._cache.get(corpus_hash)
            if entry is None:
                return
            entry.rust_index = rust_index
            self._touch(corpus_hash)

    def invalidate(self, workspace_id: str) -> None:
        with self._lock:
            if workspace_id in self._workspace_aliases:
                del self._workspace_aliases[workspace_id]
                _log.debug("BM25 cache alias invalidated for workspace %s", workspace_id)

    def invalidate_all(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._workspace_aliases.clear()

    def snapshot(self) -> dict[str, dict[str, object]]:
        with self._lock:
            if not self._cache and not self._workspace_aliases:
                return {}
            aliases = {
                workspace_id: corpus_hash
                for workspace_id, corpus_hash in sorted(self._workspace_aliases.items())
            }
            entries = {
                corpus_hash: {
                    "chunk_count": len(entry.index.chunks),
                    "access_position": self._access_order.index(corpus_hash)
                    if corpus_hash in self._access_order
                    else -1,
                    "has_embeddings": entry.chunk_embeddings is not None,
                    "has_rust_index": entry.rust_index is not None,
                }
                for corpus_hash, entry in sorted(self._cache.items())
            }
            return {"aliases": aliases, "entries": entries}

    def _touch(self, corpus_hash: str) -> None:
        try:
            self._access_order.remove(corpus_hash)
        except ValueError:
            pass
        self._access_order.append(corpus_hash)

    def _set(self, corpus_hash: str, entry: _CacheEntry) -> None:
        if corpus_hash in self._cache:
            try:
                self._access_order.remove(corpus_hash)
            except ValueError:
                pass
        elif len(self._cache) >= self._max_entries:
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            for workspace_id, alias in list(self._workspace_aliases.items()):
                if alias == lru_key:
                    del self._workspace_aliases[workspace_id]
            _log.debug("BM25 cache evicted corpus %s (LRU)", lru_key[:12])

        self._cache[corpus_hash] = entry
        self._access_order.append(corpus_hash)
