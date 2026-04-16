"""
Workspace-scoped BM25 index cache.

Caches built ``BM25Index`` objects per workspace to avoid rebuilding the index
on every search query. Entries expire after a configurable TTL and are
invalidated explicitly when sources are added or removed.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.bm25 import BM25Index
    from app.schemas.documents import Chunk

_log = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    index: BM25Index
    content_hash: str
    created_at: float = field(default_factory=time.monotonic)


def _chunks_hash(chunks: list[Chunk]) -> str:
    """Stable fingerprint for a list of chunks based on their IDs."""
    digest = hashlib.sha256("|".join(c.chunk_id for c in chunks).encode()).hexdigest()
    return digest


class BM25IndexCache:
    """
    LRU-limited, TTL-expiring BM25 index cache keyed by workspace_id.

    Thread-safe via a single lock. Suitable for FastAPI's async + thread-pool
    hybrid execution model since ``BM25Index.build`` is synchronous/CPU-bound.
    """

    def __init__(self, ttl: float = 300.0, max_entries: int = 10) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._cache: dict[str, _CacheEntry] = {}
        self._access_order: list[str] = []  # LRU tracking (front = oldest)
        self._lock = threading.Lock()

    def get_or_build(self, workspace_id: str, chunks: list[Chunk]) -> BM25Index:
        """
        Return a cached index if the content hash matches and TTL hasn't expired.
        Otherwise build a fresh index, cache it, and return it.
        """
        from app.retrieval.bm25 import BM25Index

        content_hash = _chunks_hash(chunks)

        with self._lock:
            entry = self._cache.get(workspace_id)
            now = time.monotonic()

            if entry is not None:
                age = now - entry.created_at
                if entry.content_hash == content_hash and age < self._ttl:
                    self._touch(workspace_id)
                    _log.debug("BM25 cache hit for workspace %s (age=%.1fs)", workspace_id, age)
                    return entry.index

            # Build new index outside the lock to avoid blocking other threads
            # (pass chunks by value — they're already materialised at call site)
            pass  # fall through

        # Build outside lock
        _log.debug("Building BM25 index for workspace %s (%d chunks)", workspace_id, len(chunks))
        index = BM25Index(chunks)

        with self._lock:
            self._set(workspace_id, _CacheEntry(index=index, content_hash=content_hash))

        return index

    def invalidate(self, workspace_id: str) -> None:
        with self._lock:
            if workspace_id in self._cache:
                del self._cache[workspace_id]
                try:
                    self._access_order.remove(workspace_id)
                except ValueError:
                    pass
                _log.debug("BM25 cache invalidated for workspace %s", workspace_id)

    def invalidate_all(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    # --- internals (must be called with lock held) ---

    def _touch(self, workspace_id: str) -> None:
        try:
            self._access_order.remove(workspace_id)
        except ValueError:
            pass
        self._access_order.append(workspace_id)

    def _set(self, workspace_id: str, entry: _CacheEntry) -> None:
        if workspace_id in self._cache:
            try:
                self._access_order.remove(workspace_id)
            except ValueError:
                pass
        elif len(self._cache) >= self._max_entries:
            # Evict least-recently-used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            _log.debug("BM25 cache evicted workspace %s (LRU)", lru_key)

        self._cache[workspace_id] = entry
        self._access_order.append(workspace_id)
