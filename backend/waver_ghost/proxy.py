from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def _terms(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


@dataclass(slots=True)
class GhostVerdict:
    maybe_hit: bool
    bloom_overlap: int
    cms_score: int


@dataclass(slots=True)
class GhostProxy:
    bloom: set[str] = field(default_factory=set)
    cms: dict[str, int] = field(default_factory=dict)

    def ingest(self, payload_id: str, chunks: list[str]) -> None:
        for chunk in chunks:
            for term in _terms(chunk):
                key = self._key(payload_id, term)
                self.bloom.add(key)
                self.cms[key] = self.cms.get(key, 0) + 1

    def query(self, payload_id: str, query: str) -> GhostVerdict:
        overlaps = 0
        cms_score = 0
        for term in _terms(query):
            key = self._key(payload_id, term)
            if key in self.bloom:
                overlaps += 1
            cms_score += self.cms.get(key, 0)
        return GhostVerdict(
            maybe_hit=overlaps > 0,
            bloom_overlap=overlaps,
            cms_score=cms_score,
        )

    def _key(self, payload_id: str, term: str) -> str:
        return hashlib.sha256(f"{payload_id}:{term}".encode("utf-8")).hexdigest()
