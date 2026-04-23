from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from hashlib import blake2b

_MAX_U32 = 0xFFFFFFFF


def _terms(text: str) -> set[str]:
    return {token for token in text.lower().split() if token}


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass(slots=True)
class GhostVerdict:
    maybe_hit: bool
    bloom_overlap: int
    cms_score: int


@dataclass(slots=True)
class GhostProxy:
    bloom_bits: int = 1 << 20
    bloom_hashes: int = 7
    cms_width: int = 1 << 17
    cms_depth: int = 4
    saturation_threshold: int = 150_000

    bloom: bytearray = field(init=False)
    cms: array = field(init=False)
    bits_set: int = field(init=False, default=0)
    rotations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not _is_pow2(self.bloom_bits):
            raise ValueError("bloom_bits must be a power of two")
        if not _is_pow2(self.cms_width):
            raise ValueError("cms_width must be a power of two")
        if self.bloom_hashes <= 0 or self.cms_depth <= 0:
            raise ValueError("hash counts must be positive")
        self.bloom = bytearray(self.bloom_bits // 8)
        self.cms = array("I", [0] * (self.cms_width * self.cms_depth))

    def ingest(self, payload_id: str, chunks: list[str]) -> None:
        bloom_mask = self.bloom_bits - 1
        cms_mask = self.cms_width - 1
        bloom_k = self.bloom_hashes
        cms_depth = self.cms_depth
        bloom = self.bloom
        cms = self.cms

        for chunk in chunks:
            for term in _terms(chunk):
                h1, h2 = self._hash_pair(payload_id, term)

                for i in range(bloom_k):
                    bit = (h1 + i * h2) & bloom_mask
                    byte_idx = bit >> 3
                    bit_mask = 1 << (bit & 7)
                    b = bloom[byte_idx]
                    if not (b & bit_mask):
                        bloom[byte_idx] = b | bit_mask
                        self.bits_set += 1

                for row in range(cms_depth):
                    col = (h1 + (row + 1) * h2) & cms_mask
                    idx = row * self.cms_width + col
                    if cms[idx] < _MAX_U32:
                        cms[idx] += 1

                if self.bits_set and self.approx_items() > self.saturation_threshold:
                    self.reset()

    def query(self, payload_id: str, query: str) -> GhostVerdict:
        bloom_mask = self.bloom_bits - 1
        cms_mask = self.cms_width - 1
        bloom_k = self.bloom_hashes
        cms_depth = self.cms_depth
        bloom = self.bloom
        cms = self.cms

        overlaps = 0
        cms_total = 0
        for term in _terms(query):
            h1, h2 = self._hash_pair(payload_id, term)

            present = True
            for i in range(bloom_k):
                bit = (h1 + i * h2) & bloom_mask
                if not (bloom[bit >> 3] & (1 << (bit & 7))):
                    present = False
                    break
            if present:
                overlaps += 1

            row_min = _MAX_U32
            for row in range(cms_depth):
                col = (h1 + (row + 1) * h2) & cms_mask
                val = cms[row * self.cms_width + col]
                if val < row_min:
                    row_min = val
            cms_total += row_min

        return GhostVerdict(
            maybe_hit=overlaps > 0,
            bloom_overlap=overlaps,
            cms_score=cms_total,
        )

    def approx_items(self) -> int:
        m = self.bloom_bits
        x = self.bits_set
        if x == 0:
            return 0
        if x >= m:
            return self.saturation_threshold + 1
        return int(-(m / self.bloom_hashes) * math.log(1.0 - x / m))

    def reset(self) -> None:
        for i in range(len(self.bloom)):
            self.bloom[i] = 0
        cms = self.cms
        for i in range(len(cms)):
            cms[i] = 0
        self.bits_set = 0
        self.rotations += 1

    @staticmethod
    def _hash_pair(payload_id: str, term: str) -> tuple[int, int]:
        digest = blake2b(f"{payload_id}:{term}".encode(), digest_size=16).digest()
        return (
            int.from_bytes(digest[:8], "little"),
            int.from_bytes(digest[8:], "little") | 1,
        )
