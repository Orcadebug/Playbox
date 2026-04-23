from __future__ import annotations

import random

from waver_ghost import GhostProxy


def test_ghost_proxy_detects_possible_hit() -> None:
    proxy = GhostProxy()
    proxy.ingest("payload-1", ["customer billing refund issue", "shipping update"])

    verdict = proxy.query("payload-1", "billing refund")

    assert verdict.maybe_hit is True
    assert verdict.bloom_overlap >= 1
    assert verdict.cms_score >= 1


def test_ghost_proxy_short_circuits_obvious_miss() -> None:
    proxy = GhostProxy()
    proxy.ingest("payload-1", ["customer billing refund issue"])

    verdict = proxy.query("payload-1", "warehouse logistics")

    assert verdict.maybe_hit is False
    assert verdict.bloom_overlap == 0


def test_ghost_proxy_memory_is_bounded() -> None:
    proxy = GhostProxy()
    terms = [f"word{i}" for i in range(100_000)]
    proxy.ingest("payload-1", terms)

    bloom_bytes = len(proxy.bloom)
    cms_bytes = proxy.cms.buffer_info()[1] * proxy.cms.itemsize
    assert bloom_bytes == proxy.bloom_bits // 8
    assert cms_bytes == proxy.cms_width * proxy.cms_depth * proxy.cms.itemsize
    assert bloom_bytes + cms_bytes < 4 * 1024 * 1024


def test_ghost_proxy_false_positive_rate_within_bound() -> None:
    proxy = GhostProxy(saturation_threshold=10**9)
    rng = random.Random(0xC0FFEE)
    ingest_terms = [f"i_{rng.getrandbits(64):x}" for _ in range(50_000)]
    proxy.ingest("payload-1", ingest_terms)

    ingested = set(ingest_terms)
    probes = []
    while len(probes) < 10_000:
        t = f"q_{rng.getrandbits(64):x}"
        if t not in ingested:
            probes.append(t)

    hits = 0
    for probe in probes:
        if proxy.query("payload-1", probe).maybe_hit:
            hits += 1

    fpr = hits / len(probes)
    m = proxy.bloom_bits
    k = proxy.bloom_hashes
    n = len(ingest_terms)
    theoretical = (1 - (1 - 1 / m) ** (k * n)) ** k
    assert fpr < max(0.02, 2 * theoretical)


def test_ghost_proxy_cms_saturates_without_overflow() -> None:
    proxy = GhostProxy(saturation_threshold=10**9)
    proxy.ingest("payload-1", ["hot"])

    for row in range(proxy.cms_depth):
        for col in range(proxy.cms_width):
            proxy.cms[row * proxy.cms_width + col] = 0xFFFFFFFF

    proxy.ingest("payload-1", ["hot"] * 5)

    verdict = proxy.query("payload-1", "hot")
    assert verdict.maybe_hit is True
    assert verdict.cms_score == 0xFFFFFFFF


def test_ghost_proxy_auto_rotates_on_saturation() -> None:
    proxy = GhostProxy(saturation_threshold=5_000)

    rng = random.Random(42)
    for payload_idx in range(5):
        batch = [f"t_{payload_idx}_{rng.getrandbits(32):x}" for _ in range(20_000)]
        proxy.ingest(f"p{payload_idx}", batch)

    assert proxy.rotations >= 1

    proxy.ingest("fresh", ["alpha beta gamma"])
    verdict = proxy.query("fresh", "alpha")
    assert verdict.maybe_hit is True


def test_ghost_proxy_reset_clears_state() -> None:
    proxy = GhostProxy()
    proxy.ingest("payload-1", ["alpha beta gamma"])
    assert proxy.bits_set > 0

    proxy.reset()

    assert proxy.bits_set == 0
    assert all(b == 0 for b in proxy.bloom)
    assert proxy.query("payload-1", "alpha").maybe_hit is False
