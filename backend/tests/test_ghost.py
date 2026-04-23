from __future__ import annotations

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
