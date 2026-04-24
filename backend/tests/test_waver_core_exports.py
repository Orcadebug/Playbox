from __future__ import annotations

import pytest

waver_core = pytest.importorskip("waver_core")
pytestmark = pytest.mark.skipif(
    not all(
        hasattr(waver_core, name)
        for name in ("extract_direct_embeddings", "visit_trigrams_avx512")
    ),
    reason="installed waver_core extension needs rebuild for diagnostic exports",
)


def test_waver_core_exports_pressure_valves() -> None:
    for name in (
        "mrl_encode",
        "prefilter_windows",
        "extract_direct_embeddings",
        "visit_trigrams_avx512",
    ):
        assert callable(getattr(waver_core, name))


def test_extract_direct_embeddings_truncates_rows() -> None:
    assert waver_core.extract_direct_embeddings([[1.0, 2.0, 3.0]], 2) == [[1.0, 2.0]]


def test_extract_direct_embeddings_rejects_too_wide_dim() -> None:
    with pytest.raises(RuntimeError):
        waver_core.extract_direct_embeddings([[1.0]], 2)


def test_visit_trigrams_avx512_returns_or_errors_cleanly() -> None:
    try:
        grams = waver_core.visit_trigrams_avx512("billing")
    except RuntimeError as exc:
        assert "AVX-512" in str(exc)
    else:
        assert grams
        assert all(isinstance(gram, int) for gram in grams)
