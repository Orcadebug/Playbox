from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import Any

_log = logging.getLogger(__name__)
_MODULE_NAME = "waver_core"


@lru_cache(maxsize=1)
def _load_module() -> Any | None:
    try:
        return importlib.import_module(_MODULE_NAME)
    except Exception as exc:
        _log.warning("Rust core unavailable (%s) — falling back to Python", exc)
        return None


def get_attr(name: str) -> Any | None:
    module = _load_module()
    if module is None:
        return None
    value = getattr(module, name, None)
    if value is None:
        _log.warning("Rust core missing %s.%s — falling back to Python", _MODULE_NAME, name)
    return value
