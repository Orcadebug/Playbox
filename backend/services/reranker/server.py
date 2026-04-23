from __future__ import annotations

import logging

_log = logging.getLogger(__name__)


def serve() -> None:
    """Placeholder gRPC entrypoint for the remote reranker service.

    The API contract is defined in `reranker.proto`. Local app code treats this
    service as optional and falls back to in-process ONNX or heuristic rerank.
    """

    _log.info("Remote reranker service skeleton present; generate gRPC stubs before serving")
