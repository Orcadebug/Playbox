"""
Cross-encoder tokenizer for ONNX reranker inference.

Wraps the HuggingFace ``tokenizers`` library (Rust-backed, no PyTorch) to
encode (query, passage) pairs into batched numpy arrays ready for ONNX Runtime.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)


class CrossEncoderTokenizer:
    """
    Loads a ``tokenizer.json`` from a HuggingFace model directory and encodes
    query-passage pairs for cross-encoder inference.

    Input format: ``[CLS] query [SEP] passage [SEP]``
    Output arrays: ``input_ids``, ``attention_mask``, ``token_type_ids``
    """

    def __init__(self, tokenizer_path: Path, max_length: int = 512) -> None:
        from tokenizers import Tokenizer  # type: ignore[import-untyped]

        self._max_length = max_length
        self._tok = Tokenizer.from_file(str(tokenizer_path))
        # Padding fills to the longest sequence in each batch (not max_length),
        # which saves compute on short-document batches.
        self._tok.enable_padding()
        self._tok.enable_truncation(max_length=max_length)

    def encode_pairs(
        self,
        query: str,
        passages: list[str],
        max_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Encode a single query against multiple passages.

        Returns a dict of numpy arrays with keys matching BERT-style ONNX inputs:
        ``input_ids``, ``attention_mask``, ``token_type_ids``.
        All arrays have shape ``(len(passages), seq_len)``.
        """
        if not passages:
            empty = np.empty((0, 0), dtype=np.int64)
            return {"input_ids": empty, "attention_mask": empty, "token_type_ids": empty}

        if max_length is not None and max_length != self._max_length:
            self._tok.enable_truncation(max_length=max_length)

        encodings = self._tok.encode_batch([(query, p) for p in passages])

        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids for enc in encodings], dtype=np.int64)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
