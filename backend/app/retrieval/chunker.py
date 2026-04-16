from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import replace

from app.parsers.base import ParsedDocument
from app.schemas.documents import Chunk

_TOKEN_RE = re.compile(r"\S+")


def tokenize_text(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def chunk_document(document: ParsedDocument, max_tokens: int = 500, overlap: int = 50) -> list[Chunk]:
    tokens = tokenize_text(document.content)
    if not tokens:
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= max_tokens:
        step = max(1, max_tokens // 2)
    else:
        step = max_tokens - overlap

    chunks: list[Chunk] = []
    for start in range(0, len(tokens), step):
        end = min(len(tokens), start + max_tokens)
        if start >= end:
            break
        chunk_tokens = tokens[start:end]
        chunk_id = f"{document.source_name}:{document.location.page_number or 0}:{document.location.row_number or 0}:{len(chunks)}"
        metadata = dict(document.metadata)
        metadata.update(
            {
                "chunk_index": len(chunks),
                "token_start": start,
                "token_end": end,
            }
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                content=" ".join(chunk_tokens),
                source_name=document.source_name,
                metadata=metadata,
                location=replace(document.location),
                token_count=len(chunk_tokens),
            )
        )
        if end == len(tokens):
            break
    return chunks


def chunk_documents(documents: Iterable[ParsedDocument], max_tokens: int = 500, overlap: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, max_tokens=max_tokens, overlap=overlap))
    return chunks


def chunk_documents_iter(
    documents: Iterable[ParsedDocument],
    max_tokens: int = 500,
    overlap: int = 50,
) -> Iterator[Chunk]:
    """Generator variant — yields chunks one at a time to avoid materialising
    the full list in memory. Useful for large corpora (10K+ documents)."""

    for document in documents:
        yield from chunk_document(document, max_tokens=max_tokens, overlap=overlap)
