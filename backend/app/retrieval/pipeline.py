from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from app.parsers.base import ParserDetector, ParsedDocument, ParsedFile, build_default_parser_registry
from app.parsers.plaintext import PlainTextParser
from app.retrieval.bm25 import BM25Index, BM25ScoredChunk
from app.retrieval.chunker import chunk_documents
from app.retrieval.reranker import AutoReranker, Reranker
from app.schemas.search import SearchDocument, SearchResult


def _snippet(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 1)].rstrip() + "…"


@dataclass(slots=True)
class RetrievalPipeline:
    parser_detector: ParserDetector = field(
        default_factory=lambda: ParserDetector(
            registry=build_default_parser_registry(),
            default_parser=PlainTextParser(),
        )
    )
    reranker: Reranker = field(default_factory=AutoReranker)
    chunk_size: int = 500
    chunk_overlap: int = 50
    bm25_candidates: int = 100

    def search(self, query: str, documents: Sequence[SearchDocument], top_k: int = 10) -> list[SearchResult]:
        parsed_documents = self._parse_documents(documents)
        chunks = chunk_documents(parsed_documents, max_tokens=self.chunk_size, overlap=self.chunk_overlap)
        if not chunks:
            return []

        index = BM25Index(chunks)
        bm25_hits = index.search(query, top_k=self.bm25_candidates)
        reranked = self.reranker.rerank(query, bm25_hits, top_k=top_k)
        return [self._to_search_result(hit) for hit in reranked[:top_k]]

    def _parse_documents(self, documents: Sequence[SearchDocument]) -> list[ParsedDocument]:
        parsed_documents: list[ParsedDocument] = []
        for document in documents:
            parsed_file = self.parser_detector.parse(
                file_name=document.file_name,
                content=document.data,
                media_type=document.media_type,
            )
            parsed_documents.extend(self._merge_metadata(parsed_file, document.metadata))
        return parsed_documents

    def _merge_metadata(self, parsed_file: ParsedFile, metadata: dict[str, object]) -> list[ParsedDocument]:
        merged: list[ParsedDocument] = []
        for parsed_document in parsed_file.documents:
            merged_metadata = dict(metadata)
            merged_metadata.update(parsed_document.metadata)
            merged_metadata.setdefault("parser_name", parsed_file.parser_name)
            if parsed_file.media_type is not None:
                merged_metadata.setdefault("media_type", parsed_file.media_type)
            merged.append(
                ParsedDocument(
                    content=parsed_document.content,
                    source_name=parsed_document.source_name,
                    metadata=merged_metadata,
                    location=parsed_document.location,
                )
            )
        return merged

    def _to_search_result(self, hit: BM25ScoredChunk) -> SearchResult:
        metadata = dict(hit.chunk.metadata)
        return SearchResult(
            content=hit.chunk.text,
            snippet=_snippet(hit.chunk.text),
            score=hit.rerank_score if hit.rerank_score is not None else hit.score,
            source_name=hit.chunk.source_name,
            metadata=metadata,
            chunk_id=hit.chunk.chunk_id,
            parser_name=metadata.get("parser_name") if isinstance(metadata.get("parser_name"), str) else None,
            bm25_score=hit.bm25_score if hit.bm25_score is not None else hit.score,
            rerank_score=hit.rerank_score if hit.rerank_score is not None else hit.score,
        )
