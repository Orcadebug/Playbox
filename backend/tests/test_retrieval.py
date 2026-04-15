from __future__ import annotations

from app.retrieval.bm25 import BM25Index
from app.retrieval.chunker import chunk_document
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import AutoReranker, HeuristicReranker
from app.schemas.documents import ParsedDocument, SourceLocation
from app.schemas.search import SearchDocument


def test_chunker_preserves_overlap_and_metadata() -> None:
    document = ParsedDocument(
        content="one two three four five six seven eight",
        source_name="notes.txt",
        metadata={"row_index": 4},
        location=SourceLocation(row_number=4),
    )

    chunks = chunk_document(document, max_tokens=4, overlap=2)
    assert len(chunks) == 3
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "three four five six"
    assert chunks[1].location.row_number == 4
    assert chunks[1].metadata["row_index"] == 4


def test_bm25_index_ranks_relevant_chunk_first() -> None:
    chunks = [
        chunk_document(
            ParsedDocument(content="billing issue and refund request", source_name="a.txt"),
            max_tokens=20,
        )[0],
        chunk_document(
            ParsedDocument(content="shipping delay and address change", source_name="b.txt"),
            max_tokens=20,
        )[0],
    ]

    index = BM25Index(chunks)
    hits = index.search("billing refund", top_k=2)
    assert hits[0].chunk.source_name == "a.txt"
    assert hits[0].score > hits[-1].score if len(hits) > 1 else True


def test_pipeline_search_returns_rich_results_with_fallback_reranker() -> None:
    pipeline = RetrievalPipeline(reranker=HeuristicReranker())
    results = pipeline.search(
        "billing complaint last week",
        [
            SearchDocument(
                file_name="complaints.csv",
                content=b"customer,issue,week\nAcme,billing complaint,last week\nBeta,shipping delay,this week\n",
                media_type="text/csv",
                metadata={"workspace": "demo"},
            ),
            SearchDocument(
                file_name="notes.txt",
                content=b"general product notes",
            ),
        ],
        top_k=3,
    )

    assert results
    assert results[0].source_name == "complaints.csv"
    assert "billing complaint" in results[0].content
    assert results[0].snippet is not None
    assert results[0].metadata["workspace"] == "demo"
    assert results[0].metadata["parser_name"] == "csv"


def test_auto_reranker_gracefully_falls_back_without_model() -> None:
    reranker = AutoReranker(model_path="/does/not/exist.onnx")
    assert reranker.is_fallback is True

