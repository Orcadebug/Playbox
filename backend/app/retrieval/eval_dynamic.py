from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models import Base, Document, Source
from app.retrieval.bm25_cache import BM25IndexCache
from app.retrieval.eval_harness import EvalScorecard
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranker import HeuristicReranker
from app.services.search import SearchService


@dataclass(slots=True)
class EvalDynamicConfig:
    fixture_path: Path
    top_k: int = 5
    workspace_id: str = "dynamic-eval"
    bm25_cache: BM25IndexCache | None = None


def _source_id(document: dict[str, Any]) -> str:
    value = document.get("id") or document.get("source_id")
    if value is None:
        raise ValueError(f"Dynamic document is missing id/source_id: {document}")
    return str(value)


def _content(document: dict[str, Any]) -> str:
    content = document.get("content")
    if not isinstance(content, str):
        return json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return content


def _find_document(documents: list[dict[str, Any]], source_id: str) -> dict[str, Any] | None:
    for document in documents:
        if _source_id(document) == source_id:
            return document
    return None


def _apply_step(documents: list[dict[str, Any]], step: dict[str, Any]) -> list[dict[str, Any]]:
    op = str(step.get("op") or "")
    if op == "add":
        document = step.get("document")
        if not isinstance(document, dict):
            raise ValueError("dynamic add step requires document object")
        return [*documents, copy.deepcopy(document)]
    if op == "delete":
        source_id = str(step.get("source_id") or "")
        return [document for document in documents if _source_id(document) != source_id]
    if op == "mutate":
        source_id = str(step.get("source_id") or "")
        patch = step.get("patch")
        if not isinstance(patch, str):
            raise ValueError("dynamic mutate step requires string patch")
        mutated = copy.deepcopy(documents)
        target = _find_document(mutated, source_id)
        if target is None:
            raise ValueError(f"cannot mutate missing source_id {source_id}")
        target["content"] = patch
        return mutated
    if op == "reorder":
        order = [str(item) for item in step.get("order") or []]
        by_id = {_source_id(document): document for document in documents}
        missing = [source_id for source_id in order if source_id not in by_id]
        if missing:
            raise ValueError(f"reorder references missing source ids: {missing}")
        ordered = [by_id[source_id] for source_id in order]
        ordered_ids = set(order)
        ordered.extend(
            document for document in documents if _source_id(document) not in ordered_ids
        )
        return copy.deepcopy(ordered)
    raise ValueError(f"Unsupported dynamic mutation op: {op}")


def _span_contains_text(span: dict[str, Any], needle: str) -> bool:
    lowered = needle.lower()
    return any(
        isinstance(span.get(field), str) and lowered in str(span[field]).lower()
        for field in ("text", "snippet")
    )


def _find_matching_span(result: dict[str, Any], needle: str) -> dict[str, Any] | None:
    primary = result.get("primary_span")
    if isinstance(primary, dict) and _span_contains_text(primary, needle):
        return primary
    for span in result.get("matched_spans") or []:
        if isinstance(span, dict) and _span_contains_text(span, needle):
            return span
    return None


async def _row_count(session: AsyncSession, model: type[Base]) -> int:
    return int(await session.scalar(select(func.count()).select_from(model)) or 0)


async def _run_queries(
    *,
    service: SearchService,
    documents: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    config: EvalDynamicConfig,
    previous_recall: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    case_scores: list[dict[str, Any]] = []
    failures: list[str] = []
    by_id = {_source_id(document): document for document in documents}

    for query_spec in queries:
        query = str(query_spec.get("query") or "")
        expected = query_spec.get("expected") or []
        disallowed = {str(item) for item in query_spec.get("disallowed_sources") or []}
        response = await service.search(
            query=query,
            top_k=int(query_spec.get("top_k") or config.top_k),
            source_ids=None,
            workspace_id=config.workspace_id,
            raw_sources=copy.deepcopy(documents),
            include_stored_sources=False,
        )
        results = response["results"]
        result_ids = {str(result.get("source_id")) for result in results if result.get("source_id")}
        expected_ids = {str(item.get("source_id")) for item in expected}
        found_ids: set[str] = set()

        for item in expected:
            source_id = str(item.get("source_id") or "")
            needle = str(item.get("must_include_text") or "")
            matching = [
                result for result in results if str(result.get("source_id")) == source_id
            ]
            if not matching:
                failures.append(f"{query}: missing expected source {source_id}")
                continue
            if not needle:
                found_ids.add(source_id)
                continue
            span = _find_matching_span(matching[0], needle)
            if span is None:
                failures.append(f"{query}: missing expected span {needle!r} in {source_id}")
                continue
            document = by_id.get(source_id)
            if document is not None:
                content = _content(document)
                start = int(span.get("source_start", -1))
                end = int(span.get("source_end", -1))
                if content[start:end] != str(span.get("text")):
                    failures.append(f"{query}: incorrect source offsets for {source_id}")
                    continue
            found_ids.add(source_id)

        bad_hits = sorted(disallowed & result_ids)
        if bad_hits:
            failures.append(f"{query}: disallowed sources returned {bad_hits}")

        if query_spec.get("assert_reorder_stable"):
            previous = previous_recall.get(query)
            if previous is not None and previous != found_ids:
                failures.append(f"{query}: reorder changed recalled sources")

        previous_recall[query] = found_ids
        case_scores.append(
            {
                "case_id": str(query_spec.get("id") or query),
                "query": query,
                "expected_sources": sorted(expected_ids),
                "found_sources": sorted(found_ids),
                "disallowed_hits": bad_hits,
            }
        )

    return case_scores, failures


async def _run_dynamic_with_session(
    config: EvalDynamicConfig,
    session: AsyncSession,
) -> EvalScorecard:
    started_at = perf_counter()
    payload = json.loads(config.fixture_path.read_text(encoding="utf-8"))
    documents = copy.deepcopy(payload.get("baseline_documents") or [])
    if not isinstance(documents, list):
        raise ValueError("dynamic fixture baseline_documents must be a list")
    steps = payload.get("steps") or []
    queries = payload.get("queries") or []
    if not isinstance(steps, list) or not isinstance(queries, list):
        raise ValueError("dynamic fixture steps and queries must be lists")

    cache = config.bm25_cache or BM25IndexCache(ttl=60.0, max_entries=5)
    cache_before = cache.snapshot()
    source_rows_before = await _row_count(session, Source)
    document_rows_before = await _row_count(session, Document)

    service = SearchService(session)
    service.pipeline = RetrievalPipeline(reranker=HeuristicReranker(), bm25_cache=cache)

    case_scores: list[dict[str, Any]] = []
    failures: list[str] = []
    previous_recall: dict[str, set[str]] = {}

    for step_index in range(0, len(steps) + 1):
        current_queries = [
            query for query in queries if int(query.get("after_step", 0)) == step_index
        ]
        scores, query_failures = await _run_queries(
            service=service,
            documents=documents,
            queries=current_queries,
            config=config,
            previous_recall=previous_recall,
        )
        case_scores.extend(scores)
        failures.extend(query_failures)
        if step_index < len(steps):
            documents = _apply_step(documents, steps[step_index])

    if cache.snapshot() != cache_before:
        failures.append("BM25 cache changed during dynamic raw-source eval")
    if await _row_count(session, Source) != source_rows_before:
        failures.append("dynamic eval persisted Source rows")
    if await _row_count(session, Document) != document_rows_before:
        failures.append("dynamic eval persisted Document rows")

    status: Literal["passed", "failed", "skipped"] = "failed" if failures else "passed"
    return EvalScorecard(
        profile="smoke",
        semantic_mode="auto",
        status=status,
        skipped_reason=None,
        metrics={
            "queries_evaluated": float(len(case_scores)),
            "dynamic_failure_count": float(len(failures)),
            "cache_entries": float(len(cache.snapshot())),
        },
        failed_gates=failures,
        cases_evaluated=len(case_scores),
        duration_ms=round((perf_counter() - started_at) * 1000.0, 3),
        case_scores=case_scores,
    )


async def run_dynamic_eval(
    config: EvalDynamicConfig,
    session: AsyncSession | None = None,
) -> EvalScorecard:
    if session is not None:
        return await _run_dynamic_with_session(config, session)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    try:
        async with session_factory() as db_session:
            return await _run_dynamic_with_session(config, db_session)
    finally:
        await engine.dispose()


__all__ = ["EvalDynamicConfig", "run_dynamic_eval"]
