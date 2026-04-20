from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal

import numpy as np

from app.config import get_settings
from app.parsers.base import ParserDetector, build_default_parser_registry
from app.parsers.plaintext import PlainTextParser
from app.retrieval.reranker import HeuristicReranker
from app.retrieval.span_executor import SpanExecutor
from app.retrieval.sparse_projection import NullProjection, ProjectionConfig, load_projection
from app.schemas.documents import Chunk
from app.schemas.search import SearchDocument, SearchResult

ProfileName = Literal["smoke", "regression", "latency", "semantic"]
SemanticMode = Literal["deterministic", "real", "auto"]
MatchType = Literal["exact", "semantic_or_context"]

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(slots=True)
class ExpectedHit:
    source_id: str
    must_include_text: str | None = None
    match_type: MatchType = "exact"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExpectedHit:
        source_id = str(payload.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("ExpectedHit.source_id is required")
        match_type = str(payload.get("match_type") or "exact")
        if match_type not in {"exact", "semantic_or_context"}:
            raise ValueError(f"Unsupported match_type: {match_type}")
        text = payload.get("must_include_text")
        must_include_text = str(text).strip() if text is not None else None
        return cls(
            source_id=source_id,
            must_include_text=must_include_text or None,
            match_type=match_type,  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class EvalCase:
    id: str
    query: str
    top_k: int = 5
    profiles: tuple[ProfileName, ...] = ("regression",)
    documents: list[dict[str, Any]] = field(default_factory=list)
    expected: list[ExpectedHit] = field(default_factory=list)
    disallowed_sources: tuple[str, ...] = ()
    generated_lines: dict[str, Any] | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalCase:
        case_id = str(payload.get("id") or "").strip()
        query = str(payload.get("query") or "").strip()
        if not case_id:
            raise ValueError("EvalCase.id is required")
        if not query:
            raise ValueError(f"EvalCase.query is required for {case_id}")

        top_k = int(payload.get("top_k") or 5)
        if top_k <= 0:
            raise ValueError(f"EvalCase.top_k must be > 0 for {case_id}")

        profiles_raw = payload.get("profiles") or ["regression"]
        if not isinstance(profiles_raw, list) or not profiles_raw:
            raise ValueError(f"EvalCase.profiles must be a non-empty list for {case_id}")
        profiles: list[ProfileName] = []
        for item in profiles_raw:
            value = str(item).strip()
            if value not in {"smoke", "regression", "latency", "semantic"}:
                raise ValueError(f"Unsupported profile {value!r} in case {case_id}")
            profiles.append(value)  # type: ignore[arg-type]

        raw_documents = payload.get("documents") or []
        if not isinstance(raw_documents, list):
            raise ValueError(f"EvalCase.documents must be a list for {case_id}")

        expected_raw = payload.get("expected") or []
        if not isinstance(expected_raw, list):
            raise ValueError(f"EvalCase.expected must be a list for {case_id}")
        expected = [ExpectedHit.from_dict(item) for item in expected_raw]

        disallowed_raw = payload.get("disallowed_sources") or []
        if not isinstance(disallowed_raw, list):
            raise ValueError(f"EvalCase.disallowed_sources must be a list for {case_id}")
        disallowed_sources = tuple(
            str(item).strip() for item in disallowed_raw if str(item).strip()
        )

        generated_lines = payload.get("generated_lines")
        if generated_lines is not None and not isinstance(generated_lines, dict):
            raise ValueError(f"EvalCase.generated_lines must be an object for {case_id}")

        return cls(
            id=case_id,
            query=query,
            top_k=top_k,
            profiles=tuple(profiles),
            documents=[dict(item) for item in raw_documents],
            expected=expected,
            disallowed_sources=disallowed_sources,
            generated_lines=dict(generated_lines) if generated_lines is not None else None,
            notes=(str(payload.get("notes")) if payload.get("notes") else None),
        )


@dataclass(slots=True)
class EvalRunConfig:
    profile: ProfileName
    fixtures_dir: Path
    semantic_mode: SemanticMode
    top_k_override: int | None = None
    gate_thresholds_min: dict[str, float] = field(default_factory=dict)
    gate_thresholds_max: dict[str, float] = field(default_factory=dict)
    gate_metrics: bool = True


@dataclass(slots=True)
class EvalCaseScore:
    case_id: str
    source_recall: float
    span_recall: float
    primary_span_accuracy: float
    exact_highlight_rate: float
    semantic_context_success: float
    partial_miss_rate: float
    time_to_result_ms: float
    windows_scanned: int
    window_count: int
    candidate_count: int
    shortlisted_candidates: int
    execution: dict[str, Any]
    missing_sources: list[str] = field(default_factory=list)
    missing_spans: list[str] = field(default_factory=list)
    disallowed_hits: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalScorecard:
    profile: ProfileName
    semantic_mode: SemanticMode
    status: Literal["passed", "failed", "skipped"]
    skipped_reason: str | None
    metrics: dict[str, float]
    failed_gates: list[str]
    cases_evaluated: int
    duration_ms: float
    case_scores: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def exit_code(self) -> int:
        if self.status == "failed":
            return 2
        return 0


class DeterministicEvalProjection:
    """Eval-only semantic scorer used for deterministic CI runs.

    The scorer canonicalizes synonyms/cross-lingual aliases and computes stable
    hashed bag-of-token vectors. This keeps semantic behavior deterministic while
    avoiding dependency on a local projection checkpoint.
    """

    _ALIASES: dict[str, str] = {
        "billed": "charge",
        "charged": "charge",
        "charges": "charge",
        "charge": "charge",
        "duplicate": "duplicate",
        "twice": "duplicate",
        "double": "duplicate",
        "refund": "refund",
        "reembolso": "refund",
        "chargeback": "refund",
        "failed": "failure",
        "failure": "failure",
        "declined": "failure",
        "denied": "failure",
        "timeout": "timeout",
        "timed": "timeout",
        "503": "timeout",
        "invoice": "invoice",
        "factura": "invoice",
        "ticket": "ticket",
        "incidente": "ticket",
    }

    def __init__(self, dim: int = 128) -> None:
        self.config = ProjectionConfig(hash_features=dim, dim=dim)
        self._dim = dim

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed(query)

    def encode_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, self._dim), dtype=np.float32)
        return np.vstack([self._embed(chunk.text) for chunk in chunks])

    def score(self, query_vec: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
        if chunk_vecs.size == 0:
            return np.zeros(0, dtype=np.float32)
        raw = chunk_vecs @ query_vec
        return np.maximum(raw.astype(np.float32), 0.0)

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        tokens = self._normalize_tokens(text)
        for token in tokens:
            index = self._stable_index(token)
            vector[index] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def _normalize_tokens(self, text: str) -> set[str]:
        normalized: set[str] = set()
        for token in _TOKEN_RE.findall(text.lower()):
            canonical = self._ALIASES.get(token, token)
            normalized.add(canonical)
        return normalized

    def _stable_index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % self._dim


_PROFILE_DEFAULTS: dict[ProfileName, dict[str, Any]] = {
    "smoke": {
        "semantic_mode": "deterministic",
        "gate_metrics": True,
        "min": {
            "source_recall_at_k": 0.90,
            "span_recall_at_k": 0.80,
            "primary_span_accuracy": 0.65,
            "exact_highlight_rate": 0.70,
            "semantic_context_success_rate": 0.60,
        },
        "max": {
            "partial_miss_rate": 0.60,
            "time_to_result_ms_p95": 1800.0,
        },
    },
    "regression": {
        "semantic_mode": "deterministic",
        "gate_metrics": True,
        "min": {
            "source_recall_at_k": 0.85,
            "span_recall_at_k": 0.75,
            "primary_span_accuracy": 0.60,
            "exact_highlight_rate": 0.65,
            "semantic_context_success_rate": 0.55,
        },
        "max": {
            "partial_miss_rate": 0.70,
        },
    },
    "latency": {
        "semantic_mode": "deterministic",
        "gate_metrics": False,
        "min": {},
        "max": {},
    },
    "semantic": {
        "semantic_mode": "auto",
        "gate_metrics": True,
        "min": {
            "source_recall_at_k": 0.75,
            "span_recall_at_k": 0.60,
            "semantic_context_success_rate": 0.75,
        },
        "max": {},
    },
}


def build_run_config(
    profile: ProfileName,
    fixtures_dir: Path,
    *,
    semantic_mode: SemanticMode | None = None,
    top_k_override: int | None = None,
) -> EvalRunConfig:
    defaults = _PROFILE_DEFAULTS[profile]
    selected_mode = semantic_mode or defaults["semantic_mode"]
    return EvalRunConfig(
        profile=profile,
        fixtures_dir=fixtures_dir,
        semantic_mode=selected_mode,
        top_k_override=top_k_override,
        gate_thresholds_min=dict(defaults["min"]),
        gate_thresholds_max=dict(defaults["max"]),
        gate_metrics=bool(defaults["gate_metrics"]),
    )


def load_eval_cases(fixtures_dir: Path, profile: ProfileName) -> list[EvalCase]:
    if not fixtures_dir.exists():
        raise FileNotFoundError(f"Fixture directory not found: {fixtures_dir}")

    cases: list[EvalCase] = []
    for fixture_path in sorted(fixtures_dir.glob("*.json")):
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError(f"Fixture {fixture_path} must define a 'cases' list")
        for raw_case in raw_cases:
            case = EvalCase.from_dict(dict(raw_case))
            if profile in case.profiles:
                cases.append(case)

    if not cases:
        raise ValueError(f"No eval cases found for profile {profile!r} in {fixtures_dir}")
    return cases


def _build_executor(projection: Any) -> SpanExecutor:
    detector = ParserDetector(
        registry=build_default_parser_registry(),
        default_parser=PlainTextParser(),
    )
    return SpanExecutor(
        parser_detector=detector,
        reranker=HeuristicReranker(),
        projection=projection,
    )


def _resolve_projection(mode: SemanticMode) -> tuple[Any, str | None]:
    if mode == "deterministic":
        return DeterministicEvalProjection(), None

    settings = get_settings()
    config = ProjectionConfig(
        hash_features=settings.projection_hash_features,
        dim=settings.projection_dim,
    )
    projection = load_projection(settings.projection_model_path, config)
    if isinstance(projection, NullProjection):
        return projection, f"Semantic projection unavailable at {settings.projection_model_path}"
    return projection, None


def _document_from_payload(payload: dict[str, Any], index: int) -> SearchDocument:
    source_id = str(payload.get("source_id") or f"fixture-source-{index}")
    source_name = str(payload.get("source_name") or f"fixture-{index}.txt")
    source_origin = str(payload.get("source_origin") or "raw")
    source_type = str(payload.get("source_type") or "raw")
    media_type = str(payload.get("media_type") or "text/plain")
    content = payload.get("content")
    if content is None:
        raise ValueError(f"Document {source_id} has no content")

    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "source_id": source_id,
            "source_origin": source_origin,
            "source_type": source_type,
            "title": metadata.get("title", source_name),
        }
    )
    normalized_content = content if isinstance(content, (bytes, str)) else json.dumps(content)
    return SearchDocument(
        file_name=source_name,
        content=normalized_content,
        media_type=media_type,
        metadata=metadata,
    )


def _build_generated_lines_document(spec: dict[str, Any]) -> SearchDocument:
    line_count = int(spec.get("line_count") or 0)
    target_line = int(spec.get("target_line") or 0)
    target_text = str(spec.get("target_text") or "").strip()
    if line_count <= 0:
        raise ValueError("generated_lines.line_count must be > 0")
    if target_line and (target_line < 1 or target_line > line_count):
        raise ValueError("generated_lines.target_line must be inside [1, line_count]")
    if target_line and not target_text:
        raise ValueError("generated_lines.target_text is required when target_line is set")

    prefix = str(spec.get("noise_prefix") or "noise line")
    lines: list[str] = []
    for line_number in range(1, line_count + 1):
        if target_line and line_number == target_line:
            lines.append(target_text)
            continue
        lines.append(f"{prefix} {line_number:04d}")

    return _document_from_payload(
        {
            "source_id": str(spec.get("source_id") or "generated-lines"),
            "source_name": str(spec.get("source_name") or "generated-lines.log"),
            "source_origin": str(spec.get("source_origin") or "raw"),
            "source_type": str(spec.get("source_type") or "log"),
            "media_type": str(spec.get("media_type") or "text/plain"),
            "content": "\n".join(lines),
            "metadata": dict(spec.get("metadata") or {}),
        },
        index=0,
    )


def _build_documents(case: EvalCase) -> list[SearchDocument]:
    documents = [
        _document_from_payload(payload, index)
        for index, payload in enumerate(case.documents)
    ]
    if case.generated_lines is not None:
        documents.append(_build_generated_lines_document(case.generated_lines))
    if not documents:
        raise ValueError(f"Case {case.id} has no documents")
    return documents


def _source_id(result: SearchResult) -> str | None:
    value = result.metadata.get("source_id") if isinstance(result.metadata, dict) else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _span_matches(span: dict[str, Any], needle: str) -> bool:
    lowered = needle.lower()
    for span_field in ("text", "snippet"):
        value = span.get(span_field)
        if isinstance(value, str) and lowered in value.lower():
            return True
    return False


def _match_span_evidence(result: SearchResult, needle: str) -> tuple[bool, bool, bool]:
    primary = result.primary_span if isinstance(result.primary_span, dict) else None
    if primary is not None and _span_matches(primary, needle):
        return True, True, bool(primary.get("highlights"))

    for span in result.matched_spans:
        if isinstance(span, dict) and _span_matches(span, needle):
            return True, False, bool(span.get("highlights"))

    return False, False, False


def evaluate_case(
    case: EvalCase,
    executor: SpanExecutor,
    *,
    top_k_override: int | None = None,
) -> EvalCaseScore:
    documents = _build_documents(case)
    top_k = top_k_override or case.top_k

    started_at = perf_counter()
    outcome = executor.execute(
        query=case.query,
        documents=documents,
        top_k=top_k,
        cacheable=False,
    )
    elapsed_ms = round((perf_counter() - started_at) * 1000.0, 3)

    execution = dict(outcome.execution)
    execution.setdefault("elapsed_ms", elapsed_ms)
    execution.setdefault("bytes_loaded", sum(len(document.data) for document in documents))
    execution.setdefault(
        "skipped_windows",
        max(0, int(execution.get("window_count", 0)) - int(execution.get("scanned_windows", 0))),
    )
    execution.setdefault(
        "completion_reason",
        "partial_scan_limit" if bool(execution.get("partial")) else "complete",
    )
    execution.setdefault(
        "phase_counts",
        {
            "exact_results": len(outcome.exact_results),
            "proxy_results": len(outcome.proxy_results),
            "reranked_results": len(outcome.reranked_results),
            "final_results": len(outcome.final_results),
        },
    )

    top_results = outcome.final_results[:top_k]

    expected_total = len(case.expected)
    source_hits = 0
    span_hits = 0
    primary_hits = 0
    exact_expected_total = 0
    exact_highlight_hits = 0
    semantic_expected_total = 0
    semantic_context_hits = 0
    partial_expected_total = 0
    partial_expected_misses = 0
    missing_sources: list[str] = []
    missing_spans: list[str] = []

    for expected in case.expected:
        matching_results = [
            result for result in top_results if _source_id(result) == expected.source_id
        ]
        if not matching_results:
            missing_sources.append(expected.source_id)
            if bool(execution.get("partial")):
                partial_expected_total += 1
                partial_expected_misses += 1
            if expected.match_type == "exact":
                exact_expected_total += 1
            else:
                semantic_expected_total += 1
            continue

        source_hits += 1
        result = matching_results[0]

        span_match = True
        primary_match = bool(result.primary_span)
        has_highlights = False

        if expected.must_include_text:
            span_match, primary_match, has_highlights = _match_span_evidence(
                result,
                expected.must_include_text,
            )

        if span_match:
            span_hits += 1
        else:
            missing_spans.append(expected.source_id)

        if primary_match:
            primary_hits += 1

        if expected.match_type == "exact":
            exact_expected_total += 1
            if has_highlights:
                exact_highlight_hits += 1
        else:
            semantic_expected_total += 1
            if span_match:
                semantic_context_hits += 1
            elif isinstance(result.primary_span, dict) and bool(result.primary_span.get("snippet")):
                semantic_context_hits += 1

        if bool(execution.get("partial")):
            partial_expected_total += 1
            if not span_match:
                partial_expected_misses += 1

    disallowed_hits: list[str] = []
    if case.disallowed_sources:
        disallowed = set(case.disallowed_sources)
        for result in top_results:
            source_id = _source_id(result)
            if source_id and source_id in disallowed:
                disallowed_hits.append(source_id)

    source_recall = source_hits / expected_total if expected_total else 1.0
    span_recall = span_hits / expected_total if expected_total else 1.0
    primary_accuracy = primary_hits / expected_total if expected_total else 1.0
    exact_highlight_rate = (
        exact_highlight_hits / exact_expected_total if exact_expected_total else 1.0
    )
    semantic_context_success = (
        semantic_context_hits / semantic_expected_total if semantic_expected_total else 1.0
    )
    partial_miss_rate = (
        partial_expected_misses / partial_expected_total if partial_expected_total else 0.0
    )

    return EvalCaseScore(
        case_id=case.id,
        source_recall=source_recall,
        span_recall=span_recall,
        primary_span_accuracy=primary_accuracy,
        exact_highlight_rate=exact_highlight_rate,
        semantic_context_success=semantic_context_success,
        partial_miss_rate=partial_miss_rate,
        time_to_result_ms=elapsed_ms,
        windows_scanned=int(execution.get("scanned_windows", 0)),
        window_count=int(execution.get("window_count", 0)),
        candidate_count=int(execution.get("candidate_count", 0)),
        shortlisted_candidates=int(execution.get("shortlisted_candidates", 0)),
        execution=execution,
        missing_sources=missing_sources,
        missing_spans=missing_spans,
        disallowed_hits=disallowed_hits,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = (percentile / 100.0) * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[lower])
    weight = rank - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _aggregate_metrics(case_scores: list[EvalCaseScore]) -> dict[str, float]:
    if not case_scores:
        return {
            "source_recall_at_k": 0.0,
            "span_recall_at_k": 0.0,
            "primary_span_accuracy": 0.0,
            "exact_highlight_rate": 0.0,
            "semantic_context_success_rate": 0.0,
            "partial_miss_rate": 0.0,
            "time_to_result_ms": 0.0,
            "time_to_result_ms_p95": 0.0,
            "windows_scanned_over_window_count": 0.0,
            "candidate_count": 0.0,
            "shortlisted_candidates": 0.0,
        }

    source_recall = mean(score.source_recall for score in case_scores)
    span_recall = mean(score.span_recall for score in case_scores)
    primary_accuracy = mean(score.primary_span_accuracy for score in case_scores)
    exact_highlight = mean(score.exact_highlight_rate for score in case_scores)
    semantic_context = mean(score.semantic_context_success for score in case_scores)
    partial_miss = mean(score.partial_miss_rate for score in case_scores)
    times = sorted(score.time_to_result_ms for score in case_scores)

    windows_total = sum(score.window_count for score in case_scores)
    scanned_total = sum(score.windows_scanned for score in case_scores)
    scanned_ratio = (scanned_total / windows_total) if windows_total else 1.0

    return {
        "source_recall_at_k": round(float(source_recall), 4),
        "span_recall_at_k": round(float(span_recall), 4),
        "primary_span_accuracy": round(float(primary_accuracy), 4),
        "exact_highlight_rate": round(float(exact_highlight), 4),
        "semantic_context_success_rate": round(float(semantic_context), 4),
        "partial_miss_rate": round(float(partial_miss), 4),
        "time_to_result_ms": round(float(mean(times)), 3),
        "time_to_result_ms_p95": round(_percentile(times, 95.0), 3),
        "windows_scanned_over_window_count": round(float(scanned_ratio), 4),
        "candidate_count": round(float(mean(score.candidate_count for score in case_scores)), 3),
        "shortlisted_candidates": round(
            float(mean(score.shortlisted_candidates for score in case_scores)),
            3,
        ),
    }


def _evaluate_gates(config: EvalRunConfig, metrics: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric_name, threshold in config.gate_thresholds_min.items():
        value = metrics.get(metric_name)
        if value is None:
            failures.append(f"missing metric {metric_name}")
            continue
        if value < threshold:
            failures.append(f"{metric_name}={value:.4f} < min {threshold:.4f}")

    for metric_name, threshold in config.gate_thresholds_max.items():
        value = metrics.get(metric_name)
        if value is None:
            failures.append(f"missing metric {metric_name}")
            continue
        if value > threshold:
            failures.append(f"{metric_name}={value:.4f} > max {threshold:.4f}")

    return failures


def run_eval(config: EvalRunConfig) -> EvalScorecard:
    started_at = perf_counter()
    cases = load_eval_cases(config.fixtures_dir, config.profile)

    projection, projection_warning = _resolve_projection(config.semantic_mode)
    if config.profile == "semantic" and config.semantic_mode in {"real", "auto"}:
        if isinstance(projection, NullProjection):
            duration_ms = round((perf_counter() - started_at) * 1000.0, 3)
            return EvalScorecard(
                profile=config.profile,
                semantic_mode=config.semantic_mode,
                status="skipped",
                skipped_reason=projection_warning,
                metrics={},
                failed_gates=[],
                cases_evaluated=0,
                duration_ms=duration_ms,
                case_scores=[],
            )

    executor = _build_executor(projection)

    case_scores: list[EvalCaseScore] = []
    for case in cases:
        case_scores.append(
            evaluate_case(
                case,
                executor,
                top_k_override=config.top_k_override,
            )
        )

    metrics = _aggregate_metrics(case_scores)
    failed_gates = _evaluate_gates(config, metrics) if config.gate_metrics else []

    disallowed_hits = [
        source_id
        for case_score in case_scores
        for source_id in case_score.disallowed_hits
    ]
    if disallowed_hits:
        failed_gates.append(f"disallowed sources returned: {sorted(set(disallowed_hits))}")

    status: Literal["passed", "failed", "skipped"] = "passed"
    if failed_gates:
        status = "failed"

    duration_ms = round((perf_counter() - started_at) * 1000.0, 3)
    return EvalScorecard(
        profile=config.profile,
        semantic_mode=config.semantic_mode,
        status=status,
        skipped_reason=projection_warning if status == "skipped" else None,
        metrics=metrics,
        failed_gates=failed_gates,
        cases_evaluated=len(case_scores),
        duration_ms=duration_ms,
        case_scores=[asdict(item) for item in case_scores],
    )


def format_scorecard_table(scorecard: EvalScorecard) -> str:
    lines = [
        f"Profile: {scorecard.profile}",
        f"Semantic mode: {scorecard.semantic_mode}",
        f"Status: {scorecard.status}",
    ]
    if scorecard.skipped_reason:
        lines.append(f"Skipped reason: {scorecard.skipped_reason}")
    lines.append(f"Cases evaluated: {scorecard.cases_evaluated}")
    lines.append(f"Duration ms: {scorecard.duration_ms:.3f}")

    if scorecard.metrics:
        lines.append("Metrics:")
        for key in sorted(scorecard.metrics.keys()):
            lines.append(f"  - {key}: {scorecard.metrics[key]:.4f}")

    if scorecard.failed_gates:
        lines.append("Gate failures:")
        for failure in scorecard.failed_gates:
            lines.append(f"  - {failure}")

    return "\n".join(lines)


__all__ = [
    "DeterministicEvalProjection",
    "EvalCase",
    "EvalCaseScore",
    "EvalRunConfig",
    "EvalScorecard",
    "ExpectedHit",
    "build_run_config",
    "evaluate_case",
    "format_scorecard_table",
    "load_eval_cases",
    "run_eval",
]
