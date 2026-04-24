from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

TeacherName = Literal["deterministic", "mrl"]
TextKind = Literal["query", "document"]

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_DIRS = (
    BACKEND_ROOT / "app/retrieval/eval_fixtures",
    BACKEND_ROOT / "tests/fixtures/eval",
)

SYNTHETIC_TOPICS = (
    ("duplicate billing", "duplicate charge", "customer billed twice after checkout retry"),
    ("refund decision", "refund denied", "refund was rejected because invoice was voided"),
    ("checkout timeout", "payment timeout", "checkout request timed out after confirmation"),
    ("webhook signature", "invalid webhook", "missing signature validation blocks launch"),
    ("workspace approval", "team admin consent", "owner approval is required"),
    ("export columns", "report field omission", "export is missing requested columns"),
    ("region validation", "address form province", "province field fails validation"),
    ("password reset", "account recovery link", "password reset link expires immediately"),
    ("api rate limit", "429 quota exceeded", "request quota blocks connector sync"),
    ("launch blocker", "release risk", "missing production guard is a launch blocker"),
)

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_ALIASES: dict[str, str] = {
    "billed": "charge",
    "billing": "charge",
    "charged": "charge",
    "charges": "charge",
    "payment": "charge",
    "payments": "charge",
    "invoice": "invoice",
    "duplicate": "duplicate",
    "duplicated": "duplicate",
    "double": "duplicate",
    "twice": "duplicate",
    "refund": "refund",
    "refunded": "refund",
    "failure": "failure",
    "failed": "failure",
    "error": "failure",
    "timeout": "timeout",
    "timeouts": "timeout",
    "latency": "timeout",
    "slow": "timeout",
    "launch": "launch",
    "release": "launch",
    "blocker": "blocker",
    "blocked": "blocker",
    "risk": "risk",
}
_NEG_VERBS = frozenset({"not", "no", "never", "cannot", "cant", "wont"})
_DENY_VERBS = frozenset({"denied", "rejected", "failed", "blocked", "error", "missing"})
_POS_VERBS = frozenset({"approved", "succeeded", "resolved", "fixed", "completed"})


@dataclass(slots=True)
class ProjectionConfig:
    hash_features: int = 262144
    dim: int = 256
    ngram_range: tuple[int, int] = (1, 2)
    version: str = "1"


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        norm = float(np.linalg.norm(values))
        return values / norm if norm > 0 else values
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def save_projection(path: Path, W: sparse.csr_matrix, config: ProjectionConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_json = json.dumps(asdict(config))
    csr = W.tocsr().astype(np.float32)
    np.savez(
        path,
        sparse=np.array(True),
        data=csr.data,
        indices=csr.indices,
        indptr=csr.indptr,
        shape=np.array(csr.shape),
        config=np.array(config_json),
    )


def _stable_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % dim


def _augment_with_polarity(text: str) -> str:
    lower_tokens = [token for token in _TOKEN_RE.findall(text.lower())]
    if not lower_tokens:
        return text
    token_set = set(lower_tokens)
    prefixes: list[str] = []
    if token_set & _NEG_VERBS:
        prefixes.append("NEG_")
    if token_set & _DENY_VERBS:
        prefixes.append("DENIED_")
    if token_set & _POS_VERBS:
        prefixes.append("POS_")
    if not prefixes:
        return text
    polarity_words = _NEG_VERBS | _DENY_VERBS | _POS_VERBS
    content = [token for token in lower_tokens if token not in polarity_words and len(token) > 2]
    synthetic = [f"{prefix}{token}" for prefix in prefixes for token in content]
    return f"{text} {' '.join(synthetic)}" if synthetic else text


def _deterministic_embed(text: str, config: ProjectionConfig) -> np.ndarray:
    vector = np.zeros(config.dim, dtype=np.float32)
    tokens = [_ALIASES.get(token, token) for token in _TOKEN_RE.findall(text.lower())]
    if not tokens:
        return vector
    features: list[tuple[str, float]] = [(token, 1.0) for token in tokens]
    features.extend(
        (f"{left} {right}", 1.35) for left, right in zip(tokens, tokens[1:], strict=False)
    )
    seen: set[str] = set()
    for feature, weight in features:
        if feature in seen:
            continue
        seen.add(feature)
        vector[_stable_index(feature, config.dim)] += weight
    return _normalize_rows(vector)


@dataclass(frozen=True, slots=True)
class TrainingText:
    text: str
    kind: TextKind
    source: str


@dataclass(slots=True)
class DistillationReport:
    teacher: TeacherName
    examples_loaded: int
    examples_used: int
    hash_features: int
    dim: int
    nnz: int
    validation_mean_cosine: float
    validation_p10_cosine: float
    duration_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _fixture_dirs(values: Sequence[Path] | None) -> list[Path]:
    if values:
        return [path for path in values if path.exists()]
    return [path for path in DEFAULT_FIXTURE_DIRS if path.exists()]


def _iter_fixture_files(paths: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for base in paths:
        resolved = base.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file() and resolved.suffix == ".json":
            yield resolved
            continue
        if resolved.is_dir():
            yield from sorted(resolved.rglob("*.json"))


def _document_text(document: dict[str, object]) -> str | None:
    content = document.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    data = document.get("data")
    if isinstance(data, str) and data.strip():
        return data.strip()
    return None


def load_fixture_texts(paths: Sequence[Path]) -> list[TrainingText]:
    examples: list[TrainingText] = []
    for path in _iter_fixture_files(paths):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            query = case.get("query")
            if isinstance(query, str) and query.strip():
                examples.append(
                    TrainingText(
                        text=query.strip(),
                        kind="query",
                        source=f"{path.name}:{case.get('id', 'case')}:query",
                    )
                )
            documents = case.get("documents")
            if not isinstance(documents, list):
                continue
            for document in documents:
                if not isinstance(document, dict):
                    continue
                text = _document_text(document)
                if text:
                    source_id = str(document.get("source_id") or "document")
                    examples.append(
                        TrainingText(
                            text=text,
                            kind="document",
                            source=f"{path.name}:{case.get('id', 'case')}:{source_id}",
                        )
                    )
    return _dedupe_examples(examples)


def synthetic_texts(count: int) -> list[TrainingText]:
    if count <= 0:
        return []
    examples: list[TrainingText] = []
    templates = (
        "Ticket note: {a}. Customer describes it as {b}; support normalized it to {c}.",
        "API log: {b}. The payload indicates {c}.",
        "Support summary: {c}. Short label: {a}.",
        "Search query from agent: {b}",
    )
    index = 0
    while len(examples) < count:
        a, b, c = SYNTHETIC_TOPICS[index % len(SYNTHETIC_TOPICS)]
        template = templates[index % len(templates)]
        text = template.format(a=a, b=b, c=c)
        kind: TextKind = "query" if template.startswith("Search query") else "document"
        examples.append(TrainingText(text=text, kind=kind, source=f"synthetic:{index}"))
        index += 1
    return examples


def _dedupe_examples(examples: Sequence[TrainingText]) -> list[TrainingText]:
    seen: set[tuple[str, TextKind]] = set()
    deduped: list[TrainingText] = []
    for example in examples:
        key = (example.text.strip(), example.kind)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def _limit_examples(
    examples: Sequence[TrainingText],
    *,
    max_examples: int | None,
    seed: int,
) -> list[TrainingText]:
    if max_examples is None or max_examples <= 0 or len(examples) <= max_examples:
        return list(examples)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(examples), size=max_examples, replace=False)
    return [examples[int(index)] for index in sorted(indices)]


def _deterministic_teacher(
    examples: Sequence[TrainingText],
    config: ProjectionConfig,
) -> np.ndarray:
    return np.vstack([_deterministic_embed(example.text, config) for example in examples]).astype(
        np.float32
    )


def _mrl_teacher(
    examples: Sequence[TrainingText],
    *,
    model_path: Path,
    dim: int,
    batch_size: int,
    query_prefix: str,
    document_prefix: str,
) -> np.ndarray:
    try:
        import waver_core  # type: ignore[import-not-found]
    except ImportError as exc:
        return _mrl_teacher_onnxruntime(
            examples,
            model_path=model_path,
            dim=dim,
            batch_size=batch_size,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            import_error=exc,
        )

    mrl_encode = getattr(waver_core, "mrl_encode", None)
    if not callable(mrl_encode):
        raise RuntimeError("waver_core.mrl_encode is not callable")
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    rows: list[list[float]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        query_items = [(offset, item) for offset, item in enumerate(batch) if item.kind == "query"]
        document_items = [
            (offset, item) for offset, item in enumerate(batch) if item.kind == "document"
        ]
        batch_rows: list[list[float] | None] = [None] * len(batch)
        for is_query, items in ((True, query_items), (False, document_items)):
            if not items:
                continue
            encoded = mrl_encode(
                str(model_path),
                [
                    _with_prefix(item.text, query_prefix if is_query else document_prefix)
                    for _, item in items
                ],
                dim,
                is_query,
            )
            for (offset, _), row in zip(items, encoded, strict=True):
                batch_rows[offset] = row
        rows.extend(row for row in batch_rows if row is not None)
    return _normalize_rows(np.asarray(rows, dtype=np.float32))


def _mrl_teacher_onnxruntime(
    examples: Sequence[TrainingText],
    *,
    model_path: Path,
    dim: int,
    batch_size: int,
    query_prefix: str,
    document_prefix: str,
    import_error: ImportError,
) -> np.ndarray:
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "waver_core is unavailable and Python ONNX fallback dependencies are missing. "
            "Install onnxruntime and tokenizers, or build waver_core."
        ) from import_error or exc
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    tokenizer_path = model_path.parent / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=512)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}

    rows: list[np.ndarray] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        encoded = tokenizer.encode_batch(
            [
                _with_prefix(
                    item.text,
                    query_prefix if item.kind == "query" else document_prefix,
                )
                for item in batch
            ]
        )
        seq_len = max(len(item.ids) for item in encoded)
        pad_id = tokenizer.token_to_id("[PAD]") or 0
        input_ids = np.asarray(
            [item.ids + [pad_id] * (seq_len - len(item.ids)) for item in encoded],
            dtype=np.int64,
        )
        attention_mask = np.asarray(
            [item.attention_mask + [0] * (seq_len - len(item.attention_mask)) for item in encoded],
            dtype=np.int64,
        )
        inputs: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.zeros_like(input_ids)
        output = np.asarray(session.run(None, inputs)[0], dtype=np.float32)
        if output.ndim == 2:
            if output.shape[1] < dim:
                raise RuntimeError(
                    f"MRL output dimension {output.shape[1]} is smaller than requested {dim}"
                )
            rows.append(output[:, :dim])
        elif output.ndim == 3:
            if output.shape[2] < dim:
                raise RuntimeError(
                    f"MRL output dimension {output.shape[2]} is smaller than requested {dim}"
                )
            mask = attention_mask[:, :, None].astype(np.float32)
            pooled = (output[:, :, :dim] * mask).sum(axis=1) / np.clip(
                mask.sum(axis=1),
                1.0,
                None,
            )
            rows.append(pooled)
        else:
            raise RuntimeError(f"Unsupported MRL output shape: {output.shape}")
    return _normalize_rows(np.vstack(rows).astype(np.float32))


def embed_teacher(
    examples: Sequence[TrainingText],
    *,
    teacher: TeacherName,
    config: ProjectionConfig,
    mrl_model: Path | None,
    batch_size: int,
    query_prefix: str,
    document_prefix: str,
) -> np.ndarray:
    if teacher == "deterministic":
        return _deterministic_teacher(examples, config)
    if mrl_model is None:
        raise ValueError("--mrl-model is required when --teacher=mrl")
    return _mrl_teacher(
        examples,
        model_path=mrl_model,
        dim=config.dim,
        batch_size=batch_size,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )


def _with_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    return text if text.startswith(prefix) else f"{prefix}{text}"


def _vectorize_texts(
    examples: Sequence[TrainingText],
    config: ProjectionConfig,
) -> sparse.csr_matrix:
    vectorizer = HashingVectorizer(
        n_features=config.hash_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=config.ngram_range,
    )
    texts = [_augment_with_polarity(example.text) for example in examples]
    return vectorizer.transform(texts).astype(np.float32).tocsr()


def distill_projection(
    examples: Sequence[TrainingText],
    teacher_rows: np.ndarray,
    config: ProjectionConfig,
    *,
    ridge: float,
) -> sparse.csr_matrix:
    if len(examples) != len(teacher_rows):
        raise ValueError("example and teacher row counts do not match")
    X = _vectorize_texts(examples, config).tocoo()
    sums: defaultdict[int, np.ndarray] = defaultdict(
        lambda: np.zeros(config.dim, dtype=np.float32)
    )
    weights: defaultdict[int, float] = defaultdict(float)

    for row, feature, value in zip(X.row, X.col, X.data, strict=True):
        weight = float(value)
        sums[int(feature)] += weight * teacher_rows[int(row)]
        weights[int(feature)] += weight * weight

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    threshold = 1e-8
    for feature in sorted(sums):
        vector = sums[feature] / (weights[feature] + ridge)
        nonzero = np.flatnonzero(np.abs(vector) > threshold)
        rows.extend([feature] * len(nonzero))
        cols.extend(int(col) for col in nonzero)
        data.extend(float(vector[int(col)]) for col in nonzero)

    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(config.hash_features, config.dim),
        dtype=np.float32,
    )


def validate_projection(
    examples: Sequence[TrainingText],
    teacher_rows: np.ndarray,
    W: sparse.csr_matrix,
    config: ProjectionConfig,
) -> tuple[float, float]:
    if not examples:
        return 0.0, 0.0
    projected = _vectorize_texts(examples, config) @ W
    if sparse.issparse(projected):
        projected = projected.toarray()
    projected = _normalize_rows(np.asarray(projected, dtype=np.float32))
    teacher_rows = _normalize_rows(np.asarray(teacher_rows, dtype=np.float32))
    cosine = np.sum(projected * teacher_rows, axis=1)
    return float(np.mean(cosine)), float(np.percentile(cosine, 10))


def build_projection(
    *,
    fixture_dirs: Sequence[Path] | None,
    teacher: TeacherName,
    mrl_model: Path | None,
    out: Path,
    report_out: Path | None,
    hash_features: int,
    dim: int,
    ridge: float,
    synthetic_count: int,
    max_examples: int | None,
    validation_fraction: float,
    seed: int,
    batch_size: int,
    query_prefix: str,
    document_prefix: str,
) -> DistillationReport:
    started = perf_counter()
    config = ProjectionConfig(hash_features=hash_features, dim=dim, version="distilled-v1")
    examples = _dedupe_examples(
        [
            *load_fixture_texts(_fixture_dirs(fixture_dirs)),
            *synthetic_texts(synthetic_count),
        ]
    )
    if not examples:
        raise RuntimeError("No training texts found. Provide --fixtures-dir or --synthetic-count.")
    loaded_count = len(examples)
    examples = _limit_examples(examples, max_examples=max_examples, seed=seed)

    teacher_rows = embed_teacher(
        examples,
        teacher=teacher,
        config=config,
        mrl_model=mrl_model,
        batch_size=batch_size,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(examples))
    validation_size = max(1, int(round(len(examples) * validation_fraction)))
    if len(examples) > 1:
        validation_size = min(validation_size, len(examples) - 1)
    else:
        validation_size = 0
    validation_indices = set(int(index) for index in order[:validation_size])
    train_examples = [
        example for index, example in enumerate(examples) if index not in validation_indices
    ]
    train_rows = np.vstack(
        [row for index, row in enumerate(teacher_rows) if index not in validation_indices]
    )
    validation_examples = [
        example for index, example in enumerate(examples) if index in validation_indices
    ]
    if validation_examples:
        validation_rows = np.vstack(
            [row for index, row in enumerate(teacher_rows) if index in validation_indices]
        )
    else:
        validation_examples = train_examples
        validation_rows = train_rows

    W = distill_projection(train_examples, train_rows, config, ridge=ridge)
    mean_cosine, p10_cosine = validate_projection(validation_examples, validation_rows, W, config)
    save_projection(out, W, config)

    report = DistillationReport(
        teacher=teacher,
        examples_loaded=loaded_count,
        examples_used=len(examples),
        hash_features=hash_features,
        dim=dim,
        nnz=int(W.nnz),
        validation_mean_cosine=round(mean_cosine, 6),
        validation_p10_cosine=round(p10_cosine, 6),
        duration_ms=round((perf_counter() - started) * 1000.0, 3),
    )
    if report_out is None:
        report_out = out.with_suffix(".report.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Waver's sparse projection by distilling semantic teacher embeddings."
    )
    parser.add_argument("--out", type=Path, default=Path("models/projection.npz"))
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--fixtures-dir", type=Path, action="append", default=None)
    parser.add_argument("--teacher", choices=["deterministic", "mrl"], default="mrl")
    parser.add_argument("--mrl-model", type=Path, default=Path("models/mrl/model.onnx"))
    parser.add_argument("--hash-features", type=int, default=262144)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--synthetic-count", type=int, default=240)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ",
    )
    parser.add_argument("--document-prefix", default="")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_projection(
        fixture_dirs=args.fixtures_dir,
        teacher=args.teacher,
        mrl_model=args.mrl_model,
        out=args.out,
        report_out=args.report_out,
        hash_features=args.hash_features,
        dim=args.dim,
        ridge=args.ridge,
        synthetic_count=args.synthetic_count,
        max_examples=args.max_examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        query_prefix=args.query_prefix,
        document_prefix=args.document_prefix,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
