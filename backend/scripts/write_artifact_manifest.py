from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_path() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


_bootstrap_path()

from app.config import get_settings  # noqa: E402


def sha256_prefix(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {"payload": payload}


def _projection_metadata(path: Path) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if not path.exists():
        return metadata
    try:
        with np.load(path, allow_pickle=False) as payload:
            raw = payload["config"]
            if isinstance(raw, np.ndarray):
                raw = raw.item()
            config = json.loads(str(raw))
            metadata.update(
                {
                    "hash_features": int(config.get("hash_features", 0)),
                    "dim": int(config.get("dim", 0)),
                    "version": str(config.get("version", "")),
                    "sparse": bool(payload["sparse"].item()) if "sparse" in payload else False,
                }
            )
    except Exception as exc:
        metadata["inspect_error"] = str(exc)
    return metadata


def _artifact_entry(path: Path, *, extra: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": str(path),
        "present": path.exists(),
        "sha256_prefix": sha256_prefix(path),
    }
    if extra:
        payload.update(extra)
    return payload


def build_manifest(
    *,
    model_dir: Path,
    projection_path: Path,
    reranker_model: str,
    reranker_quantized: bool,
    mrl_metadata: Path | None,
    projection_report: Path | None,
    latency_report: Path | None,
    eval_report: Path | None,
) -> dict[str, object]:
    model_name = reranker_model.replace("/", "-")
    reranker_dir = model_dir / model_name
    reranker_file = "model_quantized.onnx" if reranker_quantized else "model.onnx"
    mrl_dir = model_dir / "mrl"
    artifacts = {
        "projection": _artifact_entry(
            projection_path,
            extra=_projection_metadata(projection_path),
        ),
        "mrl": _artifact_entry(mrl_dir / "model.onnx", extra=_read_json(mrl_metadata) or {}),
        "mrl_tokenizer": _artifact_entry(mrl_dir / "tokenizer.json"),
        "reranker": _artifact_entry(
            reranker_dir / reranker_file,
            extra={"model_id": reranker_model, "quantized": reranker_quantized},
        ),
        "reranker_tokenizer": _artifact_entry(reranker_dir / "tokenizer.json"),
    }
    missing = [name for name, item in artifacts.items() if not bool(item["present"])]
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_dir": str(model_dir),
        "complete": not missing,
        "missing": missing,
        "artifacts": artifacts,
        "training": {
            "projection_report": _read_json(projection_report),
        },
        "quality": {
            "eval_report": _read_json(eval_report),
            "latency_report": _read_json(latency_report),
        },
    }


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Write Waver's production artifact manifest.")
    parser.add_argument("--model-dir", type=Path, default=Path(settings.model_dir))
    parser.add_argument(
        "--projection-path",
        type=Path,
        default=Path(settings.projection_model_path or "models/projection.npz"),
    )
    parser.add_argument("--reranker-model", default=settings.reranker_model)
    parser.add_argument(
        "--reranker-quantized",
        action=argparse.BooleanOptionalAction,
        default=settings.reranker_quantized,
    )
    parser.add_argument(
        "--mrl-metadata",
        type=Path,
        default=Path(settings.model_dir) / "mrl/mrl.json",
    )
    parser.add_argument(
        "--projection-report",
        type=Path,
        default=Path("models/projection.report.json"),
    )
    parser.add_argument("--latency-report", type=Path, default=Path(".reports/latency-gate.json"))
    parser.add_argument("--eval-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path(settings.model_dir) / "artifacts.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        model_dir=args.model_dir,
        projection_path=args.projection_path,
        reranker_model=args.reranker_model,
        reranker_quantized=args.reranker_quantized,
        mrl_metadata=args.mrl_metadata,
        projection_report=args.projection_report,
        latency_report=args.latency_report,
        eval_report=args.eval_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
