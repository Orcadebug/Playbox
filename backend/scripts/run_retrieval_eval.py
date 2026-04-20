from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def _default_fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "app" / "retrieval" / "eval_fixtures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval promise evaluation profiles.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "regression", "latency", "semantic"],
        required=True,
        help="Evaluation profile to run.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=_default_fixtures_dir(),
        help="Directory containing eval fixture packs.",
    )
    parser.add_argument(
        "--semantic-mode",
        choices=["deterministic", "real", "auto"],
        default=None,
        help="Semantic scoring mode override.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k override for all cases.",
    )
    return parser.parse_args()


def main() -> int:
    _bootstrap_path()
    from app.retrieval.eval_harness import (  # noqa: PLC0415
        build_run_config,
        format_scorecard_table,
        run_eval,
    )

    args = parse_args()
    config = build_run_config(
        args.profile,
        args.fixtures_dir,
        semantic_mode=args.semantic_mode,
        top_k_override=args.top_k,
    )
    scorecard = run_eval(config)

    print(format_scorecard_table(scorecard))
    print()
    print(json.dumps(scorecard.to_dict(), indent=2, sort_keys=True))

    return scorecard.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
