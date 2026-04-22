from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


def _default_eval_root() -> Path:
    return Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "eval"


def _default_fixtures_dir() -> Path:
    return _default_eval_root() / "tickets"


def _resolve_dynamic_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    root_relative = _default_eval_root() / path
    if root_relative.exists():
        return root_relative
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Waver promise-oriented evals.")
    parser.add_argument(
        "--profile",
        choices=["smoke", "regression", "latency", "semantic"],
        default=None,
        help="Static evaluation profile to run.",
    )
    parser.add_argument(
        "--layer",
        choices=["exact_only", "semantic_only", "rerank_only", "planner_only"],
        default=None,
        help="Run a layer-isolated eval.",
    )
    parser.add_argument(
        "--dynamic",
        type=Path,
        default=None,
        help="Run a dynamic-corpus fixture, relative to tests/fixtures/eval if needed.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=_default_fixtures_dir(),
        help="Directory containing static ticket eval fixture packs.",
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
        help="Optional top-k override for all static cases.",
    )
    parser.add_argument(
        "--cold-gate",
        action="store_true",
        help="Assert static eval does not depend on cache or projection writes.",
    )
    return parser.parse_args()


async def _run_dynamic(args: argparse.Namespace) -> int:
    from app.retrieval.eval_dynamic import EvalDynamicConfig, run_dynamic_eval
    from app.retrieval.eval_harness import format_scorecard_table

    scorecard = await run_dynamic_eval(
        EvalDynamicConfig(
            fixture_path=_resolve_dynamic_path(args.dynamic),
            top_k=args.top_k or 5,
        )
    )
    print(format_scorecard_table(scorecard))
    print()
    print(json.dumps(scorecard.to_dict(), indent=2, sort_keys=True))
    return scorecard.exit_code


def _run_static(args: argparse.Namespace) -> int:
    from app.retrieval.eval_harness import (
        build_run_config,
        format_scorecard_table,
        run_eval,
        run_layer_eval,
    )

    profile = args.profile or "smoke"
    config = build_run_config(
        profile,
        args.fixtures_dir,
        semantic_mode=args.semantic_mode,
        top_k_override=args.top_k,
        cold_gate=args.cold_gate,
    )
    scorecard = run_layer_eval(args.layer, config) if args.layer else run_eval(config)
    print(format_scorecard_table(scorecard))
    print()
    print(json.dumps(scorecard.to_dict(), indent=2, sort_keys=True))
    return scorecard.exit_code


def main() -> int:
    _bootstrap_path()
    args = parse_args()
    if args.dynamic is not None:
        return asyncio.run(_run_dynamic(args))
    return _run_static(args)


if __name__ == "__main__":
    raise SystemExit(main())
