from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

from app.retrieval.sparse_projection import ProjectionConfig, save_projection


def build_initial_projection(
    hash_features: int,
    dim: int,
    density: float,
    seed: int,
) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    matrix = sparse.random(
        hash_features,
        dim,
        density=density,
        format="csr",
        dtype=np.float32,
        random_state=rng,
    )
    return matrix.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a sparse projection checkpoint.")
    parser.add_argument("--out", type=Path, default=Path("models/projection.npz"))
    parser.add_argument("--hash-features", type=int, default=262144)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectionConfig(hash_features=args.hash_features, dim=args.dim)
    W = build_initial_projection(
        hash_features=config.hash_features,
        dim=config.dim,
        density=args.density,
        seed=args.seed,
    )
    save_projection(args.out, W, config)
    print(f"saved projection checkpoint to {args.out}")


if __name__ == "__main__":
    main()
