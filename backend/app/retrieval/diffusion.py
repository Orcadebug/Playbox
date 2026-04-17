from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(slots=True)
class DiffusionConfig:
    steps: int = 3
    beta: float = 0.5
    gamma: float = 0.3
    delta: float = 0.2


def diffuse(
    A: csr_matrix,
    L: np.ndarray,
    S: np.ndarray,
    steps: int = 3,
    beta: float = 0.5,
    gamma: float = 0.3,
    delta: float = 0.2,
) -> np.ndarray:
    L = np.asarray(L, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    C = np.zeros_like(L, dtype=np.float32)
    for _ in range(max(0, steps)):
        C = np.tanh(beta * (A @ C) + gamma * L + delta * S)
        C = np.clip(C, -5.0, 5.0)
    return C.astype(np.float32)
