from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy import sparse

from app.schemas.documents import Chunk


def _source_key(chunk: Chunk) -> str:
    source_id = chunk.metadata.get("source_id")
    return str(source_id) if source_id is not None else chunk.source_name


def _int_value(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _chunk_index(chunk: Chunk) -> int | None:
    return _int_value(chunk.metadata.get("chunk_index"))


def _same_page(left: Chunk, right: Chunk) -> bool:
    return (
        left.location.page_number is not None
        and left.location.page_number == right.location.page_number
    )


def _adjacent_row(left: Chunk, right: Chunk) -> bool:
    return (
        left.location.row_number is not None
        and right.location.row_number is not None
        and abs(left.location.row_number - right.location.row_number) == 1
    )


def _same_section(left: Chunk, right: Chunk) -> bool:
    return left.location.section is not None and left.location.section == right.location.section


def build_adjacency(candidates: list[Chunk], max_edges: int = 8) -> sparse.csr_matrix:
    n = len(candidates)
    if n == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float32)

    by_id = {chunk.chunk_id: i for i, chunk in enumerate(candidates)}
    edges: dict[int, dict[int, float]] = defaultdict(dict)

    for i, left in enumerate(candidates):
        source = _source_key(left)
        left_index = _chunk_index(left)
        adjacent_ids = left.metadata.get("adjacent_chunk_ids")
        if isinstance(adjacent_ids, list):
            for chunk_id in adjacent_ids:
                j = by_id.get(str(chunk_id))
                if j is not None and j != i:
                    edges[i][j] = max(edges[i].get(j, 0.0), 1.0)

        for j, right in enumerate(candidates):
            if i == j or source != _source_key(right):
                continue
            weight = 0.0
            right_index = _chunk_index(right)
            if (
                left_index is not None
                and right_index is not None
                and abs(left_index - right_index) == 1
            ):
                weight = max(weight, 1.0)
            if _adjacent_row(left, right):
                weight = max(weight, 0.8)
            if _same_page(left, right):
                weight = max(weight, 0.6)
            if _same_section(left, right):
                weight = max(weight, 0.5)
            if weight > 0:
                edges[i][j] = max(edges[i].get(j, 0.0), weight)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row, row_edges in edges.items():
        top_edges = sorted(row_edges.items(), key=lambda item: (-item[1], item[0]))[:max_edges]
        total = sum(weight for _, weight in top_edges)
        if total <= 0:
            continue
        for col, weight in top_edges:
            rows.append(row)
            cols.append(col)
            data.append(weight / total)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
