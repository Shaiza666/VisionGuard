"""Patch sampling and greedy coreset selection. NumPy only."""

from __future__ import annotations

import numpy as np


class PatchReservoir:
    """Uniform reservoir of patch rows, capped at a fixed capacity."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.buffer: np.ndarray | None = None
        self.seen = 0
        self.filled = 0

    def add(self, patches: np.ndarray) -> None:
        rows = np.ascontiguousarray(patches, dtype=np.float32)
        if rows.ndim != 2:
            raise ValueError("patches must have shape (N, D).")
        if not np.isfinite(rows).all():
            raise ValueError("patches contain non-finite values.")
        if self.buffer is None:
            self.buffer = np.empty((self.capacity, rows.shape[1]), dtype=np.float32)
        elif rows.shape[1] != self.buffer.shape[1]:
            raise ValueError("patch dimension changed while filling the reservoir.")
        for row in rows:
            self._add_row(row)

    def _add_row(self, row: np.ndarray) -> None:
        assert self.buffer is not None
        self.seen += 1
        if self.filled < self.capacity:
            self.buffer[self.filled] = row
            self.filled += 1
            return
        index = int(self.rng.integers(0, self.seen))
        if index < self.capacity:
            self.buffer[index] = row

    def values(self) -> np.ndarray:
        if self.buffer is None or self.filled == 0:
            raise ValueError("The patch reservoir is empty.")
        return self.buffer[: self.filled].copy()


def coreset_target_count(count: int, ratio: float, max_bank_size: int) -> int:
    if count < 1:
        raise ValueError("count must be at least 1.")
    target = max(1, int(round(count * ratio)))
    return min(target, max_bank_size, count)


def greedy_coreset(features: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Minimax facility-location coreset. Rows are copied from ``features``."""
    matrix = np.ascontiguousarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("features must have shape (N, D) with N > 0.")
    if target < 1:
        raise ValueError("target must be at least 1.")
    if target >= matrix.shape[0]:
        return matrix.copy()

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, matrix.shape[0]))
    selected = [start]
    chosen = np.zeros(matrix.shape[0], dtype=bool)
    chosen[start] = True
    nearest = _squared_distance(matrix, matrix[start])
    nearest[chosen] = -1.0

    while len(selected) < target:
        nxt = int(np.argmax(nearest))
        if nearest[nxt] <= 0:
            break
        selected.append(nxt)
        chosen[nxt] = True
        nearest = np.minimum(nearest, _squared_distance(matrix, matrix[nxt]))
        nearest[chosen] = -1.0
    return matrix[np.asarray(selected, dtype=np.int64)].copy()


def _squared_distance(features: np.ndarray, row: np.ndarray, chunk: int = 4096) -> np.ndarray:
    output = np.empty(features.shape[0], dtype=np.float32)
    vector = np.ascontiguousarray(row, dtype=np.float32)
    for start in range(0, features.shape[0], chunk):
        block = features[start : start + chunk]
        difference = block - vector
        output[start : start + chunk] = np.einsum("nd,nd->n", difference, difference)
    return output
