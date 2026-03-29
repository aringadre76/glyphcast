"""Batching helpers for tile inference."""

from __future__ import annotations

from typing import Iterator

import numpy as np


def iter_tile_batches(tiles: np.ndarray, batch_size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(tiles), batch_size):
        yield tiles[start : start + batch_size]
