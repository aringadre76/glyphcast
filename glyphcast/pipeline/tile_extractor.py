"""Tile extraction utilities."""

from __future__ import annotations

import numpy as np

from glyphcast.types import TileBatch


def extract_tiles(
    grayscale_frame: np.ndarray,
    edge_frame: np.ndarray,
    cell_size: tuple[int, int],
) -> TileBatch:
    cell_width, cell_height = cell_size
    height, width = grayscale_frame.shape
    grid_width = width // cell_width
    grid_height = height // cell_height
    cropped_width = grid_width * cell_width
    cropped_height = grid_height * cell_height
    grayscale = grayscale_frame[:cropped_height, :cropped_width]
    edges = edge_frame[:cropped_height, :cropped_width]

    tiles = []
    for row in range(grid_height):
        for col in range(grid_width):
            y0 = row * cell_height
            y1 = y0 + cell_height
            x0 = col * cell_width
            x1 = x0 + cell_width
            tiles.append(
                np.stack(
                    [grayscale[y0:y1, x0:x1], edges[y0:y1, x0:x1]],
                    axis=0,
                )
            )
    if not tiles:
        return TileBatch(
            tiles=np.zeros((0, 2, cell_height, cell_width), dtype=np.float32), grid_shape=(0, 0)
        )
    return TileBatch(tiles=np.stack(tiles).astype(np.float32), grid_shape=(grid_height, grid_width))
