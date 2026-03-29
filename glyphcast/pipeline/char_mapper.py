"""Map tile logits to ASCII characters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glyphcast.types import AsciiFrame
from glyphcast.training.glyph_dataset import build_synthetic_glyph_dataset


@dataclass(slots=True)
class CharMapper:
    charset: str

    def score_tiles(self, tiles: np.ndarray) -> np.ndarray:
        templates = build_synthetic_glyph_dataset(self.charset, cell_size=(tiles.shape[-1], tiles.shape[-2]))
        flattened_tiles = tiles.reshape(tiles.shape[0], -1)
        flattened_templates = templates.tiles.reshape(templates.tiles.shape[0], -1)
        distances = ((flattened_tiles[:, None, :] - flattened_templates[None, :, :]) ** 2).mean(axis=2)
        return -distances.astype(np.float32)

    def map_logits(self, logits: np.ndarray, grid_shape: tuple[int, int]) -> AsciiFrame:
        indices = np.argmax(logits, axis=1)
        characters = [self.charset[index] for index in indices.tolist()]
        height, width = grid_shape
        return AsciiFrame(characters=characters, width=width, height=height)
