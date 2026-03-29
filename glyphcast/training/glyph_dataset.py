"""Synthetic glyph dataset generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glyphcast.training.augment import augment_tile
from glyphcast.training.font_render import render_glyph_edge_tile, render_glyph_tile


@dataclass(slots=True)
class SyntheticGlyphDataset:
    tiles: np.ndarray
    labels: np.ndarray
    charset: list[str]


def build_synthetic_glyph_dataset(
    charset: str,
    cell_size: tuple[int, int] = (8, 12),
    augment: bool = False,
) -> SyntheticGlyphDataset:
    grayscale_tiles = []
    edge_tiles = []
    labels = []
    characters = list(charset)
    for index, character in enumerate(characters):
        grayscale = render_glyph_tile(character, cell_size=cell_size)
        edge = render_glyph_edge_tile(character, cell_size=cell_size)
        if augment:
            grayscale = augment_tile(grayscale)
            edge = augment_tile(edge, blur=False)
        grayscale_tiles.append(grayscale)
        edge_tiles.append(edge)
        labels.append(index)
    stacked = np.stack(
        [np.stack([gray, edge], axis=0) for gray, edge in zip(grayscale_tiles, edge_tiles)],
        axis=0,
    ).astype(np.float32)
    return SyntheticGlyphDataset(
        tiles=stacked,
        labels=np.asarray(labels, dtype=np.int64),
        charset=characters,
    )
