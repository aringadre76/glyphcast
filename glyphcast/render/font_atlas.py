"""Pre-rendered glyph atlas for video overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glyphcast.training.font_render import render_glyph_tile


@dataclass(slots=True)
class GlyphAtlas:
    glyphs: dict[str, np.ndarray]
    cell_size: tuple[int, int]


def build_glyph_atlas(charset: str, cell_size: tuple[int, int]) -> GlyphAtlas:
    glyphs = {character: (render_glyph_tile(character, cell_size=cell_size) * 255).astype(np.uint8) for character in charset}
    return GlyphAtlas(glyphs=glyphs, cell_size=cell_size)
