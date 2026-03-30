"""Font rasterization helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(
    font_path: str | Path | None, font_size: int
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(str(font_path), size=font_size)
    except OSError:
        return ImageFont.load_default()


def render_glyph_tile(
    character: str,
    cell_size: tuple[int, int] = (8, 12),
    font_path: str | Path | None = None,
) -> np.ndarray:
    width, height = cell_size
    image = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(image)
    font = load_font(font_path, font_size=max(height, 8))
    bbox = draw.textbbox((0, 0), character, font=font)
    glyph_width = bbox[2] - bbox[0]
    glyph_height = bbox[3] - bbox[1]
    x_pos = max((width - glyph_width) // 2 - bbox[0], 0)
    y_pos = max((height - glyph_height) // 2 - bbox[1], 0)
    draw.text((x_pos, y_pos), character, fill=255, font=font)
    return np.asarray(image, dtype=np.float32) / 255.0


def render_glyph_edge_tile(
    character: str,
    cell_size: tuple[int, int] = (8, 12),
    font_path: str | Path | None = None,
) -> np.ndarray:
    raster = render_glyph_tile(character, cell_size=cell_size, font_path=font_path)
    edges = cv2.Canny((raster * 255).astype(np.uint8), 64, 128)
    return edges.astype(np.float32) / 255.0
