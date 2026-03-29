"""Video overlay rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from glyphcast.render.compositor import composite_ascii_overlay
from glyphcast.render.font_atlas import build_glyph_atlas
from glyphcast.types import AsciiFrame


def render_ascii_overlay(frame: AsciiFrame, cell_size: tuple[int, int] = (8, 12)) -> np.ndarray:
    atlas = build_glyph_atlas("".join(sorted(set(frame.characters))), cell_size=cell_size)
    cell_width, cell_height = cell_size
    canvas = np.zeros((frame.height * cell_height, frame.width * cell_width), dtype=np.uint8)
    for index, character in enumerate(frame.characters):
        row, col = divmod(index, frame.width)
        y0 = row * cell_height
        x0 = col * cell_width
        canvas[y0 : y0 + cell_height, x0 : x0 + cell_width] = atlas.glyphs.get(
            character,
            np.zeros((cell_height, cell_width), dtype=np.uint8),
        )
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def write_ascii_video(
    frames: Iterable[AsciiFrame],
    output_path: Path,
    fps: float = 24.0,
    overlay_mode: str = "ascii_only",
    source_frames: Iterable[np.ndarray] | None = None,
    cell_size: tuple[int, int] = (8, 12),
) -> Path:
    frames = list(frames)
    if not frames:
        raise ValueError("No ASCII frames provided")
    ascii_overlays = [render_ascii_overlay(frame, cell_size=cell_size) for frame in frames]
    height, width = ascii_overlays[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    source_sequence = list(source_frames) if source_frames is not None else [None] * len(ascii_overlays)
    try:
        for overlay, source in zip(ascii_overlays, source_sequence):
            writer.write(composite_ascii_overlay(source, overlay, mode=overlay_mode))
    finally:
        writer.release()
    return output_path
