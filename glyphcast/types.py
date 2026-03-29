"""Shared dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ArrayF32 = np.ndarray


@dataclass(slots=True)
class FrameMetadata:
    source_path: Path
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(slots=True)
class AsciiFrame:
    characters: list[str]
    width: int
    height: int
    grayscale: ArrayF32 | None = None

    def as_text(self) -> str:
        rows = [
            "".join(self.characters[row * self.width : (row + 1) * self.width])
            for row in range(self.height)
        ]
        return "\n".join(rows)


@dataclass(slots=True)
class EdgeMaps:
    probability: ArrayF32
    binary: ArrayF32


@dataclass(slots=True)
class TileBatch:
    tiles: ArrayF32
    grid_shape: tuple[int, int]


@dataclass(slots=True)
class FrameArtifacts:
    source_frame: ArrayF32
    grayscale_frame: ArrayF32
    edge_maps: EdgeMaps
    ascii_frame: AsciiFrame
    logits: ArrayF32 | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
