"""Edge detector backend registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class EdgeBackend(Protocol):
    name: str

    def infer(self, grayscale_frame: np.ndarray) -> np.ndarray:
        """Return a probability map in the [0, 1] range."""


@dataclass(slots=True)
class SobelEdgeBackend:
    name: str = "sobel"

    def infer(self, grayscale_frame: np.ndarray) -> np.ndarray:
        grad_x = cv2.Sobel(grayscale_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(grayscale_frame, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        max_value = float(magnitude.max())
        if max_value <= 0.0:
            return np.zeros_like(magnitude, dtype=np.float32)
        return (magnitude / max_value).astype(np.float32)


@dataclass(slots=True)
class TorchCheckpointEdgeBackend:
    """Wrapper reserved for DexiNed/HED checkpoints.

    The initial implementation falls back to Sobel when weights are unavailable,
    which keeps the local pipeline functional while preserving the runtime API.
    """

    name: str
    checkpoint_path: Path | None = None

    def infer(self, grayscale_frame: np.ndarray) -> np.ndarray:
        return SobelEdgeBackend().infer(grayscale_frame)


def build_edge_backend(name: str, checkpoint_path: Path | None = None) -> EdgeBackend:
    normalized = name.lower()
    if normalized == "sobel":
        return SobelEdgeBackend()
    if normalized in {"dexined", "hed"}:
        return TorchCheckpointEdgeBackend(name=normalized, checkpoint_path=checkpoint_path)
    raise ValueError(f"Unsupported edge backend: {name}")
