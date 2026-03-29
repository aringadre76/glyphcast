"""Edge-map postprocessing."""

from __future__ import annotations

import cv2
import numpy as np


def postprocess_edge_probabilities(
    probabilities: np.ndarray,
    threshold: float = 0.3,
    morph_kernel_size: int = 1,
) -> np.ndarray:
    binary = (probabilities >= threshold).astype(np.float32)
    if morph_kernel_size <= 1:
        return binary
    kernel = np.ones((morph_kernel_size, morph_kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed.astype(np.float32)
