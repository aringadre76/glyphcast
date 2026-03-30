"""Dataset augmentations."""

from __future__ import annotations

import cv2
import numpy as np


def augment_tile(tile: np.ndarray, blur: bool = True, invert: bool = False) -> np.ndarray:
    augmented: np.ndarray = tile.astype(np.float32)
    if blur:
        augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=0.0)
    if invert:
        augmented = 1.0 - augmented
    return np.clip(augmented, 0.0, 1.0).astype(np.float32)
