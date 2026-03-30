"""Post-logit background suppression helpers."""

from __future__ import annotations

import numpy as np


def suppress_background_logits(
    logits: np.ndarray,
    tiles: np.ndarray,
    *,
    charset: str,
    edge_threshold: float = 0.05,
    variance_threshold: float = 0.08,
    confidence_margin: float = 0.15,
) -> np.ndarray:
    """Bias low-information tiles toward the blank class."""

    if logits.size == 0 or tiles.size == 0:
        return logits

    blank_index = charset.index(" ") if " " in charset else 0
    grayscale_tiles = tiles[:, 0]
    edge_tiles = tiles[:, 1]

    edge_density = edge_tiles.mean(axis=(1, 2))
    grayscale_variance = grayscale_tiles.var(axis=(1, 2))

    sorted_logits = np.sort(logits, axis=1)
    confidence_margin_values = sorted_logits[:, -1] - sorted_logits[:, -2]

    low_information = (edge_density <= edge_threshold) & (grayscale_variance <= variance_threshold)
    low_confidence = confidence_margin_values <= confidence_margin
    blank_mask = low_information | (low_information & low_confidence)

    suppressed = logits.copy()
    for index in np.where(blank_mask)[0]:
        suppressed[index, blank_index] = np.max(suppressed[index]) + 1.0
    return suppressed.astype(np.float32)
