"""Compositing helpers for ASCII overlays."""

from __future__ import annotations

import numpy as np


def composite_ascii_overlay(
    base_frame: np.ndarray | None,
    overlay: np.ndarray,
    mode: str = "ascii_only",
) -> np.ndarray:
    if base_frame is None or mode == "ascii_only":
        return overlay
    if mode == "blended":
        return ((0.6 * base_frame.astype(np.float32)) + (0.4 * overlay.astype(np.float32))).astype(np.uint8)
    if mode == "source_tinted":
        tinted = base_frame.copy().astype(np.float32)
        tinted[:, :, 1] = np.maximum(tinted[:, :, 1], overlay[:, :, 1])
        return np.clip(tinted, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported overlay mode: {mode}")
