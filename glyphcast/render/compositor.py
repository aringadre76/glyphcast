"""Compositing helpers for ASCII overlays."""

from __future__ import annotations

import cv2
import numpy as np


def _resize_base_to_overlay(base_frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Resize source video frame to ASCII canvas size so blend ops are shape-compatible."""
    oh, ow = overlay.shape[:2]
    bh, bw = base_frame.shape[:2]
    if (bh, bw) == (oh, ow):
        return base_frame
    return cv2.resize(base_frame, (ow, oh), interpolation=cv2.INTER_AREA)


def composite_ascii_overlay(
    base_frame: np.ndarray | None,
    overlay: np.ndarray,
    mode: str = "ascii_only",
) -> np.ndarray:
    if base_frame is None or mode == "ascii_only":
        return overlay
    base_aligned = _resize_base_to_overlay(base_frame, overlay)
    if mode == "blended":
        return (
            (0.6 * base_aligned.astype(np.float32)) + (0.4 * overlay.astype(np.float32))
        ).astype(np.uint8)
    if mode == "source_tinted":
        tinted = base_aligned.copy().astype(np.float32)
        tinted[:, :, 1] = np.maximum(tinted[:, :, 1], overlay[:, :, 1])
        return np.clip(tinted, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported overlay mode: {mode}")
