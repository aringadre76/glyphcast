"""Post-logit background suppression helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_AGENT_DEBUG_LOG = Path("/home/robot/glyphcast/.cursor/debug-f9ce89.log")
_AGENT_SESSION_ID = "f9ce89"


def agent_debug_append(record: dict) -> None:
    """Append one compact NDJSON line (session instrumentation)."""
    record.setdefault("sessionId", _AGENT_SESSION_ID)
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    _AGENT_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _AGENT_DEBUG_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _quantiles_3(arr: np.ndarray) -> list[float]:
    if arr.size == 0:
        return [0.0, 0.0, 0.0]
    q = np.percentile(arr.astype(np.float64), [10.0, 50.0, 90.0])
    return [float(q[0]), float(q[1]), float(q[2])]


def suppress_background_logits(
    logits: np.ndarray,
    tiles: np.ndarray,
    *,
    charset: str,
    grid_shape: tuple[int, int] | None = None,
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

    grayscale_mean = grayscale_tiles.mean(axis=(1, 2))
    edge_density = edge_tiles.mean(axis=(1, 2))
    grayscale_variance = grayscale_tiles.var(axis=(1, 2))

    sorted_logits = np.sort(logits, axis=1)
    confidence_margin_values = sorted_logits[:, -1] - sorted_logits[:, -2]

    low_information = (edge_density <= edge_threshold) & (grayscale_variance <= variance_threshold)
    low_confidence = confidence_margin_values <= confidence_margin
    boundary_background = np.zeros(logits.shape[0], dtype=bool)
    if grid_shape is not None:
        height, width = grid_shape
        if height * width == logits.shape[0] and height > 0 and width > 0:
            rows = np.repeat(np.arange(height), width)
            cols = np.tile(np.arange(width), height)
            on_boundary = (rows == 0) | (rows == height - 1) | (cols == 0) | (cols == width - 1)
            boundary_background = (
                on_boundary
                & (grayscale_variance <= variance_threshold)
                & (grayscale_mean >= 0.8)
            )
    blank_mask = (low_information & low_confidence) | boundary_background

    pre_argmax = np.argmax(logits, axis=1)
    pre_blank = int(np.sum(pre_argmax == blank_index))

    suppressed = logits.copy()
    for index in np.where(blank_mask)[0]:
        suppressed[index, blank_index] = np.max(suppressed[index]) + 1.0
    post_argmax = np.argmax(suppressed, axis=1)
    post_blank = int(np.sum(post_argmax == blank_index))

    # region agent log
    agent_debug_append(
        {
            "hypothesisId": "H1,H2,H4,H5",
            "where": "suppress_background_logits",
            "n": int(logits.shape[0]),
            "blankIdx": int(blank_index),
            "edgeTh": float(edge_threshold),
            "varTh": float(variance_threshold),
            "confTh": float(confidence_margin),
            "nLowInfo": int(np.count_nonzero(low_information)),
            "nLowConf": int(np.count_nonzero(low_confidence)),
            "nBoundaryBg": int(np.count_nonzero(boundary_background)),
            "nMask": int(np.count_nonzero(blank_mask)),
            "preBlank": pre_blank,
            "postBlank": post_blank,
            "grayQ": _quantiles_3(grayscale_mean),
            "edgeQ": _quantiles_3(edge_density),
            "varQ": _quantiles_3(grayscale_variance),
            "marginQ": _quantiles_3(confidence_margin_values),
        }
    )
    # endregion

    return suppressed.astype(np.float32)
