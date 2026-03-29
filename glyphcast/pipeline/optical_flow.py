"""Motion estimation helpers."""

from __future__ import annotations

import numpy as np


def estimate_scene_delta(previous_edge_map: np.ndarray, current_edge_map: np.ndarray) -> float:
    return float(np.mean(np.abs(previous_edge_map - current_edge_map)))
