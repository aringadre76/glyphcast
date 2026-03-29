"""Temporal state containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TemporalState:
    previous_logits: np.ndarray | None = None
    previous_edge_map: np.ndarray | None = None
