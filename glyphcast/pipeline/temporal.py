"""Temporal smoothing for ASCII logits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from glyphcast.pipeline.optical_flow import estimate_scene_delta
from glyphcast.pipeline.state import TemporalState


@dataclass(slots=True)
class TemporalSmoother:
    alpha: float = 0.7
    confidence_margin: float = 0.2
    scene_cut_threshold: float = 0.3
    state: TemporalState = field(default_factory=TemporalState)

    def update(self, logits: np.ndarray, edge_map: np.ndarray) -> np.ndarray:
        if self.state.previous_logits is None or self.state.previous_edge_map is None:
            self.state.previous_logits = logits.copy()
            self.state.previous_edge_map = edge_map.copy()
            return logits

        scene_delta = estimate_scene_delta(self.state.previous_edge_map, edge_map)
        if scene_delta >= self.scene_cut_threshold:
            self.state.previous_logits = logits.copy()
            self.state.previous_edge_map = edge_map.copy()
            return logits

        blended = (self.alpha * self.state.previous_logits) + ((1.0 - self.alpha) * logits)
        previous_best = np.max(self.state.previous_logits, axis=1)
        current_best = np.max(logits, axis=1)
        keep_previous = (previous_best - current_best) >= -self.confidence_margin
        smoothed = logits.copy()
        smoothed[keep_previous] = blended[keep_previous]

        self.state.previous_logits = smoothed.copy()
        self.state.previous_edge_map = edge_map.copy()
        return smoothed
