"""Edge detector orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glyphcast.models.edge_backends import EdgeBackend, build_edge_backend
from glyphcast.pipeline.postprocess import postprocess_edge_probabilities
from glyphcast.pipeline.preprocess import denoise_frame, prepare_grayscale_frame
from glyphcast.types import EdgeMaps


@dataclass(slots=True)
class EdgeDetector:
    backend: str = "dexined"
    threshold: float = 0.3
    checkpoint_path: Path | None = None
    denoise_kernel_size: int = 3
    device: str = "cpu"
    fallback_device: str = "cpu"
    mixed_precision: bool = False
    fallback_backend: str | None = None
    edge_backend: EdgeBackend = field(init=False)

    def __post_init__(self) -> None:
        self.edge_backend = build_edge_backend(
            self.backend,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            fallback_device=self.fallback_device,
            mixed_precision=self.mixed_precision,
            fallback_backend=self.fallback_backend,
        )

    def detect(self, frame_bgr: np.ndarray) -> EdgeMaps:
        grayscale = prepare_grayscale_frame(frame_bgr)
        denoised = denoise_frame(grayscale, kernel_size=self.denoise_kernel_size)
        probabilities = self.edge_backend.infer(denoised)
        binary = postprocess_edge_probabilities(probabilities, threshold=self.threshold)
        return EdgeMaps(probability=probabilities, binary=binary)
