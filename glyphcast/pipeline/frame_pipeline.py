"""End-to-end frame processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glyphcast.pipeline.background_suppression import suppress_background_logits
from glyphcast.pipeline.char_mapper import CharMapper
from glyphcast.pipeline.edge_detector import EdgeDetector
from glyphcast.pipeline.preprocess import prepare_grayscale_frame
from glyphcast.pipeline.tile_extractor import extract_tiles
from glyphcast.types import FrameArtifacts
from glyphcast.utils.profiling import timed_step


@dataclass(slots=True)
class FramePipeline:
    edge_backend: str = "dexined"
    charset: str = " .#"
    cell_size: tuple[int, int] = (8, 12)
    threshold: float = 0.3
    device: str = "cpu"
    mixed_precision: bool = False
    batch_size: int = 512
    glyph_mode: str = "template"
    edge_checkpoint: str | None = None
    edge_fallback_backend: str | None = None
    char_model_path: str | None = None
    glyph_fallback_mode: str | None = None
    fallback_device: str = "cpu"
    background_suppression: bool = False
    background_edge_threshold: float = 0.05
    background_variance_threshold: float = 0.08
    background_confidence_margin: float = 0.15
    edge_detector: EdgeDetector = field(init=False)
    char_mapper: CharMapper = field(init=False)

    def __post_init__(self) -> None:
        checkpoint_path = None if self.edge_checkpoint is None else Path(self.edge_checkpoint)
        self.edge_detector = EdgeDetector(
            backend=self.edge_backend,
            threshold=self.threshold,
            checkpoint_path=checkpoint_path,
            device=self.device,
            fallback_device=self.fallback_device,
            mixed_precision=self.mixed_precision,
            fallback_backend=self.edge_fallback_backend,
        )
        char_model_path = None if self.char_model_path is None else Path(self.char_model_path)
        self.char_mapper = CharMapper(
            charset=self.charset,
            mode=self.glyph_mode,
            model_path=char_model_path,
            device=self.device,
            fallback_device=self.fallback_device,
            batch_size=self.batch_size,
            fallback_mode=self.glyph_fallback_mode,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> FrameArtifacts:
        diagnostics: dict[str, float] = {}
        with timed_step(diagnostics, "grayscale"):
            grayscale = prepare_grayscale_frame(frame_bgr)
        with timed_step(diagnostics, "edges"):
            edge_maps = self.edge_detector.detect(frame_bgr)
        with timed_step(diagnostics, "tiles"):
            tile_batch = extract_tiles(grayscale, edge_maps.binary, cell_size=self.cell_size)
        with timed_step(diagnostics, "classification"):
            logits = self.char_mapper.score_tiles(tile_batch.tiles)
            if self.background_suppression:
                logits = suppress_background_logits(
                    logits,
                    tile_batch.tiles,
                    charset=self.charset,
                    grid_shape=tile_batch.grid_shape,
                    edge_threshold=self.background_edge_threshold,
                    variance_threshold=self.background_variance_threshold,
                    confidence_margin=self.background_confidence_margin,
                )
            ascii_frame = self.char_mapper.map_logits(logits, grid_shape=tile_batch.grid_shape)
        return FrameArtifacts(
            source_frame=frame_bgr,
            grayscale_frame=grayscale,
            edge_maps=edge_maps,
            ascii_frame=ascii_frame,
            logits=logits,
            diagnostics={**diagnostics, **self.runtime_summary()},
        )

    def runtime_summary(self) -> dict[str, str]:
        return {
            "device": self.device,
            **self.edge_detector.runtime_summary(),
            **self.char_mapper.runtime_summary(),
        }
