"""End-to-end frame processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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
    edge_detector: EdgeDetector = field(init=False)
    char_mapper: CharMapper = field(init=False)

    def __post_init__(self) -> None:
        self.edge_detector = EdgeDetector(backend=self.edge_backend, threshold=self.threshold)
        self.char_mapper = CharMapper(charset=self.charset)

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
            ascii_frame = self.char_mapper.map_logits(logits, grid_shape=tile_batch.grid_shape)
        return FrameArtifacts(
            source_frame=frame_bgr,
            grayscale_frame=grayscale,
            edge_maps=edge_maps,
            ascii_frame=ascii_frame,
            logits=logits,
            diagnostics=diagnostics,
        )
