import numpy as np

from glyphcast.pipeline.frame_pipeline import FramePipeline
from glyphcast.pipeline.tile_extractor import extract_tiles


def test_extract_tiles_uses_requested_cell_size() -> None:
    grayscale = np.arange(8 * 12, dtype=np.float32).reshape(12, 8)
    edge = grayscale.copy()

    batch = extract_tiles(grayscale, edge, cell_size=(4, 3))

    assert batch.grid_shape == (4, 2)
    assert batch.tiles.shape == (8, 2, 3, 4)


def test_frame_pipeline_produces_ascii_frame_for_simple_edge_image() -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[:, 8:, :] = 255

    pipeline = FramePipeline(
        edge_backend="sobel",
        charset=" .#",
        cell_size=(4, 3),
        threshold=0.2,
    )

    artifacts = pipeline.process_frame(frame)

    assert artifacts.ascii_frame.width == 4
    assert artifacts.ascii_frame.height == 4
    assert len(artifacts.ascii_frame.characters) == 16
    assert artifacts.logits is not None


def test_frame_pipeline_can_blank_low_information_tiles_after_logits(monkeypatch) -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)

    def fake_score_tiles(self, tiles: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.10, 0.20, 0.22]], dtype=np.float32), (tiles.shape[0], 1))

    monkeypatch.setattr("glyphcast.pipeline.char_mapper.CharMapper.score_tiles", fake_score_tiles)
    pipeline = FramePipeline(
        edge_backend="sobel",
        charset=" .#",
        cell_size=(4, 3),
        threshold=0.2,
        background_suppression=True,
    )

    artifacts = pipeline.process_frame(frame)

    assert artifacts.ascii_frame.as_text() == "    \n    \n    \n    "
