import numpy as np

from glyphcast.pipeline.background_suppression import suppress_background_logits
from glyphcast.pipeline.char_mapper import CharMapper


def test_suppress_background_logits_blanks_low_information_tile() -> None:
    logits = np.array([[0.10, 0.20, 0.22]], dtype=np.float32)
    tiles = np.zeros((1, 2, 12, 8), dtype=np.float32)

    suppressed = suppress_background_logits(logits, tiles, charset=" .#")

    assert np.argmax(suppressed, axis=1).tolist() == [0]


def test_suppress_background_logits_preserves_confident_foreground_tile() -> None:
    logits = np.array([[0.10, 0.20, 3.00]], dtype=np.float32)
    tiles = np.zeros((1, 2, 12, 8), dtype=np.float32)
    tiles[0, 0, :, ::2] = 1.0
    tiles[0, 1, :, :] = 1.0

    suppressed = suppress_background_logits(logits, tiles, charset=" .#")

    assert np.argmax(suppressed, axis=1).tolist() == [2]


def test_suppress_background_logits_keeps_grid_decode_order() -> None:
    logits = np.array(
        [
            [0.10, 0.20, 0.22],
            [0.10, 0.20, 3.00],
        ],
        dtype=np.float32,
    )
    tiles = np.zeros((2, 2, 12, 8), dtype=np.float32)
    tiles[1, 0, :, ::2] = 1.0
    tiles[1, 1, :, :] = 1.0
    mapper = CharMapper(charset=" .#")

    suppressed = suppress_background_logits(logits, tiles, charset=" .#")
    frame = mapper.map_logits(suppressed, grid_shape=(1, 2))

    assert frame.as_text() == " #"


def test_suppress_background_logits_blanks_low_variance_boundary_tile_even_with_edges() -> None:
    logits = np.array(
        [
            [0.10, 0.20, 0.90],
            [0.90, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    tiles = np.zeros((2, 2, 12, 8), dtype=np.float32)
    tiles[0, 0, :, :] = 0.97
    tiles[0, 1, :, :] = 1.0

    suppressed = suppress_background_logits(
        logits,
        tiles,
        charset=" .#",
        grid_shape=(1, 2),
    )

    assert np.argmax(suppressed, axis=1).tolist() == [0, 0]


def test_suppress_background_logits_blanks_bright_uniform_tile_even_with_spurious_edges() -> None:
    logits = np.array([[0.10, 0.20, 0.90]], dtype=np.float32)
    tiles = np.zeros((1, 2, 12, 8), dtype=np.float32)
    tiles[0, 0, :, :] = 0.97
    tiles[0, 1, :, :] = 1.0

    suppressed = suppress_background_logits(logits, tiles, charset=" .#")

    assert np.argmax(suppressed, axis=1).tolist() == [0]
