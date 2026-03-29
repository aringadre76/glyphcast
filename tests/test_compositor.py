import numpy as np

from glyphcast.render.compositor import composite_ascii_overlay


def test_composite_blended_resizes_mismatched_base_to_overlay_shape() -> None:
    overlay = np.zeros((24, 16, 3), dtype=np.uint8)
    overlay[:, :, :] = 200
    base = np.ones((96, 64, 3), dtype=np.uint8) * 50

    out = composite_ascii_overlay(base, overlay, mode="blended")

    assert out.shape == overlay.shape
    assert out.dtype == np.uint8


def test_composite_source_tinted_resizes_mismatched_base_to_overlay_shape() -> None:
    overlay = np.zeros((12, 8, 3), dtype=np.uint8)
    overlay[:, :, 1] = 100
    base = np.full((36, 24, 3), 30, dtype=np.uint8)

    out = composite_ascii_overlay(base, overlay, mode="source_tinted")

    assert out.shape == overlay.shape
    assert out.dtype == np.uint8


def test_composite_blended_skips_resize_when_shapes_already_match() -> None:
    overlay = np.zeros((10, 10, 3), dtype=np.uint8)
    base = np.ones((10, 10, 3), dtype=np.uint8) * 100

    out = composite_ascii_overlay(base, overlay, mode="blended")

    assert out.shape == (10, 10, 3)
