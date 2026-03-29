import numpy as np

from glyphcast.render.video_overlay import render_ascii_overlay
from glyphcast.types import AsciiFrame


def test_render_ascii_overlay_returns_bgr_canvas() -> None:
    frame = AsciiFrame(characters=list("ab  "), width=2, height=2)

    overlay = render_ascii_overlay(frame, cell_size=(8, 12))

    assert overlay.shape == (24, 16, 3)
    assert overlay.dtype == np.uint8
    assert int(overlay.sum()) > 0
