from glyphcast.render.terminal import render_terminal_frame
from glyphcast.types import AsciiFrame


def test_render_terminal_frame_returns_multiline_string() -> None:
    frame = AsciiFrame(characters=list("abcd"), width=2, height=2)

    rendered = render_terminal_frame(frame)

    assert rendered == "ab\ncd"
