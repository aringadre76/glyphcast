"""Terminal rendering helpers."""

from __future__ import annotations

import time
from typing import Iterable

from rich.console import Console

from glyphcast.types import AsciiFrame


def render_terminal_frame(frame: AsciiFrame) -> str:
    return frame.as_text()


def play_terminal_frames(
    frames: Iterable[AsciiFrame], fps: float = 24.0, console: Console | None = None
) -> None:
    stream = console or Console()
    delay = 1.0 / fps if fps > 0 else 0.0
    for frame in frames:
        stream.clear()
        stream.print(render_terminal_frame(frame))
        if delay > 0:
            time.sleep(delay)
