"""Text output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from glyphcast.types import AsciiFrame


def export_ascii_frames(frames: Iterable[AsciiFrame], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n\n".join(frame.as_text() for frame in frames)
    output_path.write_text(payload, encoding="utf-8")
    return output_path
