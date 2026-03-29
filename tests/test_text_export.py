from pathlib import Path

from glyphcast.render.text_export import export_ascii_frames
from glyphcast.types import AsciiFrame


def test_export_ascii_frames_writes_sequence_file(tmp_path: Path) -> None:
    frames = [AsciiFrame(characters=list("abcd"), width=2, height=2)]
    output_path = tmp_path / "frames.txt"

    export_ascii_frames(frames, output_path)

    assert output_path.read_text(encoding="utf-8") == "ab\ncd"
