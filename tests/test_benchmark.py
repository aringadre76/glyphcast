"""Tests for benchmark command edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from glyphcast.cli import app
from glyphcast.types import FrameMetadata

runner = CliRunner()


def test_benchmark_skips_empty_gif_without_index_error(tmp_path: Path) -> None:
    gif = tmp_path / "empty.gif"
    gif.write_bytes(b"x")

    with patch("glyphcast.commands.benchmark.read_gif_frames", return_value=[]):
        result = runner.invoke(app, ["benchmark", str(gif)])

    assert result.exit_code == 0
    combined = f"{result.stdout}\n{result.stderr}"
    assert "no frames" in combined.lower()


def test_benchmark_skips_empty_video_frame_list_without_index_error(tmp_path: Path) -> None:
    mp4 = tmp_path / "empty.mp4"
    mp4.write_bytes(b"")

    class FakeReader:
        def metadata(self) -> FrameMetadata:
            return FrameMetadata(
                source_path=mp4,
                fps=24.0,
                frame_count=99,
                width=64,
                height=48,
            )

        def frames(self):
            return iter(())

    with patch("glyphcast.commands.benchmark.VideoReader", return_value=FakeReader()):
        result = runner.invoke(app, ["benchmark", str(mp4)])

    assert result.exit_code == 0
    combined = f"{result.stdout}\n{result.stderr}"
    assert "no frames" in combined.lower()
