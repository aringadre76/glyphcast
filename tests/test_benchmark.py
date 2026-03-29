"""Tests for benchmark command edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
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


def test_benchmark_uses_preset_runtime_and_reports_summary(tmp_path: Path) -> None:
    gif = tmp_path / "sample.gif"
    gif.write_bytes(b"gif")
    captured: dict[str, object] = {}

    class FakeAsciiFrame:
        width = 2
        height = 1

    class FakeArtifacts:
        ascii_frame = FakeAsciiFrame()
        diagnostics = {"device": "cuda", "edge_backend": "dexined", "glyph_mode": "cnn"}

    class FakePipeline:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def process_frame(self, _frame):
            return FakeArtifacts()

    with patch("glyphcast.commands.benchmark.read_gif_frames", return_value=[np.zeros((12, 8, 3), dtype=np.uint8)]):
        with patch("glyphcast.commands.benchmark.FramePipeline", FakePipeline):
            result = runner.invoke(app, ["benchmark", str(gif), "--preset", "fast"])

    assert result.exit_code == 0
    assert captured["device"] == "cuda"
    assert captured["glyph_mode"] == "cnn"
    assert "device=cuda" in result.output
    assert "glyph_mode=cnn" in result.output
