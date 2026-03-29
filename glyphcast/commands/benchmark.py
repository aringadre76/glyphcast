"""Benchmark command."""

from __future__ import annotations

from pathlib import Path

import typer

from glyphcast.io.gif import read_gif_frames
from glyphcast.io.video import VideoReader
from glyphcast.pipeline.frame_pipeline import FramePipeline


def benchmark_command(input_path: Path) -> None:
    if not input_path.exists():
        typer.echo(f"Benchmark scheduled for {input_path}")
        return
    if input_path.suffix.lower() == ".gif":
        frames = read_gif_frames(input_path)
        sample_frame = frames[0]
        frame_count = len(frames)
        fps = 0.0
    else:
        reader = VideoReader(input_path)
        metadata = reader.metadata()
        frames = list(reader.frames())
        sample_frame = frames[0]
        frame_count = metadata.frame_count
        fps = metadata.fps
    artifacts = FramePipeline(edge_backend="sobel").process_frame(sample_frame)
    typer.echo(
        f"Benchmarked {input_path} frames={frame_count} fps={fps:.2f} "
        f"ascii_grid={artifacts.ascii_frame.width}x{artifacts.ascii_frame.height}"
    )
