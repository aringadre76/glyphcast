"""Benchmark command."""

from __future__ import annotations

from pathlib import Path

import typer

from glyphcast.config import GlyphcastConfig
from glyphcast.constants import CHARSET_PRESETS, MINIMAL_CHARSET
from glyphcast.io.gif import read_gif_frames
from glyphcast.io.video import VideoReader
from glyphcast.pipeline.frame_pipeline import FramePipeline


def benchmark_command(
    input_path: Path,
    preset: str = typer.Option("default", "--preset"),
) -> None:
    if not input_path.exists():
        typer.echo(f"Benchmark scheduled for {input_path}")
        return
    if input_path.suffix.lower() == ".gif":
        frames = read_gif_frames(input_path)
        frame_count = len(frames)
        fps = 0.0
    else:
        reader = VideoReader(input_path)
        metadata = reader.metadata()
        frames = list(reader.frames())
        frame_count = metadata.frame_count
        fps = metadata.fps

    if not frames:
        typer.echo(f"Benchmark skipped: no frames in {input_path}", err=True)
        return

    sample_frame = frames[0]
    config = GlyphcastConfig.from_preset(preset)
    pipeline = FramePipeline(
        edge_backend=config.runtime.edge_backend,
        device=config.runtime.device,
        mixed_precision=config.runtime.mixed_precision,
        batch_size=config.runtime.batch_size,
        glyph_mode=config.runtime.glyph_mode,
        edge_checkpoint=config.runtime.edge_checkpoint,
        edge_fallback_backend=config.runtime.edge_fallback_backend,
        char_model_path=config.runtime.char_model_path,
        glyph_fallback_mode=config.runtime.glyph_fallback_mode,
        fallback_device=config.runtime.fallback_device,
        charset=CHARSET_PRESETS.get(config.runtime.charset, MINIMAL_CHARSET),
        cell_size=(config.training.cell_width, config.training.cell_height),
    )
    artifacts = pipeline.process_frame(sample_frame)
    runtime_summary = " ".join(
        f"{key}={value}" for key, value in pipeline.runtime_summary().items()
    )
    typer.echo(
        f"Benchmarked {input_path} frames={frame_count} fps={fps:.2f} "
        f"ascii_grid={artifacts.ascii_frame.width}x{artifacts.ascii_frame.height} "
        f"{runtime_summary}"
    )
