"""Render command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer

from glyphcast.config import GlyphcastConfig
from glyphcast.constants import CHARSET_PRESETS, MINIMAL_CHARSET
from glyphcast.io.gif import read_gif_frames
from glyphcast.io.video import VideoReader
from glyphcast.pipeline.background_suppression import agent_debug_append
from glyphcast.pipeline.frame_pipeline import FramePipeline
from glyphcast.pipeline.temporal import TemporalSmoother
from glyphcast.render.terminal import play_terminal_frames
from glyphcast.render.text_export import export_ascii_frames
from glyphcast.render.video_overlay import write_ascii_video


def _load_frames(input_path: Path) -> list:
    suffix = input_path.suffix.lower()
    if suffix == ".gif":
        return read_gif_frames(input_path)
    return list(VideoReader(input_path).frames())


def _format_runtime_summary(pipeline: FramePipeline) -> str:
    summary = pipeline.runtime_summary()
    ordered_keys = [
        "device",
        "edge_backend",
        "edge_device",
        "edge_checkpoint",
        "glyph_mode",
        "glyph_device",
        "char_model_path",
    ]
    parts = [f"{key}={summary[key]}" for key in ordered_keys if key in summary]
    return " ".join(parts)


def render_command(
    input_path: Path,
    mode: str = typer.Option("terminal", "--mode"),
    output: Path | None = typer.Option(None, "--output"),
    preset: str = typer.Option("default", "--preset"),
) -> None:
    config = GlyphcastConfig.from_preset(preset)
    target = output or Path("artifacts/renders/output.txt")
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
        background_suppression=config.runtime.background_suppression,
        background_edge_threshold=config.runtime.background_edge_threshold,
        background_variance_threshold=config.runtime.background_variance_threshold,
        background_confidence_margin=config.runtime.background_confidence_margin,
        charset=CHARSET_PRESETS.get(config.runtime.charset, MINIMAL_CHARSET),
        cell_size=(config.training.cell_width, config.training.cell_height),
    )
    smoother = TemporalSmoother() if config.runtime.smoothing else None
    ascii_frames = []
    source_frames = _load_frames(input_path)
    charset = pipeline.charset
    blank_idx = charset.index(" ") if " " in charset else 0
    for frame_idx, frame_bgr in enumerate(source_frames):
        artifacts = pipeline.process_frame(frame_bgr)
        if smoother is not None and artifacts.logits is not None:
            pre_logits = artifacts.logits
            pre_blank = int(np.sum(np.argmax(pre_logits, axis=1) == blank_idx))
            smoothed_logits = smoother.update(pre_logits, artifacts.edge_maps.binary)
            post_blank = int(np.sum(np.argmax(smoothed_logits, axis=1) == blank_idx))
            # region agent log
            if frame_idx < 3:
                agent_debug_append(
                    {
                        "hypothesisId": "H3",
                        "where": "render.temporal",
                        "frameIdx": int(frame_idx),
                        "smoothingRan": True,
                        "preBlank": pre_blank,
                        "postBlank": post_blank,
                        "blankDropped": bool(post_blank < pre_blank),
                        "logitsSame": bool(
                            pre_logits.shape == smoothed_logits.shape
                            and np.array_equal(pre_logits, smoothed_logits)
                        ),
                    }
                )
            # endregion
            artifacts.ascii_frame = pipeline.char_mapper.map_logits(
                smoothed_logits,
                grid_shape=(artifacts.ascii_frame.height, artifacts.ascii_frame.width),
            )
        ascii_frames.append(artifacts.ascii_frame)

    if mode == "terminal":
        play_terminal_frames(ascii_frames, fps=config.render.fps)
        typer.echo(f"Rendered {len(ascii_frames)} frames in terminal mode")
        typer.echo(_format_runtime_summary(pipeline))
        return
    if mode == "text":
        output_path = target if target.suffix else target / "ascii.txt"
        export_ascii_frames(ascii_frames, output_path)
        typer.echo(f"Wrote ASCII text output to {output_path}")
        typer.echo(_format_runtime_summary(pipeline))
        return
    if mode == "video":
        output_path = target if target.suffix else target / "ascii.mp4"
        write_ascii_video(
            ascii_frames,
            output_path=output_path,
            fps=config.render.fps,
            overlay_mode=config.render.overlay_mode,
            source_frames=source_frames if config.render.overlay_mode != "ascii_only" else None,
            cell_size=(config.training.cell_width, config.training.cell_height),
        )
        typer.echo(f"Wrote ASCII video output to {output_path}")
        typer.echo(_format_runtime_summary(pipeline))
        return
    raise typer.BadParameter(f"Unsupported mode: {mode}")
