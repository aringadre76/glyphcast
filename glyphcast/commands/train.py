"""Training commands."""

from __future__ import annotations

from pathlib import Path

import typer

from glyphcast.config import GlyphcastConfig
from glyphcast.constants import CHARSET_PRESETS
from glyphcast.training.train_char_model import train_char_cnn


def train_chars_command(
    fonts: Path | None = typer.Option(None, "--fonts"),
    charset: str | None = typer.Option(None, "--charset"),
    preset: str = typer.Option("default", "--preset"),
) -> None:
    config = GlyphcastConfig.from_preset(preset)
    selected_charset_name = charset or config.runtime.charset
    selected_charset = CHARSET_PRESETS.get(selected_charset_name, selected_charset_name)
    selected_fonts = [str(fonts)] if fonts is not None else list(config.training.fonts or [])
    output_path = train_char_cnn(
        selected_charset,
        epochs=5,
        device=config.runtime.device,
        mixed_precision=config.runtime.mixed_precision,
        cell_size=(config.training.cell_width, config.training.cell_height),
        fonts=selected_fonts,
    )
    typer.echo(
        f"Trained character model with fonts={selected_fonts}, charset={selected_charset_name}, "
        f"cell={config.training.cell_width}x{config.training.cell_height}, output={output_path}"
    )
