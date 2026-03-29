"""Training commands."""

from __future__ import annotations

from pathlib import Path

import typer

from glyphcast.config import GlyphcastConfig
from glyphcast.constants import CHARSET_PRESETS
from glyphcast.training.train_char_model import train_char_cnn


def train_chars_command(
    fonts: Path = typer.Option(Path("fonts"), "--fonts"),
    charset: str = typer.Option("balanced", "--charset"),
    preset: str = typer.Option("default", "--preset"),
) -> None:
    config = GlyphcastConfig.from_preset(preset)
    selected_charset = CHARSET_PRESETS.get(charset, charset)
    output_path = train_char_cnn(selected_charset, epochs=5)
    typer.echo(
        f"Trained character model with fonts={fonts}, charset={charset}, "
        f"cell={config.training.cell_width}x{config.training.cell_height}, output={output_path}"
    )
