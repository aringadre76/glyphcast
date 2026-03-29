"""Typer command line entrypoint."""

from __future__ import annotations

import typer

from glyphcast.commands.benchmark import benchmark_command
from glyphcast.commands.models import download_models_command
from glyphcast.commands.render import render_command
from glyphcast.commands.train import train_chars_command

app = typer.Typer(help="Convert videos and GIFs into GPU-accelerated ASCII art.")
app.command("render")(render_command)
app.command("train-chars")(train_chars_command)
app.command("benchmark")(benchmark_command)
app.command("download-models")(download_models_command)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
