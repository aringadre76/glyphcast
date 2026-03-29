"""Model management commands."""

from __future__ import annotations

from pathlib import Path

import typer

from glyphcast.constants import EDGE_MODEL_DIR

MODEL_SOURCES = {
    "dexined": "https://github.com/xavysp/DexiNed",
    "hed": "https://github.com/sniklaus/pytorch-hed",
}


def download_models_command(
    edge: str = typer.Option("dexined", "--edge"),
    destination: Path = typer.Option(EDGE_MODEL_DIR, "--destination"),
) -> None:
    resolved = destination
    resolved.mkdir(parents=True, exist_ok=True)
    if edge == "all":
        names = sorted(MODEL_SOURCES)
    else:
        names = [edge]
    for name in names:
        marker = resolved / f"{name}.txt"
        marker.write_text(
            f"Download pretrained weights from {MODEL_SOURCES.get(name, 'unknown source')}\n",
            encoding="utf-8",
        )
    typer.echo(f"Prepared {resolved} for edge model downloads")
