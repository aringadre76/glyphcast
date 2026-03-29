"""Model management commands."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

import typer

from glyphcast.constants import EDGE_MODEL_DIR

MODEL_SOURCES = {
    "dexined": {
        "url": "http://cmp.felk.cvut.cz/~mishkdmy/models/DexiNed_BIPED_10.pth",
        "filename": "dexined.pt",
    },
    "hed": {
        "url": "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth",
        "filename": "hed.pth",
    },
}


def _download_model(url: str, destination: Path) -> str:
    with urlopen(url) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return sha256(payload).hexdigest()


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
    manifest: dict[str, object] = {"models": {}}
    for name in names:
        source = MODEL_SOURCES.get(name)
        if source is None:
            raise typer.BadParameter(f"Unsupported edge model: {name}")
        output_path = resolved / str(source["filename"])
        checksum = _download_model(str(source["url"]), output_path)
        manifest["models"][name] = {
            "url": source["url"],
            "path": output_path.name,
            "sha256": checksum,
        }
    (resolved / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(f"Prepared {resolved} for edge model downloads")
