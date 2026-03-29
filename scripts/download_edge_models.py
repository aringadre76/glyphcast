"""Download pretrained edge checkpoints.

The initial implementation creates the expected local artifact paths and prints
the recommended upstream sources for manual weight download.
"""

from __future__ import annotations

from pathlib import Path

from glyphcast.constants import EDGE_MODEL_DIR


MODEL_SOURCES = {
    "dexined": "https://github.com/xavysp/DexiNed",
    "hed": "https://github.com/sniklaus/pytorch-hed",
}


def prepare_download_directory(destination: Path = EDGE_MODEL_DIR) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def main() -> None:
    destination = prepare_download_directory()
    for name, url in MODEL_SOURCES.items():
        print(f"{name}: download weights from {url} into {destination}")


if __name__ == "__main__":
    main()
