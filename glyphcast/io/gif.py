"""GIF decoding helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


def read_gif_frames(path: Path) -> list[np.ndarray]:
    image = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(image):
        frames.append(np.asarray(frame.convert("RGB"))[:, :, ::-1].copy())
    return frames
