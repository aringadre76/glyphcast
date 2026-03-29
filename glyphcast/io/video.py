"""OpenCV-based video decoding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from glyphcast.types import FrameMetadata


@dataclass(slots=True)
class VideoReader:
    path: Path

    def metadata(self) -> FrameMetadata:
        capture = cv2.VideoCapture(str(self.path))
        try:
            return FrameMetadata(
                source_path=self.path,
                fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
                frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            )
        finally:
            capture.release()

    def frames(self) -> Iterator[np.ndarray]:
        capture = cv2.VideoCapture(str(self.path))
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                yield frame
        finally:
            capture.release()
