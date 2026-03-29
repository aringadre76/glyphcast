"""Frame preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np


def prepare_grayscale_frame(frame_bgr: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return grayscale.astype(np.float32) / 255.0


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def denoise_frame(grayscale_frame: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(grayscale_frame, (kernel_size, kernel_size), sigmaX=0)
