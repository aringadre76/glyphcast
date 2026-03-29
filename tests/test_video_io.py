from pathlib import Path

import cv2
import numpy as np

from glyphcast.io.video import VideoReader


def test_video_reader_reports_metadata_for_mp4(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12.0,
        (16, 12),
    )
    for _ in range(3):
        writer.write(np.zeros((12, 16, 3), dtype=np.uint8))
    writer.release()

    metadata = VideoReader(video_path).metadata()

    assert metadata.width == 16
    assert metadata.height == 12
    assert metadata.frame_count == 3
    assert metadata.fps > 0
