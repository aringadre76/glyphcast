import numpy as np

from glyphcast.pipeline.edge_detector import EdgeDetector
from glyphcast.pipeline.postprocess import postprocess_edge_probabilities
from glyphcast.pipeline.preprocess import prepare_grayscale_frame


def test_prepare_grayscale_frame_normalizes_to_unit_range() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, 2:, :] = 255

    grayscale = prepare_grayscale_frame(frame)

    assert grayscale.shape == (4, 4)
    assert float(grayscale.min()) == 0.0
    assert float(grayscale.max()) == 1.0


def test_postprocess_edge_probabilities_applies_threshold() -> None:
    probabilities = np.array(
        [[0.1, 0.5, 0.9], [0.2, 0.8, 0.4], [0.0, 0.7, 1.0]],
        dtype=np.float32,
    )

    binary = postprocess_edge_probabilities(probabilities, threshold=0.6)

    assert binary.dtype == np.float32
    assert binary.tolist() == [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ]


def test_sobel_edge_detector_returns_probability_and_binary_maps() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, 4:, :] = 255

    detector = EdgeDetector(backend="sobel", threshold=0.2)
    edge_maps = detector.detect(frame)

    assert edge_maps.probability.shape == (8, 8)
    assert edge_maps.binary.shape == (8, 8)
    assert float(edge_maps.probability.max()) > 0.0
    assert float(edge_maps.binary.sum()) > 0.0
