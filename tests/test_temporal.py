import numpy as np

from glyphcast.pipeline.temporal import TemporalSmoother


def test_temporal_smoother_blends_logits_when_motion_is_low() -> None:
    smoother = TemporalSmoother(alpha=0.5, confidence_margin=0.3, scene_cut_threshold=10.0)
    previous_logits = np.array([[2.0, 0.2], [0.2, 2.0]], dtype=np.float32)
    current_logits = np.array([[1.8, 0.5], [0.5, 1.8]], dtype=np.float32)
    edge = np.zeros((2, 2), dtype=np.float32)

    smoother.update(previous_logits, edge)
    smoothed = smoother.update(current_logits, edge)

    assert smoothed.shape == current_logits.shape
    assert smoothed[0, 0] > current_logits[0, 0]
    assert smoothed[1, 1] > current_logits[1, 1]


def test_temporal_smoother_resets_after_scene_cut() -> None:
    smoother = TemporalSmoother(alpha=0.8, confidence_margin=0.3, scene_cut_threshold=0.1)
    edge_a = np.zeros((2, 2), dtype=np.float32)
    edge_b = np.ones((2, 2), dtype=np.float32)
    logits_a = np.array([[2.0, 0.2]], dtype=np.float32)
    logits_b = np.array([[0.1, 3.0]], dtype=np.float32)

    smoother.update(logits_a, edge_a)
    smoothed = smoother.update(logits_b, edge_b)

    assert np.allclose(smoothed, logits_b)
