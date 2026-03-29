import numpy as np
import torch

from glyphcast.pipeline.edge_detector import EdgeDetector
from glyphcast.pipeline.postprocess import postprocess_edge_probabilities
from glyphcast.pipeline.preprocess import prepare_grayscale_frame
from glyphcast.models.edge_backends import TorchCheckpointEdgeBackend


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


def test_edge_detector_builds_edge_backend_once(monkeypatch) -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def infer(self, grayscale_frame: np.ndarray) -> np.ndarray:
            return np.ones_like(grayscale_frame, dtype=np.float32)

    def fake_build_edge_backend(name: str, checkpoint_path=None, **_kwargs):
        calls.append((name, checkpoint_path))
        return FakeBackend()

    monkeypatch.setattr("glyphcast.pipeline.edge_detector.build_edge_backend", fake_build_edge_backend)
    detector = EdgeDetector(backend="sobel")

    detector.detect(frame)
    detector.detect(frame)

    assert calls == [("sobel", None)]


def test_torch_checkpoint_edge_backend_runs_torchscript_checkpoint(tmp_path) -> None:
    class MeanEdgeModel(torch.nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return batch.mean(dim=1, keepdim=True)

    checkpoint_path = tmp_path / "edge_model.ts"
    traced = torch.jit.trace(MeanEdgeModel(), torch.rand(1, 3, 4, 4))
    traced.save(str(checkpoint_path))

    backend = TorchCheckpointEdgeBackend(
        name="hed",
        checkpoint_path=checkpoint_path,
        device="cpu",
        fallback_device="cpu",
    )
    grayscale = np.linspace(0.0, 1.0, num=16, dtype=np.float32).reshape(4, 4)

    probabilities = backend.infer(grayscale)

    assert probabilities.shape == (4, 4)
    assert np.allclose(probabilities, grayscale, atol=1e-5)
