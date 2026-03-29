from pathlib import Path

import torch

from glyphcast.training.glyph_dataset import build_synthetic_glyph_dataset
from glyphcast.training.train_char_model import train_char_cnn
from glyphcast.training.eval_char_model import evaluate_char_cnn
from glyphcast.models.char_cnn import AsciiCharCNN


def test_train_char_cnn_saves_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint_path = train_char_cnn(
        " .",
        epochs=1,
        destination=tmp_path / "char_cnn.pt",
        device="cpu",
        cell_size=(4, 6),
        fonts=["DejaVuSansMono.ttf"],
    )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert payload["charset"] == [" ", "."]
    assert payload["cell_size"] == [4, 6]
    assert payload["in_channels"] == 2
    assert payload["device"] == "cpu"
    assert payload["fonts"] == ["DejaVuSansMono.ttf"]


def test_evaluate_char_cnn_accepts_device_argument() -> None:
    dataset = build_synthetic_glyph_dataset(" .", cell_size=(4, 6))
    model = AsciiCharCNN(num_classes=2, in_channels=2)

    accuracy = evaluate_char_cnn(model, dataset, device="cpu")

    assert 0.0 <= accuracy <= 1.0
