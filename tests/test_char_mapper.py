import numpy as np
import torch

from glyphcast.models.char_cnn import AsciiCharCNN
from glyphcast.pipeline.char_mapper import CharMapper
from glyphcast.training.glyph_dataset import build_synthetic_glyph_dataset


def test_synthetic_glyph_dataset_matches_charset_length() -> None:
    dataset = build_synthetic_glyph_dataset(" .#", cell_size=(8, 12))

    assert len(dataset.labels) == 3
    assert dataset.tiles.shape == (3, 2, 12, 8)
    assert dataset.charset == [" ", ".", "#"]


def test_char_cnn_produces_expected_logit_shape() -> None:
    model = AsciiCharCNN(num_classes=4, in_channels=2)
    batch = torch.randn(5, 2, 12, 8)

    logits = model(batch)

    assert tuple(logits.shape) == (5, 4)


def test_char_mapper_decodes_top_logits_to_ascii_frame() -> None:
    mapper = CharMapper(charset=" .#")
    logits = np.array(
        [
            [5.0, 0.1, 0.2],
            [0.1, 5.0, 0.2],
            [0.1, 0.2, 5.0],
            [0.1, 5.0, 0.2],
        ],
        dtype=np.float32,
    )

    frame = mapper.map_logits(logits, grid_shape=(2, 2))

    assert frame.width == 2
    assert frame.height == 2
    assert frame.as_text() == " .\n#."


def test_char_mapper_cnn_mode_scores_tiles_from_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "char_cnn.pt"
    model = AsciiCharCNN(num_classes=3, in_channels=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
    torch.save(
        {"state_dict": model.state_dict(), "charset": [" ", ".", "#"]},
        checkpoint_path,
    )

    mapper = CharMapper(
        charset=" .#",
        mode="cnn",
        model_path=checkpoint_path,
        device="cpu",
        fallback_device="cpu",
        batch_size=1,
    )
    tiles = np.zeros((2, 2, 12, 8), dtype=np.float32)

    logits = mapper.score_tiles(tiles)

    assert logits.shape == (2, 3)
    assert np.argmax(logits, axis=1).tolist() == [2, 2]


def test_char_mapper_cnn_plus_template_uses_template_signal_to_break_ties(tmp_path) -> None:
    checkpoint_path = tmp_path / "char_cnn.pt"
    model = AsciiCharCNN(num_classes=3, in_channels=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    torch.save(
        {"state_dict": model.state_dict(), "charset": [" ", ".", "#"]},
        checkpoint_path,
    )

    mapper = CharMapper(
        charset=" .#",
        mode="cnn_plus_template",
        model_path=checkpoint_path,
        device="cpu",
        fallback_device="cpu",
        batch_size=4,
    )
    dataset = build_synthetic_glyph_dataset(" .#", cell_size=(8, 12))
    logits = mapper.score_tiles(dataset.tiles[[2]])

    frame = mapper.map_logits(logits, grid_shape=(1, 1))

    assert frame.as_text() == "#"
