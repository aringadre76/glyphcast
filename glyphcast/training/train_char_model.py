"""Training entrypoint for the CNN character model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from glyphcast.constants import CHAR_MODEL_DIR
from glyphcast.models.char_cnn import AsciiCharCNN
from glyphcast.models.edge_backends import resolve_torch_device
from glyphcast.training.glyph_dataset import build_synthetic_glyph_dataset


def train_char_cnn(
    charset: str,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    destination: Path | None = None,
    *,
    device: str = "cpu",
    mixed_precision: bool = False,
    cell_size: tuple[int, int] = (8, 12),
    fonts: list[str] | None = None,
    batch_size: int | None = None,
) -> Path:
    dataset = build_synthetic_glyph_dataset(charset, cell_size=cell_size, augment=True)
    resolved_device = resolve_torch_device(device)
    model = AsciiCharCNN(num_classes=len(dataset.charset))
    model.to(resolved_device)
    loader = DataLoader(
        TensorDataset(torch.tensor(dataset.tiles), torch.tensor(dataset.labels)),
        batch_size=batch_size or min(32, len(dataset.labels)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    use_amp = mixed_precision and resolved_device.type == "cuda"
    for _ in range(epochs):
        for batch_tiles, batch_labels in loader:
            batch_tiles = batch_tiles.to(resolved_device)
            batch_labels = batch_labels.to(resolved_device)
            optimizer.zero_grad()
            with torch.autocast(device_type=resolved_device.type, enabled=use_amp):
                logits = model(batch_tiles)
                loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
    output_path = destination or CHAR_MODEL_DIR / "char_cnn.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "charset": dataset.charset,
            "cell_size": [cell_size[0], cell_size[1]],
            "in_channels": 2,
            "device": resolved_device.type,
            "fonts": fonts or [],
            "mixed_precision": mixed_precision,
        },
        output_path,
    )
    return output_path
