"""Training entrypoint for the CNN character model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from glyphcast.constants import CHAR_MODEL_DIR
from glyphcast.models.char_cnn import AsciiCharCNN
from glyphcast.training.glyph_dataset import build_synthetic_glyph_dataset


def train_char_cnn(
    charset: str,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    destination: Path | None = None,
) -> Path:
    dataset = build_synthetic_glyph_dataset(charset, augment=True)
    model = AsciiCharCNN(num_classes=len(dataset.charset))
    loader = DataLoader(
        TensorDataset(torch.tensor(dataset.tiles), torch.tensor(dataset.labels)),
        batch_size=min(32, len(dataset.labels)),
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_tiles, batch_labels in loader:
            optimizer.zero_grad()
            logits = model(batch_tiles)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
    output_path = destination or CHAR_MODEL_DIR / "char_cnn.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "charset": dataset.charset}, output_path)
    return output_path
