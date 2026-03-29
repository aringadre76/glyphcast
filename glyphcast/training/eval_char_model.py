"""Evaluation helpers for glyph classifiers."""

from __future__ import annotations

import numpy as np
import torch

from glyphcast.models.char_cnn import AsciiCharCNN
from glyphcast.models.edge_backends import resolve_torch_device
from glyphcast.training.glyph_dataset import SyntheticGlyphDataset


def evaluate_char_cnn(
    model: AsciiCharCNN,
    dataset: SyntheticGlyphDataset,
    *,
    device: str = "cpu",
) -> float:
    resolved_device = resolve_torch_device(device)
    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(dataset.tiles, device=resolved_device))
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return float(np.mean(predictions == dataset.labels))
