"""Evaluation helpers for glyph classifiers."""

from __future__ import annotations

import numpy as np
import torch

from glyphcast.models.char_cnn import AsciiCharCNN
from glyphcast.training.glyph_dataset import SyntheticGlyphDataset


def evaluate_char_cnn(model: AsciiCharCNN, dataset: SyntheticGlyphDataset) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(dataset.tiles))
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return float(np.mean(predictions == dataset.labels))
