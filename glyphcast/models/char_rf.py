"""Random Forest baseline for glyph classification."""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@dataclass(slots=True)
class RandomForestCharClassifier:
    model: RandomForestClassifier

    @classmethod
    def train(cls, tiles: np.ndarray, labels: np.ndarray) -> "RandomForestCharClassifier":
        flattened = tiles.reshape(tiles.shape[0], -1)
        model = RandomForestClassifier(n_estimators=64, random_state=7)
        model.fit(flattened, labels)
        return cls(model=model)

    def predict_logits(self, tiles: np.ndarray) -> np.ndarray:
        flattened = tiles.reshape(tiles.shape[0], -1)
        probabilities = self.model.predict_proba(flattened)
        return np.log(np.clip(probabilities, 1e-6, 1.0))

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
