"""Map tile logits to ASCII characters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from glyphcast.models.char_cnn import AsciiCharCNN
from glyphcast.models.edge_backends import resolve_torch_device
from glyphcast.pipeline.batching import iter_tile_batches
from glyphcast.types import AsciiFrame
from glyphcast.training.glyph_dataset import SyntheticGlyphDataset, build_synthetic_glyph_dataset


@dataclass(slots=True)
class CharMapper:
    charset: str
    mode: str = "template"
    model_path: Path | None = None
    device: str = "cpu"
    fallback_device: str = "cpu"
    batch_size: int = 512
    fallback_mode: str | None = None
    template_dataset: SyntheticGlyphDataset | None = field(default=None, init=False)
    model: AsciiCharCNN | None = field(default=None, init=False)
    resolved_device: torch.device = field(init=False)
    effective_mode: str = field(init=False)
    checkpoint_cell_size: tuple[int, int] | None = field(default=None, init=False)
    checkpoint_in_channels: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.resolved_device = resolve_torch_device(self.device, self.fallback_device)
        self.effective_mode = self.mode
        if self.mode in {"cnn", "cnn_plus_template"}:
            try:
                self.model = self._load_model()
            except (FileNotFoundError, ValueError):
                if self.fallback_mode == "template":
                    self.effective_mode = "template"
                    self.model = None
                else:
                    raise

    def score_tiles(self, tiles: np.ndarray) -> np.ndarray:
        if self.effective_mode == "template":
            return self._score_tiles_with_templates(tiles)
        if self.effective_mode == "cnn":
            return self._score_tiles_with_cnn(tiles)
        if self.effective_mode == "cnn_plus_template":
            cnn_logits = self._score_tiles_with_cnn(tiles)
            template_logits = self._score_tiles_with_templates(tiles)
            return cnn_logits + self._normalize_rows(template_logits)
        raise ValueError(f"Unsupported glyph mode: {self.effective_mode}")

    def map_logits(self, logits: np.ndarray, grid_shape: tuple[int, int]) -> AsciiFrame:
        indices = np.argmax(logits, axis=1)
        characters = [self.charset[index] for index in indices.tolist()]
        height, width = grid_shape
        return AsciiFrame(characters=characters, width=width, height=height)

    def _score_tiles_with_templates(self, tiles: np.ndarray) -> np.ndarray:
        templates = self._get_template_dataset(cell_size=(tiles.shape[-1], tiles.shape[-2]))
        flattened_tiles = tiles.reshape(tiles.shape[0], -1)
        flattened_templates = templates.tiles.reshape(templates.tiles.shape[0], -1)
        distances = ((flattened_tiles[:, None, :] - flattened_templates[None, :, :]) ** 2).mean(axis=2)
        return -distances.astype(np.float32)

    def _score_tiles_with_cnn(self, tiles: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise FileNotFoundError("CNN glyph mode requires a valid model checkpoint.")
        self._validate_checkpoint_compatibility(tiles)

        logits_batches = []
        with torch.no_grad():
            for batch_tiles in iter_tile_batches(tiles, batch_size=self.batch_size):
                batch = torch.from_numpy(batch_tiles.astype(np.float32, copy=False)).to(
                    self.resolved_device
                )
                logits = self.model(batch).detach().cpu().numpy().astype(np.float32)
                logits_batches.append(logits)

        if not logits_batches:
            return np.zeros((0, len(self.charset)), dtype=np.float32)
        return np.concatenate(logits_batches, axis=0)

    def _load_model(self) -> AsciiCharCNN:
        if self.model_path is None:
            raise FileNotFoundError("CNN glyph mode requires a model_path.")

        payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        checkpoint_charset = payload.get("charset", list(self.charset))
        if list(checkpoint_charset) != list(self.charset):
            raise ValueError(
                "Character model charset does not match the active render charset."
            )
        if "cell_size" in payload:
            cell_width, cell_height = payload["cell_size"]
            self.checkpoint_cell_size = (int(cell_width), int(cell_height))
        self.checkpoint_in_channels = int(payload.get("in_channels", 2))

        model = AsciiCharCNN(num_classes=len(self.charset), in_channels=2)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(self.resolved_device)
        model.eval()
        return model

    def _get_template_dataset(self, cell_size: tuple[int, int]) -> SyntheticGlyphDataset:
        if self.template_dataset is None or self.template_dataset.tiles.shape[-2:] != (
            cell_size[1],
            cell_size[0],
        ):
            self.template_dataset = build_synthetic_glyph_dataset(
                self.charset,
                cell_size=cell_size,
            )
        return self.template_dataset

    def _normalize_rows(self, logits: np.ndarray) -> np.ndarray:
        minimum = logits.min(axis=1, keepdims=True)
        maximum = logits.max(axis=1, keepdims=True)
        spans = np.maximum(maximum - minimum, 1e-6)
        return ((logits - minimum) / spans).astype(np.float32)

    def _validate_checkpoint_compatibility(self, tiles: np.ndarray) -> None:
        if self.checkpoint_in_channels is not None and tiles.shape[1] != self.checkpoint_in_channels:
            raise ValueError(
                "Character model input channels do not match the extracted tile channels."
            )
        tile_cell_size = (tiles.shape[-1], tiles.shape[-2])
        if self.checkpoint_cell_size is not None and tile_cell_size != self.checkpoint_cell_size:
            raise ValueError(
                "Character model checkpoint cell size does not match the extracted tile cell size."
            )
