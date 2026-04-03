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
from glyphcast.constants import CHARSET_PRESETS


def _resolve_charset(charset: str) -> str:
    """Resolve charset preset name to actual charset string."""
    if isinstance(charset, str) and charset in CHARSET_PRESETS:
        return CHARSET_PRESETS[charset]
    return charset


# Character charset ordered by increasing visual density
DENSITY_BASED_CHARSET = " .:-=+*#%@"


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
        # Resolve charset preset names to actual charset strings
        self.charset = _resolve_charset(self.charset)
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
        tiles = self._prepare_tiles_for_scoring(tiles)
        if self.effective_mode == "template":
            return self._score_tiles_with_templates(tiles)
        if self.effective_mode == "luminance":
            return self.score_tiles_with_luminance(tiles)
        if self.effective_mode == "density":
            return self.score_tiles_with_edge_density(tiles)
        if self.effective_mode == "cnn":
            return self._score_tiles_with_cnn(tiles)
        if self.effective_mode == "cnn_plus_template":
            cnn_logits = self._score_tiles_with_cnn(tiles)
            template_logits = self._score_tiles_with_templates(tiles)
            return cnn_logits + self._normalize_rows(template_logits)
        if self.effective_mode == "edge":
            return self.score_tiles_with_edge_density(tiles)
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
        distances = ((flattened_tiles[:, None, :] - flattened_templates[None, :, :]) ** 2).mean(
            axis=2
        )
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
            raise ValueError("Character model charset does not match the active render charset.")
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

    def _prepare_tiles_for_scoring(self, tiles: np.ndarray) -> np.ndarray:
        prepared = tiles.astype(np.float32, copy=True)
        if prepared.shape[1] > 0:
            # Training data stores glyph ink as bright pixels; runtime tiles arrive as luminance.
            prepared[:, 0] = 1.0 - prepared[:, 0]
        return prepared

    def score_tiles_with_edge_density(self, tiles: np.ndarray) -> np.ndarray:
        """Score tiles based on edge density for density-based charset selection.

        Similar to the baseline approach where more edges = denser character.
        """
        # Edge tiles are in channel 1 (grayscale in channel 0)
        if tiles.shape[1] < 2:
            # No edge information, return empty scores
            return np.zeros((tiles.shape[0], len(DENSITY_BASED_CHARSET)), dtype=np.float32)

        edge_tiles = tiles[:, 1]
        # Compute edge density as mean of edge values (0-1 range)
        edge_density = edge_tiles.mean(axis=(1, 2))

        num_chars = len(DENSITY_BASED_CHARSET)
        blank_idx = DENSITY_BASED_CHARSET.index(" ")

        # Initialize scores with blank
        scores = np.zeros((tiles.shape[0], num_chars), dtype=np.float32)

        # Direct mapping from edge density to character:
        # 0.00-0.02: space (no edges - background)
        # 0.02-0.05: . (low density)
        # 0.05-0.1: : (medium-low)
        # 0.1-0.15: - (medium)
        # 0.15-0.2: + (medium-high)
        # 0.2-0.3: * (high)
        # 0.3-0.5: # (very high)
        # 0.5-0.7: % (dense)
        # 0.7-1.0: @ (very dense)

        for i in range(len(tiles)):
            density = edge_density[i]
            if density <= 0.02:
                scores[i, blank_idx] = 1.0
            elif density <= 0.05:
                scores[i, 1] = 1.0  # .
            elif density <= 0.1:
                scores[i, 2] = 1.0  # :
            elif density <= 0.15:
                scores[i, 3] = 1.0  # -
            elif density <= 0.2:
                scores[i, 4] = 1.0  # =
            elif density <= 0.3:
                scores[i, 5] = 1.0  # +
            elif density <= 0.5:
                scores[i, 6] = 1.0  # *
            elif density <= 0.7:
                scores[i, 7] = 1.0  # #
            else:
                scores[i, 9] = 1.0  # @ (skip % - use @ for highest density)

        return scores

    def score_tiles_with_luminance(self, tiles: np.ndarray) -> np.ndarray:
        """Score tiles based on grayscale luminance for character selection.

        Maps grayscale intensity to characters:
        - Dark areas (luminance < 0.45) -> @ (densest)
        - Medium-dark (0.45-0.6) -> # (dark)
        - Medium (0.6-0.75) -> % (medium)
        - Medium-bright (0.75-0.85) -> * (bright)
        - Bright (0.85-0.95) -> . (very bright)
        - Very bright (0.95-1.0) -> space (background)

        Tiles without edges always get space.
        """
        # Grayscale tiles are in channel 0, edges in channel 1
        grayscale = tiles[:, 0]  # 0 = black (dark), 1 = white (bright)
        edges = tiles[:, 1]

        luminance = grayscale.mean(axis=(1, 2))
        edge_density = edges.mean(axis=(1, 2))

        num_chars = len(DENSITY_BASED_CHARSET)
        blank_idx = DENSITY_BASED_CHARSET.index(" ")

        # Initialize scores with blank
        scores = np.zeros((tiles.shape[0], num_chars), dtype=np.float32)

        for i in range(len(tiles)):
            lum = luminance[i]
            has_edges = edge_density[i] > 0.02

            if not has_edges:
                # No edges - use space
                scores[i, blank_idx] = 1.0
            elif lum < 0.45:
                scores[i, 9] = 1.0  # @
            elif lum < 0.6:
                scores[i, 7] = 1.0  # #
            elif lum < 0.75:
                scores[i, 8] = 1.0  # %
            elif lum < 0.85:
                scores[i, 6] = 1.0  # *
            elif lum < 0.95:
                scores[i, 1] = 1.0  # .
            else:
                scores[i, blank_idx] = 1.0  # space

        return scores

    def _normalize_rows(self, logits: np.ndarray) -> np.ndarray:
        minimum = logits.min(axis=1, keepdims=True)
        maximum = logits.max(axis=1, keepdims=True)
        spans = np.maximum(maximum - minimum, 1e-6)
        return ((logits - minimum) / spans).astype(np.float32)

    def _validate_checkpoint_compatibility(self, tiles: np.ndarray) -> None:
        if (
            self.checkpoint_in_channels is not None
            and tiles.shape[1] != self.checkpoint_in_channels
        ):
            raise ValueError(
                "Character model input channels do not match the extracted tile channels."
            )
        tile_cell_size = (tiles.shape[-1], tiles.shape[-2])
        if self.checkpoint_cell_size is not None and tile_cell_size != self.checkpoint_cell_size:
            raise ValueError(
                "Character model checkpoint cell size does not match the extracted tile cell size."
            )

    def runtime_summary(self) -> dict[str, str]:
        return {
            "glyph_mode": self.effective_mode,
            "glyph_device": str(self.resolved_device),
            "char_model_path": str(self.model_path) if self.model_path is not None else "none",
        }
