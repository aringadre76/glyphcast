"""Configuration loading and presets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from glyphcast.constants import DEFAULT_CONFIG_DIR


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cuda"
    mixed_precision: bool = True
    batch_size: int = 512
    edge_backend: str = "dexined"
    charset: str = "balanced"
    smoothing: bool = True


@dataclass(slots=True)
class RenderConfig:
    columns: int = 120
    fps: float = 24.0
    overlay_mode: str = "ascii_only"


@dataclass(slots=True)
class TrainingConfig:
    cell_width: int = 8
    cell_height: int = 12
    fonts: list[str] | None = None

    def __post_init__(self) -> None:
        if self.fonts is None:
            self.fonts = ["DejaVuSansMono.ttf"]


@dataclass(slots=True)
class GlyphcastConfig:
    runtime: RuntimeConfig
    render: RenderConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GlyphcastConfig":
        return cls(
            runtime=RuntimeConfig(**data.get("runtime", {})),
            render=RenderConfig(**data.get("render", {})),
            training=TrainingConfig(**data.get("training", {})),
        )

    @classmethod
    def from_preset(cls, preset: str) -> "GlyphcastConfig":
        return load_config(DEFAULT_CONFIG_DIR / f"{preset}.yaml")


def load_config(path: Path) -> GlyphcastConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GlyphcastConfig.from_dict(payload)
