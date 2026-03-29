"""Edge detector backend registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from torch import nn

try:
    from kornia.filters import DexiNed as KorniaDexiNed
except ImportError:  # pragma: no cover - exercised when edge extras are not installed.
    KorniaDexiNed = None


class EdgeBackend(Protocol):
    name: str

    def infer(self, grayscale_frame: np.ndarray) -> np.ndarray:
        """Return a probability map in the [0, 1] range."""


@dataclass(slots=True)
class SobelEdgeBackend:
    name: str = "sobel"

    def infer(self, grayscale_frame: np.ndarray) -> np.ndarray:
        grad_x = cv2.Sobel(grayscale_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(grayscale_frame, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)
        max_value = float(magnitude.max())
        if max_value <= 0.0:
            return np.zeros_like(magnitude, dtype=np.float32)
        return (magnitude / max_value).astype(np.float32)


class _HedDoubleConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, layer_count: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_channels
        for _ in range(layer_count):
            layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = output_channels
        self.features = nn.Sequential(*layers)
        self.projection = nn.Conv2d(output_channels, 1, kernel_size=1)

    def forward(
        self,
        batch: torch.Tensor,
        *,
        down_sampling: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = batch
        if down_sampling:
            features = nn.functional.max_pool2d(features, kernel_size=2, stride=2)
        features = self.features(features)
        return features, self.projection(features)


class HedEdgeModel(nn.Module):
    """Compact HED-style model compatible with ControlNet HED checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = _HedDoubleConvBlock(input_channels=3, output_channels=64, layer_count=2)
        self.block2 = _HedDoubleConvBlock(input_channels=64, output_channels=128, layer_count=2)
        self.block3 = _HedDoubleConvBlock(input_channels=128, output_channels=256, layer_count=3)
        self.block4 = _HedDoubleConvBlock(input_channels=256, output_channels=512, layer_count=3)
        self.block5 = _HedDoubleConvBlock(input_channels=512, output_channels=512, layer_count=3)

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = batch - self.norm
        features, projection1 = self.block1(features)
        features, projection2 = self.block2(features, down_sampling=True)
        features, projection3 = self.block3(features, down_sampling=True)
        features, projection4 = self.block4(features, down_sampling=True)
        _, projection5 = self.block5(features, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


def resolve_torch_device(requested_device: str, fallback_device: str = "cpu") -> torch.device:
    normalized = requested_device.lower()
    if normalized.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(normalized)
        return torch.device(fallback_device)
    return torch.device(normalized)


def _normalize_probability_map(probability: np.ndarray) -> np.ndarray:
    minimum = float(probability.min())
    maximum = float(probability.max())
    if maximum - minimum <= 1e-8:
        return np.zeros_like(probability, dtype=np.float32)
    scaled = (probability - minimum) / (maximum - minimum)
    return scaled.astype(np.float32)


@dataclass(slots=True)
class TorchCheckpointEdgeBackend:
    """Torch-backed edge detector with runtime device selection."""

    name: str
    checkpoint_path: Path | None = None
    device: str = "cpu"
    fallback_device: str = "cpu"
    mixed_precision: bool = False
    fallback_backend: str | None = None
    model: torch.jit.ScriptModule | nn.Module | None = None
    resolved_device: torch.device = field(init=False)
    sobel_fallback: SobelEdgeBackend | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.resolved_device = resolve_torch_device(self.device, self.fallback_device)
        try:
            self.model = self._load_model()
        except (FileNotFoundError, ImportError):
            if self.fallback_backend == "sobel":
                self.sobel_fallback = SobelEdgeBackend()
                self.model = None
            else:
                raise

    def infer(self, grayscale_frame: np.ndarray) -> np.ndarray:
        if self.sobel_fallback is not None:
            return self.sobel_fallback.infer(grayscale_frame)
        assert self.model is not None
        tensor = torch.from_numpy(grayscale_frame.astype(np.float32, copy=False))
        batch = tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(self.resolved_device)
        use_amp = self.mixed_precision and self.resolved_device.type == "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=self.resolved_device.type, enabled=use_amp):
                outputs = self.model(batch)
        probability = self._coerce_output(outputs, output_shape=grayscale_frame.shape)
        return _normalize_probability_map(probability)

    def _load_model(self) -> torch.jit.ScriptModule | nn.Module:
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            if self.checkpoint_path.suffix in {".ts", ".torchscript"}:
                model = torch.jit.load(
                    str(self.checkpoint_path),
                    map_location=self.resolved_device,
                )
                model.eval()
                return model
            model = self._load_state_dict_model()
            model.to(self.resolved_device)
            model.eval()
            return model
        if self.name == "dexined":
            if KorniaDexiNed is None:
                raise ImportError(
                    "DexiNed requires kornia. Install the project with the [edge] extras."
                )
            model = KorniaDexiNed(pretrained=True)
            model.to(self.resolved_device)
            model.eval()
            return model
        raise FileNotFoundError(
            f"{self.name} requires a checkpoint at {self.checkpoint_path or 'a configured path'}"
        )

    def _load_state_dict_model(self) -> nn.Module:
        assert self.checkpoint_path is not None
        if self.name == "hed":
            model = HedEdgeModel()
        elif self.name == "dexined":
            if KorniaDexiNed is None:
                raise ImportError(
                    "DexiNed requires kornia. Install the project with the [edge] extras."
                )
            model = KorniaDexiNed(pretrained=False)
        else:
            raise ValueError(f"Unsupported torch checkpoint backend: {self.name}")

        payload = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        model.load_state_dict(state_dict, strict=True)
        return model

    def _coerce_output(
        self,
        outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        *,
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        if isinstance(outputs, torch.Tensor):
            tensor = outputs
        else:
            tensors = list(outputs)
            if self.name == "hed":
                resized = [
                    nn.functional.interpolate(
                        projection,
                        size=output_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for projection in tensors
                ]
                tensor = torch.stack([torch.sigmoid(item) for item in resized], dim=0).mean(dim=0)
            else:
                tensor = tensors[-1]

        if tensor.ndim == 4:
            tensor = tensor[0, 0]
        elif tensor.ndim == 3:
            tensor = tensor[0]
        return tensor.detach().float().cpu().numpy()


def build_edge_backend(
    name: str,
    checkpoint_path: Path | None = None,
    *,
    device: str = "cpu",
    fallback_device: str = "cpu",
    mixed_precision: bool = False,
    fallback_backend: str | None = None,
) -> EdgeBackend:
    normalized = name.lower()
    if normalized == "sobel":
        return SobelEdgeBackend()
    if normalized in {"dexined", "hed"}:
        return TorchCheckpointEdgeBackend(
            name=normalized,
            checkpoint_path=checkpoint_path,
            device=device,
            fallback_device=fallback_device,
            mixed_precision=mixed_precision,
            fallback_backend=fallback_backend,
        )
    raise ValueError(f"Unsupported edge backend: {name}")
