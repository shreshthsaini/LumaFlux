"""Full-reference distance over HIDRO-VQA quality-aware features.

The official HIDRO-VQA release extracts normalized ResNet-50 features at the
native and half-resolution scales, then concatenates them into a 4096-D vector
per frame. Its published regressor is trained for no-reference LIVE-HDR MOS and
is not released. For paired evaluation, this module reports the mean L2
distance between those official features for reference and test frames.
Lower values are better and identical inputs produce zero.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .backends import EvalBackendUnavailable

_METRIC = "fr_hidrovqa"
_DEFAULT_WEIGHTS = Path("weights/hidrovqa/checkpoint_HDR_FT_SDR_PT.tar")
_models: dict[tuple[str, str], "HIDROFeatureExtractor"] = {}


class HIDROFeatureExtractor(nn.Module):
    """The two-scale feature path from the official HIDRO-VQA release."""

    def __init__(
        self,
        encoder: nn.Module,
        n_features: int,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.n_features = n_features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Kept for strict compatibility with the released CONTRIQUE state dict.
        self.projector = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.BatchNorm1d(n_features),
            nn.ReLU(),
            nn.Linear(n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )

    def _one_scale(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.avgpool(self.encoder(frames)).flatten(1)
        return F.normalize(features, dim=1)

    def extract_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Return concatenated native and half-resolution global features."""
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("HIDRO-VQA inputs must have shape (3, H, W) or (B, 3, H, W)")
        if min(frames.shape[-2:]) < 2:
            raise ValueError("HIDRO-VQA inputs must be at least 2x2 pixels")
        half = F.interpolate(frames, scale_factor=0.5, mode="bicubic", align_corners=False)
        return torch.cat((self._one_scale(frames), self._one_scale(half)), dim=1)


def _weights_path() -> Path:
    return Path(os.environ.get("HIDROVQA_WEIGHTS", _DEFAULT_WEIGHTS)).expanduser().resolve()


def build_hidrovqa_model(
    weights_path: str | Path | None = None,
    *,
    encoder: nn.Module | None = None,
    n_features: int = 2048,
    projection_dim: int = 128,
    load_weights: bool = True,
) -> HIDROFeatureExtractor:
    """Build the feature extractor, optionally with an injected test encoder."""
    try:
        from torchvision.models import resnet50
    except (ImportError, RuntimeError) as exc:
        raise EvalBackendUnavailable(_METRIC, f"torchvision import failed: {exc}") from exc

    if encoder is None:
        encoder = resnet50(weights=None)
    model = HIDROFeatureExtractor(encoder, n_features, projection_dim)
    if not load_weights:
        return model.eval()

    path = Path(weights_path).expanduser().resolve() if weights_path else _weights_path()
    if not path.is_file():
        raise EvalBackendUnavailable(
            _METRIC,
            f"checkpoint missing at {path}; set HIDROVQA_WEIGHTS to the official checkpoint",
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise EvalBackendUnavailable(_METRIC, f"cannot load checkpoint {path}: {exc}") from exc
    return model.eval()


def _get_model(device: torch.device) -> HIDROFeatureExtractor:
    path = _weights_path()
    key = (str(path), str(device))
    if key not in _models:
        _models[key] = build_hidrovqa_model(path).to(device)
    return _models[key]


def fr_hidrovqa(
    pred_pq: torch.Tensor,
    target_pq: torch.Tensor,
    *,
    model: HIDROFeatureExtractor | None = None,
) -> torch.Tensor:
    """Return mean HIDRO feature distance for paired PQ/BT.2020 frames."""
    if pred_pq.shape != target_pq.shape:
        raise ValueError("FR-HIDROVQA inputs must have matching shapes")
    single = pred_pq.ndim == 3
    if single:
        pred_pq, target_pq = pred_pq.unsqueeze(0), target_pq.unsqueeze(0)
    if pred_pq.ndim != 4 or pred_pq.shape[1] != 3:
        raise ValueError("FR-HIDROVQA inputs must have shape (3, H, W) or (B, 3, H, W)")
    extractor = model if model is not None else _get_model(pred_pq.device)
    extractor = extractor.to(pred_pq.device).eval()
    with torch.no_grad():
        features = extractor.extract_features(torch.cat((pred_pq.float(), target_pq.float())))
        pred_features, target_features = features.chunk(2)
        distance = torch.linalg.vector_norm(pred_features - target_features, dim=1)
    return distance.mean()


def probe_fr_hidrovqa() -> None:
    """Execute a small end-to-end checkpoint and feature probe."""
    frame = torch.full((3, 64, 64), 0.5)
    fr_hidrovqa(frame, frame)
