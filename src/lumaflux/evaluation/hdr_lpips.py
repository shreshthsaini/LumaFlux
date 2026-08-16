"""LPIPS computed on PU21-encoded HDR frames."""

from __future__ import annotations

from pathlib import Path

import torch

from ..color.spaces import PU21_PEAK, pu21_encode
from ..color.transfer import pq_eotf_nits
from .backends import EvalBackendUnavailable

_METRIC = "hdr_lpips"
_lpips_models: dict[str, torch.nn.Module] = {}


def _get_model(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key in _lpips_models:
        return _lpips_models[key]
    try:
        import lpips
        from torchvision.models import AlexNet_Weights
    except (ImportError, RuntimeError) as exc:
        raise EvalBackendUnavailable(_METRIC, f"lpips import failed: {exc}") from exc

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(AlexNet_Weights.DEFAULT.url).name
    if not checkpoint.is_file():
        raise EvalBackendUnavailable(
            _METRIC,
            f"pretrained AlexNet weights are not cached at {checkpoint}",
        )
    try:
        model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except (OSError, RuntimeError, ValueError) as exc:
        raise EvalBackendUnavailable(_METRIC, f"LPIPS initialization failed: {exc}") from exc
    _lpips_models[key] = model
    return model


def hdr_lpips(pred_pq: torch.Tensor, target_pq: torch.Tensor) -> torch.Tensor:
    """Return HDR-LPIPS for matching PQ/BT.2020 tensors."""
    if pred_pq.shape != target_pq.shape:
        raise ValueError("HDR-LPIPS inputs must have matching shapes")
    if pred_pq.ndim == 3:
        pred_pq, target_pq = pred_pq.unsqueeze(0), target_pq.unsqueeze(0)
    if pred_pq.ndim != 4 or pred_pq.shape[1] != 3:
        raise ValueError("HDR-LPIPS inputs must have shape (3, H, W) or (B, 3, H, W)")
    model = _get_model(pred_pq.device)
    a = pu21_encode(pq_eotf_nits(pred_pq)) / PU21_PEAK * 2.0 - 1.0
    b = pu21_encode(pq_eotf_nits(target_pq)) / PU21_PEAK * 2.0 - 1.0
    with torch.no_grad():
        return model(a, b).mean()


def probe_hdr_lpips() -> None:
    """Execute a small end-to-end backend probe."""
    frame = torch.full((3, 64, 64), 0.5)
    hdr_lpips(frame, frame)
