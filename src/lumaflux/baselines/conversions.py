"""Shared output conversions for relative-radiance HDR baselines."""

from __future__ import annotations

import torch

from ..color.gamut import rgb709_to_rgb2020
from ..color.spaces import luma_2020
from ..color.transfer import pq_oetf_nits

M709_LUMA = (0.2126, 0.7152, 0.0722)


def luma_709(rgb: torch.Tensor) -> torch.Tensor:
    """Return linear-light BT.709 luminance while preserving the channel axis."""
    if rgb.shape[-3] != 3:
        raise ValueError(f"expected three RGB channels, got shape {tuple(rgb.shape)}")
    weights = rgb.new_tensor(M709_LUMA)
    shape = [1] * rgb.ndim
    shape[-3] = 3
    return (rgb * weights.view(shape)).sum(dim=-3, keepdim=True)


def anchor_relative_luminance(
    relative_rgb709: torch.Tensor,
    *,
    percentile: float = 99.5,
    target_nits: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor relative linear BT.709 radiance to an absolute mastering scale.

    The Luma-Eval baseline protocol decision is shared by LEDiff and X2HDR:
    scale each frame so its 99.5th-percentile BT.709 luminance is 1,000 cd/m2,
    then clip linear RGB to the 1,000-nit mastering range. The returned scalar
    is the applied multiplier. Keeping this rule in one function prevents the
    two relative-luminance methods from receiving different calibration.
    """
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if target_nits <= 0.0:
        raise ValueError("target_nits must be positive")
    if relative_rgb709.shape[-3] != 3:
        raise ValueError(
            f"expected three RGB channels, got shape {tuple(relative_rgb709.shape)}"
        )
    relative = relative_rgb709.to(torch.float32).clamp(min=0.0)
    luminance = luma_709(relative)
    anchor = torch.quantile(luminance.reshape(-1), percentile / 100.0)
    if not torch.isfinite(anchor) or anchor <= 0.0:
        scale = anchor.new_tensor(1.0)
    else:
        scale = anchor.new_tensor(target_nits) / anchor
    return (relative * scale).clamp(0.0, target_nits), scale


def relative_linear_709_to_pq2020(
    relative_rgb709: torch.Tensor,
    *,
    percentile: float = 99.5,
    target_nits: float = 1000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert relative linear BT.709 RGB to anchored PQ-coded BT.2020 RGB."""
    anchored_709, scale = anchor_relative_luminance(
        relative_rgb709,
        percentile=percentile,
        target_nits=target_nits,
    )
    nits_2020 = rgb709_to_rgb2020(anchored_709).clamp(0.0, target_nits)
    return pq_oetf_nits(nits_2020), scale


def linear_709_nits_to_pq2020(
    rgb709_nits: torch.Tensor,
    *,
    peak_nits: float = 1000.0,
) -> torch.Tensor:
    """Convert absolute linear BT.709 RGB in cd/m2 to PQ-coded BT.2020 RGB."""
    if peak_nits <= 0.0:
        raise ValueError("peak_nits must be positive")
    nits_2020 = rgb709_to_rgb2020(rgb709_nits.clamp(0.0, peak_nits))
    return pq_oetf_nits(nits_2020.clamp(0.0, peak_nits))


def pq2020_luminance_nits(rgb_pq2020: torch.Tensor) -> torch.Tensor:
    """Convenience helper used by protocol tests."""
    from ..color.transfer import pq_eotf_nits

    return luma_2020(pq_eotf_nits(rgb_pq2020))
