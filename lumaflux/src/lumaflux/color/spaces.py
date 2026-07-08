"""Derived color spaces: Y'CbCr (BT.2020 NCL), ICtCp (BT.2100), PU21.

Unless stated otherwise, RGB tensors are channel-first ``(..., 3, H, W)``.
"""

from __future__ import annotations

import torch

from .transfer import pq_eotf_nits, pq_oetf_nits

# BT.2020 luma coefficients (also used on linear light for the physical
# luminance descriptor Y = m_2020^T x_lin in the paper).
M2020_LUMA = torch.tensor([0.2627, 0.6780, 0.0593])

# BT.709 luma coefficients.
M709_LUMA = torch.tensor([0.2126, 0.7152, 0.0722])


def luma_2020(rgb: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Weighted BT.2020 luma/luminance along the channel dim (kept, size 1)."""
    w = M2020_LUMA.to(dtype=rgb.dtype, device=rgb.device)
    shape = [1] * rgb.dim()
    shape[channel_dim] = 3
    return (rgb * w.view(shape)).sum(dim=channel_dim, keepdim=True)


def rgb_to_ycbcr2020(rgb: torch.Tensor) -> torch.Tensor:
    """Non-constant-luminance Y'CbCr from non-linear BT.2020 R'G'B' (full range)."""
    r, g, b = rgb.unbind(dim=-3)
    y = 0.2627 * r + 0.6780 * g + 0.0593 * b
    cb = (b - y) / 1.8814
    cr = (r - y) / 1.4746
    return torch.stack([y, cb, cr], dim=-3)


def ycbcr2020_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    y, cb, cr = ycbcr.unbind(dim=-3)
    r = y + 1.4746 * cr
    b = y + 1.8814 * cb
    g = (y - 0.2627 * r - 0.0593 * b) / 0.6780
    return torch.stack([r, g, b], dim=-3)


# --- ICtCp (ITU-R BT.2100-2, PQ variant) -----------------------------------

_M_RGB2020_TO_LMS = torch.tensor(
    [
        [1688.0, 2146.0, 262.0],
        [683.0, 2951.0, 462.0],
        [99.0, 309.0, 3688.0],
    ]
) / 4096.0

_M_LMSP_TO_ICTCP = torch.tensor(
    [
        [2048.0, 2048.0, 0.0],
        [6610.0, -13613.0, 7003.0],
        [17933.0, -17390.0, -543.0],
    ]
) / 4096.0


def rgb2020_linear_to_ictcp(rgb_linear_nits: torch.Tensor) -> torch.Tensor:
    """Linear BT.2020 RGB in cd/m^2 -> ICtCp (PQ-encoded, BT.2100)."""
    m1 = _M_RGB2020_TO_LMS.to(rgb_linear_nits.dtype).to(rgb_linear_nits.device)
    lms = torch.einsum("ij,...jhw->...ihw", m1, rgb_linear_nits.clamp(min=0.0))
    lms_p = pq_oetf_nits(lms)
    m2 = _M_LMSP_TO_ICTCP.to(rgb_linear_nits.dtype).to(rgb_linear_nits.device)
    return torch.einsum("ij,...jhw->...ihw", m2, lms_p)


def pq2020_to_ictcp(rgb_pq: torch.Tensor) -> torch.Tensor:
    """PQ-encoded BT.2020 RGB signal [0,1] -> ICtCp."""
    return rgb2020_linear_to_ictcp(pq_eotf_nits(rgb_pq))


# --- PU21 (Mantiuk & Azimi 2021), "banding+glare" fit -----------------------

_PU21_P = (0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.0915030316, 0.9046962098, 596.3148142)
PU21_PEAK = 595.3939  # encoding of 10,000 cd/m^2; used for PSNR peak.


def pu21_encode(nits: torch.Tensor) -> torch.Tensor:
    """Absolute luminance/linear color in cd/m^2 -> perceptually uniform units.

    Valid for 0.005..10,000 cd/m^2; output roughly [0, 600] with 100 cd/m^2
    mapping to ~256 (SDR-comparable scale).
    """
    p = _PU21_P
    y = nits.clamp(min=0.005, max=10_000.0)
    num = p[0] + p[1] * y.pow(p[3])
    den = 1.0 + p[2] * y.pow(p[3])
    return (p[6] * ((num / den).pow(p[4]) - p[5])).clamp(min=0.0)
