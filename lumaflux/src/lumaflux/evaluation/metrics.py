"""HDR quality metrics in perceptually-uniform PU21 space (Sec. 6.1).

PSNR / PSNR(Y) / SSIM are computed on PU21-encoded absolute luminance,
following the paper. Inputs are PQ/BT.2020 signals in [0,1] unless noted.
"""

from __future__ import annotations

import torch

from ..color.spaces import PU21_PEAK, luma_2020, pu21_encode
from ..color.transfer import pq_eotf_nits


def _pu21(x_pq: torch.Tensor) -> torch.Tensor:
    return pu21_encode(pq_eotf_nits(x_pq))


def pu21_psnr(pred_pq: torch.Tensor, target_pq: torch.Tensor) -> torch.Tensor:
    mse = (_pu21(pred_pq) - _pu21(target_pq)).pow(2).mean()
    return 10.0 * torch.log10(PU21_PEAK**2 / mse.clamp(min=1e-12))


def pu21_psnr_y(pred_pq: torch.Tensor, target_pq: torch.Tensor) -> torch.Tensor:
    yp = pu21_encode(luma_2020(pq_eotf_nits(pred_pq)))
    yt = pu21_encode(luma_2020(pq_eotf_nits(target_pq)))
    mse = (yp - yt).pow(2).mean()
    return 10.0 * torch.log10(PU21_PEAK**2 / mse.clamp(min=1e-12))


def pu21_ssim(pred_pq: torch.Tensor, target_pq: torch.Tensor) -> torch.Tensor:
    """SSIM on PU21-encoded luminance (gaussian 11x11, standard constants)."""
    import torch.nn.functional as F

    yp = pu21_encode(luma_2020(pq_eotf_nits(pred_pq))) / PU21_PEAK
    yt = pu21_encode(luma_2020(pq_eotf_nits(target_pq))) / PU21_PEAK
    if yp.dim() == 3:
        yp, yt = yp.unsqueeze(0), yt.unsqueeze(0)

    sigma, ksize = 1.5, 11
    coords = torch.arange(ksize, dtype=yp.dtype, device=yp.device) - ksize // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).view(1, 1, 1, -1)

    def blur(x):
        x = F.conv2d(x, g, padding=(0, ksize // 2))
        return F.conv2d(x, g.transpose(-1, -2), padding=(ksize // 2, 0))

    c1, c2 = 0.01**2, 0.03**2
    mu_p, mu_t = blur(yp), blur(yt)
    var_p = blur(yp * yp) - mu_p**2
    var_t = blur(yt * yt) - mu_t**2
    cov = blur(yp * yt) - mu_p * mu_t
    ssim_map = ((2 * mu_p * mu_t + c1) * (2 * cov + c2)) / (
        (mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)
    )
    return ssim_map.mean()
