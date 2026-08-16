"""Physical luminance descriptors used to build T_phys and s_g (Sec. 4.1)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .spaces import luma_2020
from .transfer import CORPUS_PEAK_NITS, bt1886_eotf, pq_eotf_nits
from .gamut import rgb709_to_rgb2020


def sdr_to_linear2020(x_sdr: torch.Tensor) -> torch.Tensor:
    """8-bit-style SDR BT.709 signal [0,1] -> linear BT.2020 [0,1].

    The paper extracts physical cues from "input frame linear light in
    BT.2020 using PQ EOTF" when the input is already PQ; for SDR inputs we
    decode display light with BT.1886 and rotate primaries.
    """
    lin709 = bt1886_eotf(x_sdr)
    return rgb709_to_rgb2020(lin709).clamp(0.0, 1.0)


def pq_to_linear2020(
    x_pq: torch.Tensor,
    peak_nits: float = CORPUS_PEAK_NITS,
) -> torch.Tensor:
    """PQ BT.2020 signal -> linear light where 1 is ``peak_nits``."""
    if peak_nits <= 0.0:
        raise ValueError("peak_nits must be positive")
    return pq_eotf_nits(x_pq) / peak_nits


def saturation(x_lin: torch.Tensor) -> torch.Tensor:
    """Per-pixel saturation: (max - min) / (max + eps) over channels."""
    mx = x_lin.amax(dim=-3, keepdim=True)
    mn = x_lin.amin(dim=-3, keepdim=True)
    return (mx - mn) / (mx + 1e-6)


def log_grad_mag(y: torch.Tensor) -> torch.Tensor:
    """log(1 + |grad Y|) via Sobel filters. ``y`` is (..., 1, H, W)."""
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      dtype=y.dtype, device=y.device).view(1, 1, 3, 3) / 4.0
    ky = kx.transpose(-1, -2)
    shape = y.shape
    y2 = y.reshape(-1, 1, shape[-2], shape[-1])
    gx = F.conv2d(F.pad(y2, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(y2, (1, 1, 1, 1), mode="replicate"), ky)
    # eps inside the sqrt: its gradient is NaN at exactly 0 (flat regions).
    mag = (gx.pow(2) + gy.pow(2) + 1e-12).sqrt()
    return torch.log1p(mag).reshape(shape)


def physical_maps(x_lin2020: torch.Tensor) -> torch.Tensor:
    """Stack [Y, log(1+|gradY|), sat] -> (..., 3, H, W) input of Eq. 6."""
    y = luma_2020(x_lin2020)
    return torch.cat([y, log_grad_mag(y), saturation(x_lin2020)], dim=-3)


def global_stats(x_lin2020: torch.Tensor) -> torch.Tensor:
    """s_g = [mu_Y, sigma_Y, p95, p99] per sample -> (B, 4)."""
    y = luma_2020(x_lin2020)
    b = y.shape[0]
    flat = y.reshape(b, -1)
    q = torch.quantile(flat.float(), torch.tensor([0.95, 0.99], device=y.device), dim=1).to(y.dtype)
    return torch.stack([flat.mean(dim=1), flat.std(dim=1), q[0], q[1]], dim=1)


def spectral_bands(y: torch.Tensor, num_bands: int = 8) -> torch.Tensor:
    """K-band radial energy pooling of the luminance spectrum (Sec. 4.1).

    Single rfft2 pass; bands are equal-width annuli in normalized radial
    frequency. ``y`` is (B, 1, H, W); returns (B, K) log-energies.
    """
    b, _, h, w = y.shape
    spec = torch.fft.rfft2(y.float(), norm="ortho").abs().pow(2)  # (B,1,H,W//2+1)
    fy = torch.fft.fftfreq(h, device=y.device).abs()
    fx = torch.fft.rfftfreq(w, device=y.device)
    rad = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)  # (H, W//2+1)
    rad = (rad / rad.max().clamp(min=1e-8)).clamp(max=1.0 - 1e-6)
    idx = (rad * num_bands).long()  # (H, W//2+1)
    flat_spec = spec.reshape(b, -1)
    flat_idx = idx.reshape(-1).expand(b, -1)
    bands = torch.zeros(b, num_bands, device=y.device, dtype=flat_spec.dtype)
    bands.scatter_add_(1, flat_idx, flat_spec)
    counts = torch.zeros(num_bands, device=y.device, dtype=flat_spec.dtype)
    counts.scatter_add_(0, idx.reshape(-1), torch.ones_like(idx.reshape(-1), dtype=flat_spec.dtype))
    bands = bands / counts.clamp(min=1.0)
    return torch.log1p(bands).to(y.dtype)
