"""Physical feature extraction (Sec. 4.1, Eq. 6).

Builds, from the SDR input frame:
  * T_phys  - Conv3x3([Y, log(1+|gradY|), sat]) spatial descriptor map,
  * g       - MLP_g([mu_Y, sigma_Y, p95, p99]) global statistics vector,
  * r       - K-band FFT spectral descriptor of the luminance.
and provides token-grid pooling aligned with Flux's packed 2x2 latent
patch grid so the descriptors can modulate attention tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..color.luminance import (
    global_stats,
    physical_maps,
    sdr_to_linear2020,
    spectral_bands,
)
from ..color.spaces import luma_2020


class PhysicalEncoder(nn.Module):
    def __init__(self, channels: int = 32, stats_dim: int = 16, num_bands: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.mlp_g = nn.Sequential(nn.Linear(4, stats_dim), nn.SiLU(), nn.Linear(stats_dim, stats_dim))
        self.num_bands = num_bands
        self.channels = channels
        self.stats_dim = stats_dim

    def forward(self, x_sdr: torch.Tensor) -> dict[str, torch.Tensor]:
        """x_sdr: (B,3,H,W) BT.709 signal in [0,1].

        Returns ``t_phys`` (B,C,H,W), ``g`` (B,stats_dim), ``r`` (B,K).
        """
        x_lin = sdr_to_linear2020(x_sdr)
        maps = physical_maps(x_lin)
        t_phys = self.conv(maps)
        g = self.mlp_g(global_stats(x_lin))
        r = spectral_bands(luma_2020(x_lin), self.num_bands)
        return {"t_phys": t_phys, "g": g, "r": r}

    @staticmethod
    def to_tokens(t_phys: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        """Pool the descriptor map to the latent token grid -> (B, N, C)."""
        pooled = F.adaptive_avg_pool2d(t_phys, grid_hw)
        return pooled.flatten(2).transpose(1, 2)
