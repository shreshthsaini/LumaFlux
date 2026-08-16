"""HDR Residual Coupler, Sec. 4.5, Eq. 15.

    z_out = z_res + lambda(t, l) * ( W_p T_phys + W_c C_perc(T_perc) )

W_p / W_c are 1x1 (pointwise) projections into the token dimension.
Perceptual tokens are resampled from the SigLIP patch grid onto the latent
token grid so the fused residual is spatially aligned. The projections are
zero-initialized: the coupler is exactly inactive at step 0 and the learned
schedule lambda(t,l) decays low-frequency corrections as t -> 0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_tokens(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
    """(B, Np, d) tokens on a square-ish grid -> (B, N, d) on ``grid_hw``."""
    b, n, d = tokens.shape
    side = int(math.sqrt(n))
    if side * side != n:  # non-square token sets fall back to mean broadcast
        pooled = tokens.mean(dim=1, keepdim=True)
        return pooled.expand(b, grid_hw[0] * grid_hw[1], d)
    grid = tokens.transpose(1, 2).reshape(b, d, side, side)
    grid = F.interpolate(grid, size=grid_hw, mode="bilinear", align_corners=False)
    return grid.flatten(2).transpose(1, 2)


def _bottleneck(in_dim: int, out_dim: int, rank: int) -> nn.Sequential:
    projection = nn.Sequential(
        nn.Linear(in_dim, rank, bias=False),
        nn.SiLU(),
        nn.Linear(rank, out_dim, bias=False),
    )
    nn.init.zeros_(projection[-1].weight)
    return projection


class HDRResidualCoupler(nn.Module):
    def __init__(self, phys_channels: int, model_dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.w_p = _bottleneck(phys_channels, model_dim, bottleneck)
        self.w_c = _bottleneck(model_dim, model_dim, bottleneck)

    def forward(
        self,
        z_res: torch.Tensor,        # (B, N_img, d) block residual output
        phys_tokens: torch.Tensor,  # (B, N_img, C_phys)
        perc_tokens: torch.Tensor,  # (B, Np, d) connected SigLIP tokens
        lam: torch.Tensor,          # (B,) schedule lambda(t, l)
        grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        perc = resample_tokens(perc_tokens, grid_hw)
        fused = self.w_p(phys_tokens) + self.w_c(perc)
        return z_res + lam.view(-1, 1, 1) * fused
