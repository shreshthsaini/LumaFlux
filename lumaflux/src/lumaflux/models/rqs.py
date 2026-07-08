"""Rational-Quadratic Spline (RQS) tone-field decoder, Sec. 4.6, Eq. 16-17.

A lightweight head predicts monotone spline parameters (knot widths xi,
heights eta, derivatives s) from the final latent z_T; the spline expands
the VAE-decoded luma while Conv1x1 layers refine chroma:

    Y_hat       = RQS(Y_out; xi, eta, s)
    [U_hat,V_hat] = Conv1x1([U, V])
    x_hdr       = M_YUV->RGB([Y_hat, U_hat, V_hat])   (PQ, BT.2020)

The spline follows Durkan et al. (2019): strictly monotone, differentiable,
invertible on [0,1] with bounded derivatives. Parameters are predicted
per-sample (a global tone field) and initialize to the identity map.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..color.spaces import rgb_to_ycbcr2020, ycbcr2020_to_rgb

_MIN_BIN = 1e-3
_MIN_DERIV = 1e-3


def rqs_apply(
    y: torch.Tensor,        # (B, ...) values in [0, 1]
    widths: torch.Tensor,   # (B, K) unnormalized
    heights: torch.Tensor,  # (B, K) unnormalized
    derivs: torch.Tensor,   # (B, K+1) unnormalized
) -> torch.Tensor:
    """Monotone rational-quadratic spline on [0,1] -> [0,1] (Durkan et al.)."""
    b, k = widths.shape
    w = _MIN_BIN + (1 - _MIN_BIN * k) * F.softmax(widths, dim=-1)
    h = _MIN_BIN + (1 - _MIN_BIN * k) * F.softmax(heights, dim=-1)
    d = _MIN_DERIV + F.softplus(derivs)

    cum_w = F.pad(w.cumsum(-1), (1, 0))  # (B, K+1) knot x-positions
    cum_h = F.pad(h.cumsum(-1), (1, 0))  # (B, K+1) knot y-positions
    cum_w[..., -1] = 1.0
    cum_h[..., -1] = 1.0

    flat = y.clamp(0.0, 1.0).reshape(b, -1)
    idx = (torch.searchsorted(cum_w[..., 1:-1].contiguous(), flat.contiguous())).clamp(max=k - 1)

    in_w = w.gather(-1, idx)
    in_left = cum_w.gather(-1, idx)
    in_h = h.gather(-1, idx)
    out_left = cum_h.gather(-1, idx)
    d_left = d.gather(-1, idx)
    d_right = d.gather(-1, (idx + 1).clamp(max=k))
    slope = in_h / in_w

    theta = ((flat - in_left) / in_w).clamp(0.0, 1.0)
    t1m = theta * (1 - theta)
    num = in_h * (slope * theta.pow(2) + d_left * t1m)
    den = slope + (d_left + d_right - 2 * slope) * t1m
    out = out_left + num / den.clamp(min=1e-8)
    return out.reshape(y.shape).clamp(0.0, 1.0)


class RQSToneFieldDecoder(nn.Module):
    """Predicts a per-sample tone spline from z_T and applies it to the
    YUV(BT.2020) representation of the VAE output."""

    def __init__(self, latent_channels: int, num_knots: int = 8, hidden: int = 128,
                 strength: float = 1.0) -> None:
        super().__init__()
        self.num_knots = num_knots
        self.head = nn.Sequential(
            nn.Linear(latent_channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * num_knots + 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        with torch.no_grad():
            # softplus(bias) + MIN_DERIV == 1 -> exact identity spline at init.
            ident = math.log(math.expm1(1.0 - _MIN_DERIV))
            self.head[-1].bias[2 * num_knots :].fill_(ident)
        self.chroma = nn.Conv2d(2, 2, kernel_size=1)
        with torch.no_grad():
            self.chroma.weight.copy_(torch.eye(2).view(2, 2, 1, 1))
            self.chroma.bias.zero_()
        # User-facing expansion strength (paper: broader-impact "strength" knob).
        self.strength = strength

    def spline_params(self, z_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z_t: (B, C, h, w) or (B, N, C) final latent -> (xi, eta, s)."""
        if z_t.dim() == 4:
            pooled = z_t.float().mean(dim=(-2, -1))
        else:
            pooled = z_t.float().mean(dim=1)
        params = self.head(pooled.to(next(self.parameters()).dtype))
        k = self.num_knots
        widths = params[:, :k]
        heights = params[:, k : 2 * k]
        derivs = params[:, 2 * k :]
        return widths, heights, derivs

    def forward(self, x_out: torch.Tensor, z_t: torch.Tensor) -> dict[str, torch.Tensor]:
        """x_out: (B,3,H,W) VAE-decoded frame in [0,1] interpreted as
        PQ/BT.2020 R'G'B'. Returns the tone-expanded frame and spline params
        (for the smoothness regularizer)."""
        widths, heights, derivs = self.spline_params(z_t)
        yuv = rgb_to_ycbcr2020(x_out.clamp(0.0, 1.0))
        y, u, v = yuv[:, :1], yuv[:, 1:2], yuv[:, 2:3]
        y_hat = rqs_apply(y, widths, heights, derivs)
        if self.strength != 1.0:
            y_hat = y + self.strength * (y_hat - y)
        uv_hat = self.chroma(torch.cat([u, v], dim=1))
        out = ycbcr2020_to_rgb(torch.cat([y_hat, uv_hat], dim=1)).clamp(0.0, 1.0)
        return {"hdr": out, "widths": widths, "heights": heights, "derivs": derivs}


def spline_smoothness_loss(widths: torch.Tensor, heights: torch.Tensor,
                           derivs: torch.Tensor) -> torch.Tensor:
    """Penalize adjacent knot slope changes (L_spline-smooth in Eq. 18)."""
    k = widths.shape[-1]
    w = _MIN_BIN + (1 - _MIN_BIN * k) * F.softmax(widths, dim=-1)
    h = _MIN_BIN + (1 - _MIN_BIN * k) * F.softmax(heights, dim=-1)
    slopes = h / w
    d = _MIN_DERIV + F.softplus(derivs)
    return (slopes[:, 1:] - slopes[:, :-1]).pow(2).mean() + (d[:, 1:] - d[:, :-1]).pow(2).mean()
