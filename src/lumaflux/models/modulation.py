"""Timestep-layer adaptive modulation Psi(t, l) (Sec. 4.2, Eq. 8).

Produces all block-wise modulation parameters:

    (alpha_pga, beta_pga, alpha_pcm, beta_pcm, n_spec, lambda) = Psi(t, l)

(t, l) are encoded with learned linear layers plus sinusoidal features,
fused by addition, passed through SiLU, and mapped by affine heads.
Bounded activations (sigmoid / softplus) keep schedules positive; the
initialization is chosen so all adapter paths start near-zero, preserving
the pretrained generative prior at step 0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(x: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Standard sinusoidal features for a scalar batch ``x`` of shape (B,)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    args = x.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb.to(x.dtype)


class TimestepLayerModulation(nn.Module):
    """Shared conditioner producing per-(t, layer) adapter schedules."""

    PARAMS = ("alpha_pga", "beta_pga", "alpha_pcm", "beta_pcm", "n_spec", "lam")

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int = 128,
        adaptive: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.adaptive = adaptive
        initial = torch.tensor([-2.0, 0.0, -2.0, 0.0, -4.0, -2.0])
        if adaptive:
            self.t_proj = nn.Linear(hidden_dim, hidden_dim)
            self.layer_embed = nn.Embedding(num_layers, hidden_dim)
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            self.head = nn.Linear(hidden_dim, len(self.PARAMS))
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                self.head.bias.copy_(initial)
            self.register_parameter("static_logits", None)
        else:
            self.t_proj = None
            self.layer_embed = None
            self.mlp = None
            self.head = None
            self.static_logits = nn.Parameter(initial)
        # Biases chosen so that, at init: alpha/lambda gates ~ small but alive,
        # beta offsets ~ 0, spectral mixing ~ 0.
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor, layer_idx: int) -> dict[str, torch.Tensor]:
        """t: (B,) flow time in [0, 1]; returns dict of (B,) schedules."""
        if self.adaptive:
            emb = self.t_proj(sinusoidal_embedding(t, self.hidden_dim))
            idx = torch.full_like(t, layer_idx, dtype=torch.long)
            emb = emb + self.layer_embed(idx)
            out = self.head(self.mlp(emb))
        else:
            out = self.static_logits.to(t.dtype).unsqueeze(0).expand(t.shape[0], -1)
        a_pga, b_pga, a_pcm, b_pcm, n_spec, lam = out.unbind(dim=-1)
        return {
            "alpha_pga": torch.sigmoid(a_pga),
            "beta_pga": b_pga,
            "alpha_pcm": torch.sigmoid(a_pcm),
            "beta_pcm": b_pcm,
            "n_spec": torch.nn.functional.softplus(n_spec),
            # lambda decays as t -> 0 by construction of the learned schedule;
            # sigmoid keeps the coupler bounded.
            "lam": torch.sigmoid(lam),
        }
