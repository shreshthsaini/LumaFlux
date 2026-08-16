"""Perceptual Cross-Modulation (PCM), Sec. 4.4, Eq. 13-14.

FiLM conditioning of transformer hidden states on frozen SigLIP
embeddings:

    T'_perc       = C_perc(T_perc)
    [gamma, zeta] = alpha_pcm * MLP(T'_perc) + beta_pcm
    PCM(h)        = gamma . LN(h) + zeta

``gamma``/``zeta`` are channel-wise (computed from attention-pooled
perceptual tokens), which keeps the conditioning scale-invariant and
stable across brightness levels and diffusion timesteps. We apply the
FiLM term as a zero-initialized *residual* (h + gamma.LN(h) + zeta) so the
frozen backbone is exactly preserved at initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PerceptualConnector(nn.Module):
    """C_perc: bottleneck projection from SigLIP to an MM-DiT channel size."""

    def __init__(self, siglip_dim: int, model_dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(siglip_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, model_dim),
        )

    def forward(self, t_perc: torch.Tensor) -> torch.Tensor:
        return self.proj(t_perc)


class PCMModulator(nn.Module):
    def __init__(self, model_dim: int, bottleneck: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, 2 * model_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        h: torch.Tensor,           # (B, N, d) hidden states
        perc_tokens: torch.Tensor,  # (B, Np, d) connected SigLIP tokens
        alpha: torch.Tensor,        # (B,) Psi schedules
        beta: torch.Tensor,
    ) -> torch.Tensor:
        b = h.shape[0]
        pooled = perc_tokens.mean(dim=1)  # (B, d)
        coeff = alpha.view(b, 1) * self.mlp(pooled) + beta.view(b, 1)
        gamma, zeta = coeff.chunk(2, dim=-1)
        return h + gamma.unsqueeze(1) * self.norm(h) + zeta.unsqueeze(1)
