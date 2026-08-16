"""Physically-Guided Adaptation (PGA), Sec. 4.3, Eq. 9-12.

A gated low-rank residual added to the frozen value projection W_V of each
attention layer:

    R_base = A B                       (rank r)
    G_phys = sigmoid(P_v [T_phys || g])        per-token, per-head gate
    g_FFT  = softplus(W_r r)                   per-head spectral gate
    R(t,l) = (alpha A B + beta I) G_phys (I + n_spec Diag(g_FFT))
    W_V <- W_V^(0) + R(t,l)

Implemented functionally: the wrapped ``to_v`` computes
``v + gate * (alpha * A(B(x)) + beta * x)`` where ``gate`` combines the
per-token/per-head physical gate with the spectral term, broadcast over
head dimensions. Text/context tokens (single-stream blocks prepend them)
receive a neutral gate of 1 so physical cues only modulate image tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PGAValueAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        phys_channels: int,
        stats_dim: int,
        num_bands: int,
        rank: int = 8,
        spectral: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)  # residual starts at zero
        self.p_v = nn.Linear(phys_channels + stats_dim, heads)
        self.w_r = nn.Linear(num_bands, heads) if spectral else None
        if self.w_r is not None:
            nn.init.zeros_(self.w_r.weight)

    def forward(
        self,
        x: torch.Tensor,            # (B, N, dim) post-norm hidden states
        v: torch.Tensor,            # (B, N, dim) frozen value projection
        phys_tokens: torch.Tensor,  # (B, N_img, C_phys)
        g: torch.Tensor,            # (B, stats_dim)
        r: torch.Tensor,            # (B, K)
        alpha: torch.Tensor,        # (B,) Psi schedules
        beta: torch.Tensor,
        n_spec: torch.Tensor,
        num_prefix_tokens: int = 0,
    ) -> torch.Tensor:
        b, n, d = x.shape
        low_rank = self.up(self.down(x))
        resid = alpha.view(b, 1, 1) * low_rank + beta.view(b, 1, 1) * x

        # Physical per-token/per-head gate on image tokens (Eq. 10).
        cond = torch.cat(
            [phys_tokens, g.unsqueeze(1).expand(-1, phys_tokens.shape[1], -1)], dim=-1
        )
        gate_img = torch.sigmoid(self.p_v(cond))  # (B, N_img, heads)
        if num_prefix_tokens > 0:
            ones = gate_img.new_ones(b, num_prefix_tokens, self.heads)
            gate = torch.cat([ones, gate_img], dim=1)
        else:
            gate = gate_img
        # Spectral gate (Eq. 11) mixed in via the n_spec schedule (Eq. 12).
        if self.w_r is not None:
            g_fft = torch.nn.functional.softplus(self.w_r(r))  # (B, heads)
            gate = gate * (1.0 + n_spec.view(b, 1, 1) * g_fft.unsqueeze(1))

        gate = gate[:, :n]  # safety for any token-count mismatch
        resid = resid.view(b, n, self.heads, self.head_dim) * gate.unsqueeze(-1)
        return v + resid.view(b, n, d)


class PGAPatchedToV(nn.Module):
    """Drop-in replacement for ``attn.to_v`` that adds the PGA residual.

    Conditioning tensors are read from a shared mutable ``context`` dict
    populated by the enclosing Luma block wrapper right before the frozen
    attention call.
    """

    def __init__(self, base: nn.Module, adapter: PGAValueAdapter, context: dict) -> None:
        super().__init__()
        self.base = base
        # Deliberately NOT registered as a submodule: the adapter is owned
        # (and registered) by the LumaBlockWrapper as ``.pga`` so checkpoint
        # keys are unique and readable.
        object.__setattr__(self, "_adapter", adapter)
        self.context = context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.base(x)
        ctx = self.context
        if not ctx.get("active", False):
            return v
        return self._adapter(
            x,
            v,
            phys_tokens=ctx["phys_tokens"],
            g=ctx["g"],
            r=ctx["r"],
            alpha=ctx["alpha_pga"],
            beta=ctx["beta_pga"],
            n_spec=ctx["n_spec"],
            num_prefix_tokens=ctx.get("num_prefix_tokens", 0),
        )
