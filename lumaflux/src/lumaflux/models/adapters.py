"""Luma-MMDiT block wrappers: insert PGA / PCM / Coupler into frozen Flux.

Both ``FluxTransformerBlock`` (dual-stream) and ``FluxSingleTransformerBlock``
share the call signature::

    (hidden_states, encoder_hidden_states, temb, image_rotary_emb,
     joint_attention_kwargs) -> (encoder_hidden_states, hidden_states)

so a single wrapper type handles both. Per block, the wrapper:

  1. evaluates the shared conditioner Psi(t, l) (Sec. 4.2),
  2. applies PCM residual FiLM to the image-stream hidden states (Sec. 4.4),
  3. arms the PGA value-projection patch with physical cues (Sec. 4.3),
  4. runs the frozen block,
  5. applies the HDR Residual Coupler to the image-stream output (Sec. 4.5).

Conditioning tensors travel through a shared mutable ``context`` dict that
the parent :class:`~lumaflux.models.luma_flux.LumaFluxModel` populates once
per transformer forward (blocks execute sequentially, so per-block fields
written in step 3 cannot race).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .coupler import HDRResidualCoupler
from .modulation import TimestepLayerModulation
from .pcm import PCMModulator
from .pga import PGAPatchedToV, PGAValueAdapter


class LumaBlockWrapper(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        layer_idx: int,
        modulation: TimestepLayerModulation,
        context: dict,
        *,
        dim: int,
        heads: int,
        phys_channels: int,
        stats_dim: int,
        num_bands: int,
        rank: int = 8,
        single_stream: bool = False,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.single_stream = single_stream
        self.modulation = modulation  # shared module (registered once on parent)
        self.context = context
        self.pga = PGAValueAdapter(dim, heads, phys_channels, stats_dim, num_bands, rank=rank)
        self.pcm = PCMModulator(dim)
        self.coupler = HDRResidualCoupler(phys_channels, dim)
        inner.attn.to_v = PGAPatchedToV(inner.attn.to_v, self.pga, context)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Any = None,
        joint_attention_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.context
        if not ctx.get("active", False):
            return self.inner(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        sched = self.modulation(ctx["t"], self.layer_idx)
        # PCM on the image stream (Eq. 13-14).
        hidden_states = self.pcm(
            hidden_states, ctx["perc_tokens"], sched["alpha_pcm"], sched["beta_pcm"]
        )
        # Arm the PGA value patch for the frozen attention call (Eq. 9-12).
        ctx["alpha_pga"] = sched["alpha_pga"]
        ctx["beta_pga"] = sched["beta_pga"]
        ctx["n_spec"] = sched["n_spec"]
        ctx["num_prefix_tokens"] = (
            encoder_hidden_states.shape[1] if self.single_stream else 0
        )

        encoder_hidden_states, hidden_states = self.inner(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        # HDR Residual Coupler on the image-stream residual output (Eq. 15).
        hidden_states = self.coupler(
            hidden_states,
            ctx["phys_tokens"],
            ctx["perc_tokens"],
            sched["lam"],
            ctx["grid_hw"],
        )
        return encoder_hidden_states, hidden_states


def wrap_flux_blocks(
    transformer: nn.Module,
    modulation: TimestepLayerModulation,
    context: dict,
    *,
    phys_channels: int,
    stats_dim: int,
    num_bands: int,
    rank: int = 8,
) -> list[LumaBlockWrapper]:
    """Wrap every dual- and single-stream block in-place; returns wrappers."""
    dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    heads = transformer.config.num_attention_heads
    common = dict(
        modulation=modulation,
        context=context,
        dim=dim,
        heads=heads,
        phys_channels=phys_channels,
        stats_dim=stats_dim,
        num_bands=num_bands,
        rank=rank,
    )
    wrappers: list[LumaBlockWrapper] = []
    layer = 0
    for i, block in enumerate(transformer.transformer_blocks):
        w = LumaBlockWrapper(block, layer, single_stream=False, **common)
        transformer.transformer_blocks[i] = w
        wrappers.append(w)
        layer += 1
    for i, block in enumerate(transformer.single_transformer_blocks):
        w = LumaBlockWrapper(block, layer, single_stream=True, **common)
        transformer.single_transformer_blocks[i] = w
        wrappers.append(w)
        layer += 1
    return wrappers
