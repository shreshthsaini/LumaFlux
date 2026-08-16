"""Luma-MMDiT block wrappers: insert PGA / PCM / Coupler into frozen Flux.

``FluxTransformerBlock`` uses separate image and context streams. Diffusers
0.32 passes a combined stream to ``FluxSingleTransformerBlock``, while newer
versions can pass the streams separately. One wrapper handles both supported
forms. Per block, the wrapper:

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
        bottleneck: int = 64,
        enable_pga: bool = True,
        pga_spectral: bool = True,
        enable_pcm: bool = True,
        enable_coupler: bool = True,
        single_stream: bool = False,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.single_stream = single_stream
        self.modulation = modulation  # shared module (registered once on parent)
        self.context = context
        self.pga = (
            PGAValueAdapter(
                dim,
                heads,
                phys_channels,
                stats_dim,
                num_bands,
                rank=rank,
                spectral=pga_spectral,
            )
            if enable_pga
            else None
        )
        self.pcm = PCMModulator(dim, bottleneck=bottleneck) if enable_pcm else None
        self.coupler = (
            HDRResidualCoupler(phys_channels, dim, bottleneck=bottleneck)
            if enable_coupler
            else None
        )
        if self.pga is not None:
            inner.attn.to_v = PGAPatchedToV(inner.attn.to_v, self.pga, context)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        image_rotary_emb: Any = None,
        joint_attention_kwargs: dict | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Diffusers 0.32 gradient checkpointing supplies the legacy single-block
        # arguments positionally as (hidden_states, temb, image_rotary_emb).
        if (
            self.single_stream
            and encoder_hidden_states is not None
            and encoder_hidden_states.ndim == 2
        ):
            image_rotary_emb = temb
            temb = encoder_hidden_states
            encoder_hidden_states = None

        ctx = self.context
        if not ctx.get("active", False):
            if self.single_stream and encoder_hidden_states is None:
                return self.inner(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            return self.inner(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        sched = self.modulation(ctx["t"], self.layer_idx)
        legacy_single_stream = self.single_stream and encoder_hidden_states is None
        num_prefix_tokens = ctx["num_context_tokens"] if legacy_single_stream else 0
        prefix = None
        if legacy_single_stream:
            prefix = hidden_states[:, :num_prefix_tokens]
            hidden_states = hidden_states[:, num_prefix_tokens:]

        # PCM on the image stream (Eq. 13-14).
        if self.pcm is not None:
            hidden_states = self.pcm(
                hidden_states,
                ctx["perc_tokens"],
                sched["alpha_pcm"],
                sched["beta_pcm"],
            )
        # Arm the PGA value patch for the frozen attention call (Eq. 9-12).
        if self.pga is not None:
            ctx["alpha_pga"] = sched["alpha_pga"]
            ctx["beta_pga"] = sched["beta_pga"]
            ctx["n_spec"] = sched["n_spec"]
        if legacy_single_stream:
            hidden_states = torch.cat([prefix, hidden_states], dim=1)
        ctx["num_prefix_tokens"] = num_prefix_tokens
        if self.single_stream and encoder_hidden_states is not None:
            ctx["num_prefix_tokens"] = encoder_hidden_states.shape[1]

        if legacy_single_stream:
            hidden_states = self.inner(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            prefix = hidden_states[:, :num_prefix_tokens]
            hidden_states = hidden_states[:, num_prefix_tokens:]
        else:
            encoder_hidden_states, hidden_states = self.inner(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # HDR Residual Coupler on the image-stream residual output (Eq. 15).
        if self.coupler is not None:
            hidden_states = self.coupler(
                hidden_states,
                ctx["phys_tokens"],
                ctx["perc_tokens"],
                sched["lam"],
                ctx["grid_hw"],
            )
        if legacy_single_stream:
            return torch.cat([prefix, hidden_states], dim=1)
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
    bottleneck: int = 64,
    enable_pga: bool = True,
    pga_spectral: bool = True,
    enable_pcm: bool = True,
    enable_coupler: bool = True,
) -> list[LumaBlockWrapper]:
    """Wrap every dual- and single-stream block in-place; returns wrappers."""
    if not any((enable_pga, enable_pcm, enable_coupler)):
        return []
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
        bottleneck=bottleneck,
        enable_pga=enable_pga,
        pga_spectral=pga_spectral,
        enable_pcm=enable_pcm,
        enable_coupler=enable_coupler,
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
