"""LumaFluxModel: frozen Flux MM-DiT + trainable LumaFlux modules (Sec. 4).

Trainable components (everything else is frozen):
  * PhysicalEncoder            (T_phys, g, r)
  * PerceptualConnector x2     (SigLIP -> model dim, SigLIP -> context dim)
  * learned null context tokens + pooled embedding (prompt-free operation)
  * TimestepLayerModulation    Psi(t, l)
  * per-block PGA / PCM / HDR Residual Coupler (inside block wrappers)
  * RQSToneFieldDecoder

The model is prompt-free: the frozen backbone's text pathway receives a
small set of *learned* context tokens concatenated with projected SigLIP
tokens, and the pooled projection is a learned vector fused with the
global luminance statistics embedding MLP_g(s_g).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import wrap_flux_blocks
from .modulation import TimestepLayerModulation
from .pcm import PerceptualConnector
from .physical import PhysicalEncoder
from .rqs import RQSToneFieldDecoder


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H/2*W/2, C*4) 2x2 patch packing (Flux convention)."""
    b, c, h, w = latents.shape
    x = latents.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(packed: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(B, N, C*4) -> (B, C, H, W) inverse of :func:`pack_latents`."""
    b, n, cd = packed.shape
    c = cd // 4
    x = packed.view(b, h // 2, w // 2, c, 2, 2)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c, h, w)


def make_img_ids(h: int, w: int, device, dtype) -> torch.Tensor:
    ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(h // 2, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(w // 2, device=device, dtype=dtype)[None, :]
    return ids.reshape(-1, 3)


class LumaFluxModel(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        siglip: nn.Module,
        *,
        rank: int = 8,
        num_knots: int = 8,
        phys_channels: int = 32,
        stats_dim: int = 16,
        num_bands: int = 8,
        num_null_tokens: int = 8,
        modulation_hidden: int = 128,
        siglip_image_size: int | None = None,
        guidance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.vae = vae
        self.siglip = siglip
        for module in (transformer, vae, siglip):
            module.requires_grad_(False)

        cfg = transformer.config
        self.inner_dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.context_dim = cfg.joint_attention_dim
        self.pooled_dim = cfg.pooled_projection_dim
        self.guidance_embeds = bool(getattr(cfg, "guidance_embeds", False))
        self.guidance_scale = guidance_scale
        self.latent_channels = vae.config.latent_channels
        self.vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
        self.vae_scaling = getattr(vae.config, "scaling_factor", 1.0) or 1.0
        self.vae_shift = getattr(vae.config, "shift_factor", 0.0) or 0.0

        siglip_dim = siglip.config.hidden_size
        self.siglip_image_size = siglip_image_size or siglip.config.image_size

        num_layers = len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)
        self.context: dict = {"active": False}
        self.physical = PhysicalEncoder(phys_channels, stats_dim, num_bands)
        self.perc_connector = PerceptualConnector(siglip_dim, self.inner_dim)
        self.ctx_connector = PerceptualConnector(siglip_dim, self.context_dim)
        self.null_context = nn.Parameter(torch.randn(num_null_tokens, self.context_dim) * 0.02)
        self.null_pooled = nn.Parameter(torch.zeros(self.pooled_dim))
        self.pooled_from_stats = nn.Linear(stats_dim, self.pooled_dim)
        nn.init.zeros_(self.pooled_from_stats.weight)
        self.modulation = TimestepLayerModulation(num_layers, modulation_hidden)
        self.block_wrappers = wrap_flux_blocks(
            transformer,
            self.modulation,
            self.context,
            phys_channels=phys_channels,
            stats_dim=stats_dim,
            num_bands=num_bands,
            rank=rank,
        )
        self.rqs = RQSToneFieldDecoder(self.latent_channels, num_knots=num_knots)

    # ------------------------------------------------------------------ VAE
    @torch.no_grad()
    def encode_image(self, x01: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [0,1] -> latent (B,C,h,w)."""
        posterior = self.vae.encode(x01 * 2.0 - 1.0).latent_dist
        z = posterior.mode()  # deterministic: keeps inference reproducible
        return (z - self.vae_shift) * self.vae_scaling

    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """latent (B,C,h,w) -> (B,3,H,W) in [0,1] (PQ/BT.2020 signal)."""
        z = z / self.vae_scaling + self.vae_shift
        x = self.vae.decode(z).sample
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)

    # -------------------------------------------------------- conditioning
    def prepare_condition(self, x_sdr: torch.Tensor) -> dict:
        """Compute all SDR-derived conditioning for a batch (B,3,H,W)."""
        b = x_sdr.shape[0]
        phys = self.physical(x_sdr)
        size = self.siglip_image_size
        x_sig = F.interpolate(x_sdr, size=(size, size), mode="bilinear", align_corners=False)
        x_sig = (x_sig - 0.5) / 0.5
        with torch.no_grad():
            sig_tokens = self.siglip(pixel_values=x_sig).last_hidden_state
        perc_tokens = self.perc_connector(sig_tokens)
        ctx_tokens = torch.cat(
            [self.null_context.unsqueeze(0).expand(b, -1, -1), self.ctx_connector(sig_tokens)],
            dim=1,
        )
        pooled = self.null_pooled.unsqueeze(0).expand(b, -1) + self.pooled_from_stats(phys["g"])
        return {
            "phys": phys,
            "perc_tokens": perc_tokens,
            "ctx_tokens": ctx_tokens,
            "pooled": pooled,
        }

    # ------------------------------------------------------------ velocity
    def velocity(
        self,
        z_packed: torch.Tensor,
        t: torch.Tensor,
        cond: dict,
        latent_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Predict the flow velocity for packed latents ``z`` at time ``t``."""
        h, w = latent_hw
        grid_hw = (h // 2, w // 2)
        ctx = self.context
        ctx.update(
            active=True,
            t=t,
            phys_tokens=PhysicalEncoder.to_tokens(cond["phys"]["t_phys"], grid_hw),
            g=cond["phys"]["g"],
            r=cond["phys"]["r"],
            perc_tokens=cond["perc_tokens"],
            grid_hw=grid_hw,
        )
        img_ids = make_img_ids(h, w, z_packed.device, z_packed.dtype)
        txt_ids = torch.zeros(cond["ctx_tokens"].shape[1], 3, device=z_packed.device, dtype=z_packed.dtype)
        guidance = None
        if self.guidance_embeds:
            guidance = torch.full_like(t, self.guidance_scale)
        try:
            out = self.transformer(
                hidden_states=z_packed,
                encoder_hidden_states=cond["ctx_tokens"],
                pooled_projections=cond["pooled"],
                timestep=t,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )[0]
        finally:
            ctx["active"] = False
        return out

    # ------------------------------------------------------------- decode
    def decode_hdr(self, z_packed: torch.Tensor, latent_hw: tuple[int, int]) -> dict:
        """Final latent -> PQ/BT.2020 HDR frame via VAE + RQS tone field.

        Returns the RQS output dict plus ``vae_out``, the pre-RQS decode.
        """
        z = unpack_latents(z_packed, *latent_hw)
        x_out = self.decode_latent(z)
        out = self.rqs(x_out, z)
        out["vae_out"] = x_out
        return out

    # ------------------------------------------------------------ training
    def forward(
        self,
        sdr: torch.Tensor,
        hdr: torch.Tensor,
        bridge_noise: float = 0.05,
        do_recon: bool = False,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict:
        """One training-step computation (used as the DDP entry point).

        Builds the noisy SDR->HDR latent bridge, predicts the velocity, and
        (optionally) decodes a one-step x0 estimate for the Eq. 18 losses.
        Running this inside ``forward`` keeps every gradient-producing op
        under the distributed wrapper so multi-process gradient sync works.
        """
        with torch.no_grad():
            z_sdr = self.encode_image(sdr)
            z_hdr = self.encode_image(hdr)
        latent_hw = z_hdr.shape[-2:]
        if noise is None:
            noise = torch.randn_like(z_sdr)
        if t is None:
            t = torch.rand(z_hdr.shape[0], device=z_hdr.device, dtype=z_hdr.dtype)
        z_start = z_sdr + bridge_noise * noise
        tt = t.view(-1, 1, 1, 1)
        z_t_packed = pack_latents((1.0 - tt) * z_hdr + tt * z_start)

        cond = self.prepare_condition(sdr)
        v_pred = self.velocity(z_t_packed, t, cond, latent_hw)
        out = {
            "v_pred": v_pred,
            "v_target": pack_latents(z_start - z_hdr),
            "t": t,
        }
        if do_recon:
            z0_packed = z_t_packed - t.view(-1, 1, 1) * v_pred
            out["decoded"] = self.decode_hdr(z0_packed, latent_hw)
        return out

    # ---------------------------------------------------------- checkpoint
    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        frozen_prefixes = ("vae.", "siglip.")
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad and not name.startswith(frozen_prefixes):
                state[name] = param.detach().cpu()
        return state

    def save_adapters(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(self.trainable_state_dict(), str(path))

    def load_adapters(self, path: str | Path, strict: bool = True) -> None:
        from safetensors.torch import load_file

        state = load_file(str(path))
        missing, unexpected = self.load_state_dict(state, strict=False)
        if strict and unexpected:
            raise RuntimeError(f"Unexpected keys in adapter checkpoint: {unexpected[:5]}...")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
