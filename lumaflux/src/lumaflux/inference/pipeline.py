"""LumaFluxPipeline: prompt-free SDR -> HDR conversion (Algorithm 1).

Starts the ODE at the SDR latent z_1 = E_VAE(x_sdr) (optionally with a
small amount of bridge noise matching training) and integrates the
adapter-steered velocity field down to t = 0 with an Euler solver over
``num_steps`` (40 by default, as in the paper). The final latent is decoded
by the frozen VAE and tone-expanded by the RQS head into PQ/BT.2020.
"""

from __future__ import annotations

import torch

from ..models.luma_flux import LumaFluxModel, pack_latents


def time_shift(ts: torch.Tensor, shift: float) -> torch.Tensor:
    """Flux-style timestep shifting; shift=1 leaves the grid uniform."""
    return shift * ts / (1.0 + (shift - 1.0) * ts)


class LumaFluxPipeline:
    def __init__(self, model: LumaFluxModel, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(
        self,
        x_sdr: torch.Tensor,
        num_steps: int = 40,
        bridge_noise: float = 0.05,
        time_shift_factor: float = 1.0,
        strength: float = 1.0,
        generator: torch.Generator | None = None,
        return_intermediate: bool = False,
    ) -> dict:
        """Convert a batch of SDR frames to HDR.

        Args:
            x_sdr: (B,3,H,W) BT.709 SDR signal in [0,1]; H, W must be
                divisible by 2 * vae_scale.
            num_steps: ODE steps (paper: 40).
            bridge_noise: noise added to the SDR anchor (match training).
            strength: RQS tone-expansion strength in [0, ~2].
            generator: optional RNG for reproducibility.

        Returns dict with ``hdr`` (B,3,H,W) PQ/BT.2020 in [0,1] and ``vae_out``
        (pre-RQS decode), plus intermediates if requested.
        """
        model = self.model
        x_sdr = x_sdr.to(self.device)
        mult = 2 * model.vae_scale
        if x_sdr.shape[-1] % mult or x_sdr.shape[-2] % mult:
            raise ValueError(f"Input H/W must be divisible by {mult}, got {x_sdr.shape[-2:]}")

        cond = model.prepare_condition(x_sdr)
        z = model.encode_image(x_sdr)
        latent_hw = z.shape[-2:]
        if bridge_noise > 0:
            noise = torch.randn(z.shape, generator=generator, device=self.device, dtype=z.dtype)
            z = z + bridge_noise * noise
        z = pack_latents(z)

        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        ts = time_shift(ts, time_shift_factor)
        trajectory = []
        for i in range(num_steps):
            t = ts[i].expand(z.shape[0])
            v = model.velocity(z, t, cond, latent_hw)
            z = z + (ts[i + 1] - ts[i]) * v
            if return_intermediate:
                trajectory.append(z)

        old_strength = model.rqs.strength
        model.rqs.strength = strength
        try:
            decoded = model.decode_hdr(z, latent_hw)
        finally:
            model.rqs.strength = old_strength

        out = {"hdr": decoded["hdr"].cpu(), "vae_out": decoded["vae_out"].cpu()}
        if return_intermediate:
            out["trajectory"] = trajectory
        return out
