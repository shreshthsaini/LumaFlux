"""LumaFluxPipeline: prompt-free SDR -> HDR conversion (Algorithm 1).

Starts at flow time one from the training-matched noisy SDR latent, then
integrates the adapter-steered velocity field down to t = 0. The final latent
is decoded by the frozen VAE and tone-expanded by the RQS head into PQ/BT.2020.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ..models.luma_flux import LumaFluxModel, pack_latents, unpack_latents


def time_shift(ts: torch.Tensor, shift: float) -> torch.Tensor:
    """Flux-style timestep shifting; shift=1 leaves the grid uniform."""
    return shift * ts / (1.0 + (shift - 1.0) * ts)


class LumaFluxPipeline:
    def __init__(
        self,
        model: LumaFluxModel,
        device: str | torch.device = "cpu",
        *,
        cfg: Mapping[str, Any] | None = None,
        bridge_noise: float | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        configured_noise = (cfg or {}).get("train", {}).get("bridge_noise", 0.05)
        self.bridge_noise = float(configured_noise if bridge_noise is None else bridge_noise)
        if self.bridge_noise < 0.0:
            raise ValueError("bridge_noise must be non-negative")

    @torch.no_grad()
    def __call__(
        self,
        x_sdr: torch.Tensor,
        num_steps: int = 8,
        time_shift_factor: float = 1.0,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        previous_spline_params: dict[str, torch.Tensor] | None = None,
        spline_ema: float = 0.0,
        tone_strength: float = 1.0,
        return_intermediate: bool = False,
        **deprecated_options: Any,
    ) -> dict:
        """Convert a batch of SDR frames to HDR.

        Args:
            x_sdr: (B,3,H,W) BT.709 SDR signal in [0,1]; H, W must be
                divisible by 2 * vae_scale.
            num_steps: Euler integration steps from one to zero.
            noise: optional bridge noise. Reuse it across video frames for
                temporally stable sampling.
            generator: optional RNG for reproducibility.
            previous_spline_params: actually applied parameters from the prior
                frame, used as the EMA state.
            spline_ema: prior-frame weight in [0, 1). Zero disables smoothing.
            tone_strength: optional RQS output scaling. One applies the learned
                tone field without additional scaling.

        Returns ``hdr`` as PQ/BT.2020 capped at the model mastering peak. The
        ``vae_out`` entry is the unconformed pre-RQS decode.
        """
        if "strength" in deprecated_options:
            raise TypeError(
                "strength was removed because inference now follows the full "
                "training-matched transport from t=1 to t=0"
            )
        if deprecated_options:
            names = ", ".join(sorted(deprecated_options))
            raise TypeError(f"unexpected sampling option(s): {names}")
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if not 0.0 <= spline_ema < 1.0:
            raise ValueError("spline_ema must be in [0, 1)")

        model = self.model
        x_sdr = x_sdr.to(self.device)
        mult = 2 * model.vae_scale
        if x_sdr.shape[-1] % mult or x_sdr.shape[-2] % mult:
            raise ValueError(f"Input H/W must be divisible by {mult}, got {x_sdr.shape[-2:]}")

        backbone_dtype = next(model.transformer.parameters()).dtype
        amp_enabled = self.device.type == "cuda" and backbone_dtype in {
            torch.bfloat16,
            torch.float16,
        }
        with torch.autocast(
            device_type=self.device.type,
            dtype=backbone_dtype,
            enabled=amp_enabled,
        ):
            cond = model.prepare_condition(x_sdr)
            z_sdr = model.encode_image(x_sdr)
            latent_hw = z_sdr.shape[-2:]
            if noise is None:
                noise = torch.randn(
                    z_sdr.shape, generator=generator, device=self.device, dtype=z_sdr.dtype
                )
            else:
                if noise.shape != z_sdr.shape:
                    raise ValueError(
                        f"noise shape must match SDR latent {tuple(z_sdr.shape)}, "
                        f"got {tuple(noise.shape)}"
                    )
                noise = noise.to(device=self.device, dtype=z_sdr.dtype)
            z = pack_latents(z_sdr + self.bridge_noise * noise)

            ts = time_shift(
                torch.linspace(1.0, 0.0, num_steps + 1, device=self.device),
                time_shift_factor,
            )
            trajectory = []
            for i in range(num_steps):
                t = ts[i].expand(z.shape[0])
                v = model.velocity(z, t, cond, latent_hw)
                z = z + (ts[i + 1] - ts[i]) * v
                if return_intermediate:
                    trajectory.append(z)

            if model.rqs is None:
                decoded = model.decode_hdr(z, latent_hw)
                spline_params = {}
            else:
                final_latent = unpack_latents(z, *latent_hw)
                current_params = model.rqs.spline_params(final_latent)
                applied_params = current_params
                if previous_spline_params is not None and spline_ema > 0.0:
                    names = ("widths", "heights", "derivs")
                    previous = tuple(
                        previous_spline_params[name].to(device=value.device, dtype=value.dtype)
                        for name, value in zip(names, current_params)
                    )
                    applied_params = tuple(
                        spline_ema * old + (1.0 - spline_ema) * new
                        for old, new in zip(previous, current_params)
                    )

                old_strength = model.rqs.strength
                model.rqs.strength = tone_strength
                try:
                    decoded = model.decode_hdr(z, latent_hw, spline_params=applied_params)
                finally:
                    model.rqs.strength = old_strength

                spline_params = {
                    name: decoded[name].detach().cpu()
                    for name in ("widths", "heights", "derivs")
                }
            out = {
                "hdr": decoded["hdr"].cpu(),
                "vae_out": decoded["vae_out"].cpu(),
                "spline_params": spline_params,
                "bridge_noise": noise.detach().cpu(),
            }
            if return_intermediate:
                out["trajectory"] = trajectory
            return out
