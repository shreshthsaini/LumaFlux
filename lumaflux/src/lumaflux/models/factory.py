"""Model factory: build the LumaFlux stack from a YAML/dict config.

Two modes:
  * pretrained: load FLUX.1-dev (or -schnell) and SigLIP from the Hub,
  * tiny: randomly-initialized miniature stack with identical structure,
    fully offline - used for CPU smoke tests, CI, and debugging.
"""

from __future__ import annotations

from typing import Any

import torch
import yaml

from .luma_flux import LumaFluxModel


def _tiny_transformer():
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=16,  # packed: 4 latent channels x 2x2
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        guidance_embeds=False,
        axes_dims_rope=(4, 6, 6),
    )


def _tiny_vae():
    from diffusers import AutoencoderKL

    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 8, 8, 8),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=64,
    )


def _tiny_siglip():
    from transformers import SiglipVisionConfig, SiglipVisionModel

    cfg = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=64,
        patch_size=16,
    )
    return SiglipVisionModel(cfg)


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict[str, Any], device: str | torch.device = "cpu",
                dtype: torch.dtype | None = None) -> LumaFluxModel:
    mcfg = cfg.get("model", cfg)
    backbone = mcfg.get("backbone", "tiny")
    siglip_id = mcfg.get("siglip", "tiny")

    if backbone == "tiny":
        transformer = _tiny_transformer()
        vae = _tiny_vae()
    else:
        from diffusers import AutoencoderKL, FluxTransformer2DModel

        load_kw = {"torch_dtype": dtype} if dtype is not None else {}
        transformer = FluxTransformer2DModel.from_pretrained(
            backbone, subfolder="transformer", **load_kw
        )
        vae = AutoencoderKL.from_pretrained(backbone, subfolder="vae", **load_kw)

    if siglip_id == "tiny":
        siglip = _tiny_siglip()
    else:
        from transformers import SiglipVisionModel

        siglip = SiglipVisionModel.from_pretrained(siglip_id)

    model = LumaFluxModel(
        transformer,
        vae,
        siglip,
        rank=mcfg.get("rank", 8),
        num_knots=mcfg.get("num_knots", 8),
        phys_channels=mcfg.get("phys_channels", 32),
        stats_dim=mcfg.get("stats_dim", 16),
        num_bands=mcfg.get("num_bands", 8),
        num_null_tokens=mcfg.get("num_null_tokens", 8),
        modulation_hidden=mcfg.get("modulation_hidden", 128),
        guidance_scale=mcfg.get("guidance_scale", 1.0),
    )
    if dtype is not None and backbone == "tiny":
        model = model.to(dtype)
    return model.to(device)
