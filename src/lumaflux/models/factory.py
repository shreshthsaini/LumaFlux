"""Model factory: build the LumaFlux stack from a YAML/dict config.

Two modes:
  * pretrained: load FLUX.1-dev (or -schnell) and SigLIP from the Hub,
  * tiny: randomly-initialized miniature stack with identical structure,
    fully offline - used for CPU smoke tests, CI, and debugging.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import yaml

from .luma_flux import LumaFluxModel


def _hub_offline() -> bool:
    """True when hub access is disabled, so loads must stay cache-local."""
    return os.environ.get("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")


def _disable_cudnn_sdpa() -> None:
    """cuDNN's fused attention graph fails (mha_graph.execute / illegal
    memory access) on GH200 with torch 2.11+cu128 for this model's shapes;
    flash and mem-efficient SDPA backends are unaffected."""
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


_DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_torch_dtype(value: str | torch.dtype | None) -> torch.dtype | None:
    if value is None or isinstance(value, torch.dtype):
        return value
    try:
        return _DTYPES[value.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype: {value}") from exc


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


def build_model(
    cfg: dict[str, Any],
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype | None = None,
) -> LumaFluxModel:
    _disable_cudnn_sdpa()
    mcfg = cfg.get("model", cfg)
    backbone = mcfg.get("backbone", "tiny")
    siglip_id = mcfg.get("siglip", "tiny")
    torch_dtype = resolve_torch_dtype(dtype if dtype is not None else mcfg.get("dtype"))

    if backbone == "tiny":
        transformer = _tiny_transformer()
        vae = _tiny_vae()
        if torch_dtype is not None:
            transformer.to(dtype=torch_dtype)
            vae.to(dtype=torch_dtype)
    else:
        from diffusers import AutoencoderKL, FluxTransformer2DModel

        load_kw = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
        if _hub_offline():
            # diffusers 0.39 probes the hub API even under HF_HUB_OFFLINE.
            load_kw["local_files_only"] = True
        transformer = FluxTransformer2DModel.from_pretrained(
            backbone, subfolder="transformer", **load_kw
        )
        vae = AutoencoderKL.from_pretrained(backbone, subfolder="vae", **load_kw)

    if siglip_id == "tiny":
        siglip = _tiny_siglip()
        if torch_dtype is not None:
            siglip.to(dtype=torch_dtype)
    else:
        from transformers import SiglipVisionModel

        load_kw = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
        if _hub_offline():
            load_kw["local_files_only"] = True
        siglip = SiglipVisionModel.from_pretrained(siglip_id, **load_kw)

    model = LumaFluxModel(
        transformer,
        vae,
        siglip,
        rank=mcfg.get("rank", 8),
        bottleneck=mcfg.get("bottleneck", 64),
        num_knots=mcfg.get("num_knots", 8),
        phys_channels=mcfg.get("phys_channels", 32),
        stats_dim=mcfg.get("stats_dim", 16),
        num_bands=mcfg.get("num_bands", 8),
        num_null_tokens=mcfg.get("num_null_tokens", 8),
        modulation_hidden=mcfg.get("modulation_hidden", 128),
        guidance_scale=mcfg.get("guidance_scale", 1.0),
        peak_nits=mcfg.get("peak_nits", 1000.0),
        enable_pga=mcfg.get("enable_pga", True),
        pga_spectral=mcfg.get("pga_spectral", True),
        enable_pcm=mcfg.get("enable_pcm", True),
        enable_coupler=mcfg.get("enable_coupler", True),
        rqs_mode=mcfg.get("rqs_mode", "monotone"),
        lora_mode=mcfg.get("lora_mode", "none"),
        lora_rank=mcfg.get("lora_rank"),
        time_layer_modulation=mcfg.get("time_layer_modulation", True),
    )
    if mcfg.get("gradient_checkpointing", False):
        if not hasattr(transformer, "enable_gradient_checkpointing"):
            raise RuntimeError("Configured transformer does not support gradient checkpointing")
        transformer.enable_gradient_checkpointing()
    return model.to(device)
