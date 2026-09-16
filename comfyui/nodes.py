"""ComfyUI nodes for LumaFlux: prompt-free SDR-to-HDR conversion on a frozen FLUX.1-dev.

Cloning the repository into ComfyUI/custom_nodes registers these nodes through the
root __init__.py; this folder can also be symlinked there on its own when the
lumaflux package is pip-installed.

The nodes wrap the reference ``LumaFluxPipeline`` from the LumaFlux package, so
the sampling loop, the video stabilization, and the tone decoder are exactly the
released ones. What this file adds is model loading from ComfyUI's model folders,
batch-as-video handling, an SDR preview, and HDR-preserving save nodes.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths

from .hdr_io import pad_to_multiple, pq_to_sdr_preview, write_png16

log = logging.getLogger("LumaFlux.comfyui")

ADAPTER_REPO = "shreshthsaini/LumaFlux"
ADAPTER_FILES = ("lumaflux-main.safetensors", "lumaflux-hdrtv1k.safetensors")
FLUX_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_FROM_HUB = "black-forest-labs/FLUX.1-dev (Hugging Face, gated)"
SIGLIP_REPO = "google/siglip-so400m-patch14-384"
INSTALL_HINT = (
    "The LumaFlux nodes need the lumaflux package and its inference dependencies. Clone the whole "
    "LumaFlux repository into ComfyUI/custom_nodes and run `pip install -r requirements.txt` from "
    "it with ComfyUI's Python, or `pip install -e .` the repository into that environment."
)

_HERE = os.path.dirname(os.path.abspath(__file__))
FLUX_CONFIG_DIR = os.path.join(_HERE, "flux_configs")

# Paper configuration of the trainable modules (configs/model/flux_dev.yaml in the
# LumaFlux repository). Both released checkpoints share it.
MODEL_HPARAMS = dict(
    rank=8,
    bottleneck=64,
    num_knots=8,
    phys_channels=32,
    stats_dim=16,
    num_bands=8,
    num_null_tokens=8,
    modulation_hidden=128,
    guidance_scale=1.0,
    peak_nits=1000.0,
)
BRIDGE_NOISE = 0.05  # train.bridge_noise in the paper configuration
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

LUMAFLUX_DIR = os.path.join(folder_paths.models_dir, "lumaflux")
os.makedirs(LUMAFLUX_DIR, exist_ok=True)
if "lumaflux" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("lumaflux", LUMAFLUX_DIR)
else:
    folder_paths.folder_names_and_paths["lumaflux"] = ([LUMAFLUX_DIR], {".safetensors"})


def _require_lumaflux() -> None:
    """Import the package; when this folder sits inside a clone of the repository, use its src/."""
    try:
        import lumaflux  # noqa: F401
        return
    except ImportError:
        pass
    src = os.path.join(os.path.dirname(_HERE), "src")
    if os.path.isdir(os.path.join(src, "lumaflux")) and src not in sys.path:
        sys.path.insert(0, src)
        try:
            import lumaflux  # noqa: F401
            return
        except ImportError as exc:
            raise ImportError(INSTALL_HINT) from exc
    raise ImportError(INSTALL_HINT)


def _adapter_path(name: str) -> str:
    path = folder_paths.get_full_path("lumaflux", name)
    if path is not None:
        return path
    if name not in ADAPTER_FILES:
        raise FileNotFoundError(f"{name} is not in {LUMAFLUX_DIR} and is not a released LumaFlux checkpoint")
    from huggingface_hub import hf_hub_download

    log.info("LumaFlux: downloading %s from %s", name, ADAPTER_REPO)
    return hf_hub_download(ADAPTER_REPO, name, local_dir=LUMAFLUX_DIR)


class LumaFluxHandle:
    """What the loader hands to the converter: the reference pipeline plus provenance."""

    def __init__(self, pipe, adapters: str, transformer: str, dtype: torch.dtype) -> None:
        self.pipe = pipe
        self.adapters = adapters
        self.transformer = transformer
        self.dtype = dtype

    @property
    def model(self):
        return self.pipe.model

    @property
    def device(self):
        return self.pipe.device


class LumaFluxLoader:
    @classmethod
    def INPUT_TYPES(cls):
        adapters = sorted(set(folder_paths.get_filename_list("lumaflux")) | set(ADAPTER_FILES))
        transformers = [FLUX_FROM_HUB] + folder_paths.get_filename_list("diffusion_models")
        vaes = [FLUX_FROM_HUB] + folder_paths.get_filename_list("vae")
        return {
            "required": {
                "adapters": (
                    adapters,
                    {
                        "default": ADAPTER_FILES[0],
                        "tooltip": "LumaFlux adapter checkpoint in models/lumaflux. The two released files "
                        "download from Hugging Face on first use. lumaflux-main is the general model; "
                        "lumaflux-hdrtv1k is the in-domain HDRTV1K model.",
                    },
                ),
                "flux_transformer": (
                    transformers,
                    {
                        "default": FLUX_FROM_HUB,
                        "tooltip": "Frozen FLUX.1-dev backbone: the gated diffusers repo (needs a logged-in "
                        "Hugging Face account that accepted the license), or a single-file flux1-dev "
                        "checkpoint from models/diffusion_models.",
                    },
                ),
                "flux_vae": (
                    vaes,
                    {
                        "default": FLUX_FROM_HUB,
                        "tooltip": "FLUX VAE: from the diffusers repo, or ae.safetensors from models/vae.",
                    },
                ),
                "siglip": (
                    "STRING",
                    {"default": SIGLIP_REPO, "tooltip": "Hugging Face id of the frozen SigLIP vision encoder."},
                ),
                "dtype": (list(DTYPES), {"default": "bf16", "tooltip": "Backbone precision. The paper runs bf16."}),
            }
        }

    RETURN_TYPES = ("LUMAFLUX_MODEL",)
    FUNCTION = "load"
    CATEGORY = "LumaFlux"
    DESCRIPTION = (
        "Loads the frozen FLUX.1-dev transformer and VAE, the frozen SigLIP encoder, and one LumaFlux "
        "adapter checkpoint (71M parameters). Everything else in ComfyUI is unloaded first: the backbone "
        "alone takes about 24 GB in bf16."
    )

    def load(self, adapters, flux_transformer, flux_vae, siglip, dtype):
        _require_lumaflux()
        from diffusers import AutoencoderKL, FluxTransformer2DModel
        from transformers import SiglipVisionModel

        from lumaflux.inference.pipeline import LumaFluxPipeline
        from lumaflux.models import factory
        from lumaflux.models.luma_flux import LumaFluxModel

        torch_dtype = DTYPES[dtype]
        device = mm.get_torch_device()
        adapter_path = _adapter_path(adapters)

        mm.unload_all_models()
        mm.soft_empty_cache()
        # Same workaround the reference loader applies (cuDNN fused attention faults on some GPUs).
        if hasattr(factory, "_disable_cudnn_sdpa"):
            factory._disable_cudnn_sdpa()

        if flux_transformer == FLUX_FROM_HUB:
            transformer = FluxTransformer2DModel.from_pretrained(
                FLUX_REPO, subfolder="transformer", torch_dtype=torch_dtype
            )
        else:
            transformer = FluxTransformer2DModel.from_single_file(
                folder_paths.get_full_path_or_raise("diffusion_models", flux_transformer),
                config=FLUX_CONFIG_DIR,
                subfolder="transformer",
                torch_dtype=torch_dtype,
            )
        if flux_vae == FLUX_FROM_HUB:
            vae = AutoencoderKL.from_pretrained(FLUX_REPO, subfolder="vae", torch_dtype=torch_dtype)
        else:
            vae = AutoencoderKL.from_single_file(
                folder_paths.get_full_path_or_raise("vae", flux_vae),
                config=FLUX_CONFIG_DIR,
                subfolder="vae",
                torch_dtype=torch_dtype,
            )
        siglip_model = SiglipVisionModel.from_pretrained(siglip, torch_dtype=torch_dtype)

        model = LumaFluxModel(transformer, vae, siglip_model, **MODEL_HPARAMS)
        model.load_adapters(adapter_path)
        pipe = LumaFluxPipeline(model, device=device, cfg={"train": {"bridge_noise": BRIDGE_NOISE}})
        log.info("LumaFlux: loaded %s on %s (%s)", adapters, device, dtype)
        return (LumaFluxHandle(pipe, adapters, flux_transformer, torch_dtype),)


class LumaFluxSDRToHDR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LUMAFLUX_MODEL",),
                "image": (
                    "IMAGE",
                    {"tooltip": "8-bit SDR (BT.709) frames in [0, 1]. A batch is treated as one video sequence."},
                ),
                "steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Euler steps from t=1 to t=0. The paper uses 8; more costs time without "
                        "improving quality.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "shared_noise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reuse one bridge-noise realization for every frame of the batch "
                        "(video stabilization, on in the paper).",
                    },
                ),
                "rqs_ema": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 0.99,
                        "step": 0.01,
                        "tooltip": "Weight of the previous frame's tone-curve parameters (video stabilization, "
                        "0.8 in the paper). Ignored for a single image.",
                    },
                ),
                "tone_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Scales the learned tone expansion. 1.0 is the paper; 0 leaves the VAE "
                        "decode untouched.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("hdr_pq", "preview")
    OUTPUT_TOOLTIPS = (
        "PQ / BT.2020 signal in [0, 1], mastered at 1000 nits. Save it with the LumaFlux HDR save nodes; "
        "a regular Save Image node would quantize it to 8 bits and drop the HDR tagging.",
        "SDR tone-mapped view of the HDR result, for Preview Image.",
    )
    FUNCTION = "convert"
    CATEGORY = "LumaFlux"
    DESCRIPTION = (
        "Converts SDR frames to HDR (PQ, BT.2020) with the reference LumaFlux pipeline: no prompt, "
        "8 steps, shared noise and tone-curve smoothing across a batch."
    )

    def convert(self, model, image, steps, seed, shared_noise, rqs_ema, tone_strength):
        pipe = model.pipe
        frames = image.movedim(-1, 1).float()
        n = frames.shape[0]
        mult = 2 * pipe.model.vae_scale
        generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
        ema = float(rqs_ema) if n > 1 else 0.0
        pbar = comfy.utils.ProgressBar(n)
        noise = None
        previous = None
        outputs = []
        for i in range(n):
            mm.throw_exception_if_processing_interrupted()
            x, (h, w) = pad_to_multiple(frames[i], mult)
            out = pipe(
                x.unsqueeze(0),
                num_steps=int(steps),
                noise=noise,
                generator=generator,
                previous_spline_params=previous,
                spline_ema=ema,
                tone_strength=float(tone_strength),
            )
            if shared_noise and noise is None:
                noise = out["bridge_noise"]
            previous = out["spline_params"] if ema > 0.0 else None
            outputs.append(out["hdr"][0, :, :h, :w].float())
            pbar.update(1)
        hdr = torch.stack(outputs).movedim(1, -1).contiguous()
        preview = pq_to_sdr_preview(hdr, peak_nits=pipe.model.peak_nits)
        return (hdr, preview)


class LumaFluxHDRPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_pq": ("IMAGE", {"tooltip": "PQ / BT.2020 output of LumaFlux SDR to HDR."}),
                "white_nits": (
                    "FLOAT",
                    {
                        "default": 203.0,
                        "min": 50.0,
                        "max": 1000.0,
                        "step": 1.0,
                        "tooltip": "Luminance shown as SDR white. 203 nits is the BT.2408 reference; lower "
                        "brightens the preview.",
                    },
                ),
                "method": (["reinhard", "clip"], {"default": "reinhard"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "tonemap"
    CATEGORY = "LumaFlux"
    DESCRIPTION = "Tone-maps the PQ output to an SDR image for viewing on an ordinary monitor."

    def tonemap(self, hdr_pq, white_nits, method):
        _require_lumaflux()
        return (pq_to_sdr_preview(hdr_pq, white_nits=float(white_nits), method=method),)


class LumaFluxSaveHDRPNG:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_pq": ("IMAGE", {"tooltip": "PQ / BT.2020 output of LumaFlux SDR to HDR."}),
                "filename_prefix": ("STRING", {"default": "lumaflux/hdr"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LumaFlux"
    DESCRIPTION = (
        "Writes each frame as a 16-bit PNG holding the PQ / BT.2020 code values (signal x 65535), "
        "tagged with a cICP chunk so HDR-aware viewers display it as HDR. Same layout as the frames "
        "the LumaFlux CLI writes."
    )

    def save(self, hdr_pq, filename_prefix="lumaflux/hdr"):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, hdr_pq.shape[2], hdr_pq.shape[1]
        )
        results = []
        for frame in hdr_pq:
            arr = (frame.clamp(0.0, 1.0) * 65535.0).round().cpu().numpy().astype(np.uint16)
            file = f"{filename}_{counter:05}_.png"
            write_png16(os.path.join(full_output_folder, file), arr)
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1
        return {"ui": {"images": results}}


class LumaFluxSaveHDRVideo:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hdr_pq": ("IMAGE", {"tooltip": "PQ / BT.2020 frames from LumaFlux SDR to HDR."}),
                "filename_prefix": ("STRING", {"default": "lumaflux/hdr"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
                "crf": ("INT", {"default": 16, "min": 0, "max": 51, "tooltip": "libx265 quality; lower is larger and better."}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "LumaFlux"
    DESCRIPTION = (
        "Encodes the frames as 10-bit HEVC (yuv420p10le) tagged BT.2020 / PQ with 1000-nit mastering "
        "metadata, the paper's delivery format. Needs an ffmpeg with a 10-bit libx265 on PATH."
    )

    def save(self, hdr_pq, filename_prefix="lumaflux/hdr", fps=24.0, crf=16):
        _require_lumaflux()
        from lumaflux.inference.video import HDRVideoWriter

        n, h, w, _ = hdr_pq.shape
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, w, h
        )
        file = f"{filename}_{counter:05}_.mp4"
        with HDRVideoWriter(os.path.join(full_output_folder, file), h, w, fps=float(fps), crf=int(crf)) as writer:
            for frame in hdr_pq:
                writer.write(frame.movedim(-1, 0).float())
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {
    "LumaFluxLoader": LumaFluxLoader,
    "LumaFluxSDRToHDR": LumaFluxSDRToHDR,
    "LumaFluxHDRPreview": LumaFluxHDRPreview,
    "LumaFluxSaveHDRPNG": LumaFluxSaveHDRPNG,
    "LumaFluxSaveHDRVideo": LumaFluxSaveHDRVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LumaFluxLoader": "LumaFlux Loader",
    "LumaFluxSDRToHDR": "LumaFlux SDR to HDR",
    "LumaFluxHDRPreview": "LumaFlux HDR Preview (tone map)",
    "LumaFluxSaveHDRPNG": "Save HDR PNG (16-bit PQ)",
    "LumaFluxSaveHDRVideo": "Save HDR Video (HEVC 10-bit PQ)",
}
