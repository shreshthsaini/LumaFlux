"""LumaFlux demo: 8-bit SDR image in, PQ/BT.2020 HDR out.

Prompt-free rectified-flow bridge on a frozen FLUX.1-dev backbone (gated; set
HF_TOKEN in the Space secrets). Returns a tone-mapped preview for SDR displays,
the 16-bit PNG holding the HDR signal, and the predicted tone curve.
"""

import os
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import numpy as np
import torch

try:
    import spaces  # ZeroGPU
except ImportError:
    class spaces:  # noqa: N801
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

from huggingface_hub import hf_hub_download
from lumaflux.data.tmo import tmo_bt2446c_gm
from lumaflux.inference.pipeline import LumaFluxPipeline
from lumaflux.models.factory import build_model, load_config
from lumaflux.models.rqs import rqs_apply
from lumaflux.utils.io import save_hdr_png16

HERE = Path(__file__).resolve().parent
CONFIG = os.environ.get("LUMAFLUX_CONFIG", str(HERE / "configs" / "flux_dev.yaml"))
ADAPTERS = os.environ.get("LUMAFLUX_ADAPTERS", "lumaflux-main.safetensors")
LOAD_MODEL = os.environ.get("LOAD_MODEL", "1") == "1"
HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"
MAX_SIDE = int(os.environ.get("MAX_SIDE", "1536"))

pipe = None
mult = 16
load_error = None

if LOAD_MODEL and (HAS_GPU or "tiny" in CONFIG):
    try:
        cfg = load_config(CONFIG)
        model = build_model(cfg, device=DEVICE)
        if "tiny" not in CONFIG:
            path = ADAPTERS if os.path.exists(ADAPTERS) else hf_hub_download("shreshthsaini/LumaFlux", ADAPTERS)
            model.load_adapters(path)
        pipe = LumaFluxPipeline(model, device=DEVICE, cfg=cfg)
        mult = 2 * model.vae_scale
    except Exception:
        load_error = traceback.format_exc()


def _prepare(image: np.ndarray) -> tuple[torch.Tensor, int, int]:
    sdr = torch.from_numpy(np.ascontiguousarray(image[..., :3]).transpose(2, 0, 1)).float() / 255.0
    h, w = sdr.shape[-2:]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        sdr = torch.nn.functional.interpolate(sdr.unsqueeze(0), scale_factor=s, mode="bilinear", align_corners=False, antialias=True).squeeze(0)
        h, w = sdr.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        sdr = torch.nn.functional.pad(sdr.unsqueeze(0), (0, pw, 0, ph), mode="replicate").squeeze(0)
    return sdr, h, w


def _tone_curve_png(params, out_path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        xs = torch.linspace(0, 1, 256).unsqueeze(0)
        ys = rqs_apply(xs, params["widths"][:1], params["heights"][:1], params["derivs"][:1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(xs[0].numpy(), ys[0].numpy(), label="RQS tone field")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="identity")
    ax.set_xlabel("VAE luma Y_out")
    ax.set_ylabel("expanded luma Y_hat")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


@spaces.GPU(duration=120)
def convert(image, num_steps, seed):
    if pipe is None:
        raise gr.Error(
            "Model is not loaded. This Space needs a GPU and an HF_TOKEN secret with access to the "
            "gated FLUX.1-dev weights."
        )
    if image is None:
        raise gr.Error("Upload an SDR image.")
    sdr, h, w = _prepare(image)
    gen = torch.Generator(device=pipe.device).manual_seed(int(seed))
    out = pipe(sdr.unsqueeze(0), num_steps=int(num_steps), generator=gen)
    hdr = out["hdr"][0][:, :h, :w]

    tmp = Path(tempfile.mkdtemp())
    hdr_path = tmp / "lumaflux_hdr_pq2020.png"
    save_hdr_png16(hdr, hdr_path)
    preview = tmo_bt2446c_gm(hdr.unsqueeze(0)).squeeze(0)
    preview_np = (preview.clamp(0, 1) * 255).byte().numpy().transpose(1, 2, 0)
    curve_path = _tone_curve_png(out["spline_params"], tmp / "tone_curve.png") if out["spline_params"] else None
    return preview_np, str(hdr_path), curve_path


DESCRIPTION = """
# LumaFlux: SDR to HDR

Physically-guided inverse tone mapping on a frozen FLUX.1-dev transformer. No text prompt, no strength knob:
the SDR frame is encoded, lightly noised, and transported in 8 steps to an HDR latent, then decoded and
tone-expanded to 10-bit PQ / BT.2020 with a rational-quadratic spline predicted per image.

Upload an 8-bit SDR image. You get an SDR preview of the HDR result (BT.2446c tone-mapped, since most
screens cannot show PQ), the 16-bit PNG carrying the actual HDR signal, and the predicted tone curve.
Inputs longer than 1536 px on a side are downscaled.

[Paper (arXiv 2604.02787)](https://arxiv.org/abs/2604.02787) · [Code](https://github.com/shreshthsaini/LumaFlux) · [Weights](https://huggingface.co/shreshthsaini/LumaFlux) · [Project page](https://shreshthsaini.github.io/LumaFlux/)
"""

with gr.Blocks(title="LumaFlux SDR to HDR") as demo:
    gr.Markdown(DESCRIPTION)
    if pipe is None:
        why = "no GPU is attached" if not HAS_GPU else "the model failed to load"
        gr.Markdown(
            f"**The model is not loaded because {why}.** Conversion needs ZeroGPU (or a larger GPU) and an "
            "`HF_TOKEN` secret that has accepted the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) license."
            + (f"\n\n```\n{load_error[-1500:]}\n```" if load_error else "")
        )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="SDR input (BT.709, 8-bit)", type="numpy")
            steps = gr.Slider(1, 24, value=8, step=1, label="Transport steps")
            seed = gr.Number(value=0, precision=0, label="Seed")
            btn = gr.Button("Convert to HDR", variant="primary")
        with gr.Column():
            preview = gr.Image(label="HDR result, SDR preview (BT.2446c)")
            hdr_file = gr.File(label="HDR output: 16-bit PNG, PQ / BT.2020")
            curve = gr.Image(label="Predicted RQS tone curve")
    btn.click(convert, [inp, steps, seed], [preview, hdr_file, curve])
    gr.Markdown(
        "```bibtex\n@article{saini2026lumaflux,\n  title   = {LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided Diffusion Transformers},\n"
        "  author  = {Saini, Shreshth and others},\n  journal = {arXiv preprint arXiv:2604.02787},\n  year    = {2026}\n}\n```"
    )

if __name__ == "__main__":
    demo.launch()
