"""Gradio demo: upload an SDR image, get the HDR reconstruction.

Outputs:
  * a 16-bit PNG holding the PQ/BT.2020 HDR signal (for HDR-capable tools),
  * an SDR preview of the HDR result (BT.2446c tone-mapped) for viewing on
    standard displays,
  * the predicted RQS tone curve.

The "strength" slider exposes the tone-expansion control recommended in the
paper's broader-impact discussion.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..data.tmo import tmo_bt2446c_gm
from ..models.factory import build_model, load_config
from ..models.rqs import rqs_apply
from ..inference.pipeline import LumaFluxPipeline
from ..utils.io import save_hdr_png16


def build_demo(pipe: LumaFluxPipeline, mult: int):
    import gradio as gr

    def convert(image: np.ndarray, num_steps: int, strength: float, seed: int):
        if image is None:
            return None, None, None
        sdr = torch.from_numpy(image[..., :3].transpose(2, 0, 1)).float() / 255.0
        h, w = sdr.shape[-2:]
        ph, pw = (-h) % mult, (-w) % mult
        if ph or pw:
            sdr = torch.nn.functional.pad(
                sdr.unsqueeze(0), (0, pw, 0, ph), mode="replicate").squeeze(0)
        gen = torch.Generator(device=pipe.device).manual_seed(int(seed))
        out = pipe(sdr.unsqueeze(0), num_steps=int(num_steps),
                   strength=float(strength), generator=gen)
        hdr = out["hdr"][0][:, :h, :w]

        tmp = Path(tempfile.mkdtemp())
        hdr_path = tmp / "lumaflux_hdr_pq2020.png"
        save_hdr_png16(hdr, hdr_path)

        preview = tmo_bt2446c_gm(hdr.unsqueeze(0)).squeeze(0)
        preview_np = (preview.clamp(0, 1) * 255).byte().numpy().transpose(1, 2, 0)

        # Plot the learned tone curve.
        model = pipe.model
        with torch.no_grad():
            z = model.encode_image(sdr.unsqueeze(0).to(pipe.device))
            widths, heights, derivs = model.rqs.spline_params(z)
            xs = torch.linspace(0, 1, 256).unsqueeze(0)
            ys = rqs_apply(xs, widths[:1].cpu(), heights[:1].cpu(), derivs[:1].cpu())
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(xs[0].numpy(), ys[0].numpy(), label="RQS tone field")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="identity")
        ax.set_xlabel("VAE luma Y_out")
        ax.set_ylabel("expanded luma Y_hat")
        ax.legend()
        fig.tight_layout()
        curve_path = tmp / "tone_curve.png"
        fig.savefig(curve_path, dpi=120)
        plt.close(fig)
        return preview_np, str(hdr_path), str(curve_path)

    with gr.Blocks(title="LumaFlux SDR->HDR") as demo:
        gr.Markdown(
            "# LumaFlux: SDR → HDR\n"
            "Physically-guided inverse tone mapping with a frozen Flux DiT. "
            "Upload an 8-bit SDR frame; receive a 10-bit PQ/BT.2020 HDR "
            "reconstruction (16-bit PNG) plus an SDR preview."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="SDR input (BT.709)")
                steps = gr.Slider(1, 60, value=40, step=1, label="ODE steps")
                strength = gr.Slider(0.0, 2.0, value=1.0, step=0.05,
                                     label="Tone expansion strength")
                seed = gr.Number(value=0, precision=0, label="Seed")
                btn = gr.Button("Convert to HDR", variant="primary")
            with gr.Column():
                preview = gr.Image(label="HDR result (SDR preview via BT.2446c)")
                hdr_file = gr.File(label="HDR output: 16-bit PNG, PQ/BT.2020")
                curve = gr.Image(label="Predicted RQS tone curve")
        btn.click(convert, [inp, steps, strength, seed], [preview, hdr_file, curve])
    return demo


def main(argv=None):
    p = argparse.ArgumentParser(description="LumaFlux Gradio demo")
    p.add_argument("--config", required=True)
    p.add_argument("--adapters", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    model = build_model(cfg, device=args.device)
    if args.adapters:
        model.load_adapters(args.adapters)
    pipe = LumaFluxPipeline(model, device=args.device)
    demo = build_demo(pipe, mult=2 * model.vae_scale)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
