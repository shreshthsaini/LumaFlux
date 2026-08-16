"""CLI: convert an SDR video or frame directory to HDR (PQ/BT.2020)."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from tqdm import tqdm

from ..models.factory import build_model, load_config
from ..utils.io import extract_video_frames, ffmpeg_available, load_sdr_png
from .pipeline import LumaFluxPipeline
from .video import HDRVideoWriter, write_hdr_frames

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _pad_to_multiple(x: torch.Tensor, mult: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = torch.nn.functional.pad(x.unsqueeze(0), (0, pw, 0, ph), mode="replicate").squeeze(0)
    return x, (h, w)


def main(argv=None):
    p = argparse.ArgumentParser(description="LumaFlux SDR->HDR inference")
    p.add_argument("--config", required=True)
    p.add_argument("--adapters", default=None, help="trained adapter .safetensors")
    p.add_argument("--input", required=True, help="SDR video file or frame directory")
    p.add_argument("--output", required=True, help="output .mp4/.mov or directory for PNG frames")
    p.add_argument("--steps", "--num-steps", dest="steps", type=int, default=8)
    p.add_argument(
        "--shared-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reuse one bridge-noise tensor across frames",
    )
    p.add_argument(
        "--rqs-ema",
        type=float,
        default=None,
        help="prior-frame weight for RQS parameter smoothing; zero disables it",
    )
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    model = build_model(cfg, device=args.device)
    if args.adapters:
        model.load_adapters(args.adapters)
    pipe = LumaFluxPipeline(model, device=args.device, cfg=cfg)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    inp = Path(args.input)
    with tempfile.TemporaryDirectory() as td:
        is_video_input = False
        if inp.is_dir():
            frame_paths = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        elif inp.suffix.lower() in IMAGE_EXTS:
            frame_paths = [inp]
        else:
            is_video_input = True
            if not ffmpeg_available():
                raise SystemExit("ffmpeg required to decode video inputs")
            frame_paths = extract_video_frames(inp, td, bit_depth=8, max_frames=args.max_frames)
        if args.max_frames:
            frame_paths = frame_paths[: args.max_frames]
        if not frame_paths:
            raise SystemExit(f"no frames found in {inp}")

        is_sequence = is_video_input or len(frame_paths) > 1
        shared_noise_enabled = is_sequence if args.shared_noise is None else args.shared_noise
        spline_ema = (0.8 if is_sequence else 0.0) if args.rqs_ema is None else args.rqs_ema
        if not 0.0 <= spline_ema < 1.0:
            raise SystemExit("--rqs-ema must be in [0,1)")

        mult = 2 * model.vae_scale

        def convert_frames():
            shared_noise = None
            previous_spline_params = None
            for frame_path in tqdm(frame_paths, desc="lumaflux"):
                sdr = load_sdr_png(frame_path)
                sdr, (height, width) = _pad_to_multiple(sdr, mult)
                out = pipe(
                    sdr.unsqueeze(0),
                    num_steps=args.steps,
                    noise=shared_noise,
                    generator=gen,
                    previous_spline_params=previous_spline_params,
                    spline_ema=spline_ema,
                )
                if shared_noise_enabled and shared_noise is None:
                    shared_noise = out["bridge_noise"]
                previous_spline_params = out["spline_params"] if spline_ema > 0.0 else None
                yield out["hdr"][0][:, :height, :width]

        out_path = Path(args.output)
        if out_path.suffix.lower() in {".mp4", ".mov", ".mkv"}:
            first_input = load_sdr_png(frame_paths[0])
            height, width = first_input.shape[-2:]
            with HDRVideoWriter(
                out_path,
                height,
                width,
                fps=args.fps,
                peak_nits=model.peak_nits,
            ) as writer:
                for frame in convert_frames():
                    writer.write(frame)
            print(f"wrote HDR video: {out_path}")
        else:
            paths = write_hdr_frames(convert_frames(), out_path)
            print(f"wrote {len(paths)} HDR frames to {out_path}")


if __name__ == "__main__":
    main()
