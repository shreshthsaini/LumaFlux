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
from .video import write_hdr_frames, write_hdr_video

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
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    model = build_model(cfg, device=args.device)
    if args.adapters:
        model.load_adapters(args.adapters)
    pipe = LumaFluxPipeline(model, device=args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    inp = Path(args.input)
    with tempfile.TemporaryDirectory() as td:
        if inp.is_dir():
            frame_paths = sorted(p for p in inp.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        else:
            if not ffmpeg_available():
                raise SystemExit("ffmpeg required to decode video inputs")
            frame_paths = extract_video_frames(inp, td, bit_depth=8, max_frames=args.max_frames)
        if args.max_frames:
            frame_paths = frame_paths[: args.max_frames]
        if not frame_paths:
            raise SystemExit(f"no frames found in {inp}")

        mult = 2 * model.vae_scale
        hdr_frames = []
        for fp in tqdm(frame_paths, desc="lumaflux"):
            sdr = load_sdr_png(fp)
            sdr, (h, w) = _pad_to_multiple(sdr, mult)
            out = pipe(sdr.unsqueeze(0), num_steps=args.num_steps,
                       strength=args.strength, generator=gen)
            hdr_frames.append(out["hdr"][0][:, :h, :w])

    out_path = Path(args.output)
    if out_path.suffix.lower() in {".mp4", ".mov", ".mkv"}:
        write_hdr_video(hdr_frames, out_path, fps=args.fps)
        print(f"wrote HDR video: {out_path}")
    else:
        paths = write_hdr_frames(hdr_frames, out_path)
        print(f"wrote {len(paths)} HDR frames to {out_path}")


if __name__ == "__main__":
    main()
