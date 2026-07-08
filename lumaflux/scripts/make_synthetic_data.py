#!/usr/bin/env python3
"""Generate tiny synthetic HDR clips for smoke tests and CI.

Creates procedurally-rendered PQ/BT.2020 frames (gradients, gaussian
highlight blobs, color sweeps - content with genuine >100-nit structure)
laid out like a real HDR source tree, so the curation pipeline can be
exercised end-to-end without downloading datasets.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumaflux.color.transfer import pq_oetf_nits  # noqa: E402
from lumaflux.utils.io import save_hdr_png16  # noqa: E402


def render_frame(t: float, size: int, scene_seed: int, peak_nits: float = 1000.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(scene_seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, size), torch.linspace(0, 1, size), indexing="ij"
    )
    # Diffuse base: slowly varying color gradient (~5-80 nits).
    base_hue = torch.rand(3, generator=g) * 0.7 + 0.15
    base = torch.stack([
        (5 + 75 * (0.5 + 0.5 * torch.sin(2 * math.pi * (xx + 0.3 * t + h))))
        for h in base_hue
    ])
    # Moving specular highlights (up to peak_nits).
    n_blobs = 3
    cx = torch.rand(n_blobs, generator=g)
    cy = torch.rand(n_blobs, generator=g)
    vel = (torch.rand(n_blobs, 2, generator=g) - 0.5) * 0.4
    color = torch.rand(n_blobs, 3, generator=g) * 0.5 + 0.5
    for i in range(n_blobs):
        px = (cx[i] + vel[i, 0] * t) % 1.0
        py = (cy[i] + vel[i, 1] * t) % 1.0
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        blob = torch.exp(-d2 / 0.002)
        base += peak_nits * color[i].view(3, 1, 1) * blob
    # Fine texture so gradients/spectra are non-trivial.
    noise = torch.rand(3, size, size, generator=g)
    base = base * (0.9 + 0.2 * noise)
    return pq_oetf_nits(base.clamp(0.0, peak_nits))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/synthetic")
    p.add_argument("--num-scenes", type=int, default=2)
    p.add_argument("--frames-per-scene", type=int, default=3)
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()

    out = Path(args.out_dir)
    for cat, off in (("pgc", 0), ("ugc", 100)):
        for s in range(args.num_scenes):
            scene_dir = out / cat / f"scene{s:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            for f in range(args.frames_per_scene):
                frame = render_frame(f / max(1, args.frames_per_scene - 1),
                                     args.size, scene_seed=off + s)
                save_hdr_png16(frame, scene_dir / f"frame_{f:04d}.png")
    print(f"synthetic HDR frames written to {out}")


if __name__ == "__main__":
    main()
