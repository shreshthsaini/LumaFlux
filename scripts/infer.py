#!/usr/bin/env python3
"""Convert an SDR image, frame directory, or video to HDR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run prompt-free LumaFlux SDR-to-HDR inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="SDR image, frame directory, or video")
    parser.add_argument(
        "--output",
        required=True,
        help="directory for PNG16 frames, or an MP4, MOV, or MKV video",
    )
    parser.add_argument(
        "--adapters",
        required=True,
        help=(
            "adapter file from https://huggingface.co/shreshthsaini/LumaFlux"
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/model/flux_dev.yaml",
        help="model configuration",
    )
    parser.add_argument("--steps", type=int, default=8, help="transport steps per frame")
    parser.add_argument(
        "--shared-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reuse one noise realization across a sequence; automatic for video",
    )
    parser.add_argument(
        "--rqs-ema",
        type=float,
        default=None,
        help="prior-frame spline weight; automatic for video and zero for one image",
    )
    parser.add_argument("--fps", type=float, default=24.0, help="output video frame rate")
    parser.add_argument("--device", default=None, help="Torch device; selected automatically")
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="integer controlling repeatable noise",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="optional frame limit")
    args = parser.parse_args(argv)

    forwarded = [
        "--config",
        args.config,
        "--adapters",
        args.adapters,
        "--input",
        args.input,
        "--output",
        args.output,
        "--steps",
        str(args.steps),
        "--fps",
        str(args.fps),
        "--seed",
        str(args.random_state),
    ]
    if args.shared_noise is not None:
        forwarded.append("--shared-noise" if args.shared_noise else "--no-shared-noise")
    if args.rqs_ema is not None:
        forwarded.extend(("--rqs-ema", str(args.rqs_ema)))
    if args.device is not None:
        forwarded.extend(("--device", args.device))
    if args.max_frames is not None:
        forwarded.extend(("--max-frames", str(args.max_frames)))

    from lumaflux.inference.cli import main as infer_main

    infer_main(forwarded)


if __name__ == "__main__":
    main()
