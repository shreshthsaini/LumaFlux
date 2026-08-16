#!/usr/bin/env python3
"""Run the registered LumaFlux training configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train LumaFlux adapters with the registered configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/train/main.yaml",
        help="model, data, and training configuration",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="training manifest; overrides data.manifest in the configuration",
    )
    parser.add_argument("--output-dir", default="runs/lumaflux", help="checkpoints and logs")
    parser.add_argument("--resume", default=None, help="full checkpoint to resume")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="optional training-length override",
    )
    args = parser.parse_args(argv)

    forwarded = ["--config", args.config, "--output-dir", args.output_dir]
    if args.manifest is not None:
        forwarded.extend(("--manifest", args.manifest))
    if args.resume is not None:
        forwarded.extend(("--resume", args.resume))
    if args.max_steps is not None:
        forwarded.extend(("--max-steps", str(args.max_steps)))

    from lumaflux.training.train import main as train_main

    train_main(forwarded)


if __name__ == "__main__":
    main()
