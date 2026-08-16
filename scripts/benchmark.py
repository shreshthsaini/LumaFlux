#!/usr/bin/env python3
"""Score a Luma-Eval predictions directory and print its summary table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score paired PQ BT.2020 predictions with the Luma-Eval protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="directory containing predictions_manifest.jsonl and paired PNG16 files",
    )
    parser.add_argument(
        "--output-dir",
        default="results/luma_eval",
        help="CSV and Markdown score tables",
    )
    parser.add_argument(
        "--video-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute temporal metrics for records grouped as video",
    )
    parser.add_argument(
        "--optional-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run installed optional metric backends and report unavailable ones",
    )
    parser.add_argument("--max-records", type=int, default=None, help="optional scoring limit")
    args = parser.parse_args(argv)

    predictions_dir = Path(args.predictions_dir)
    predictions_manifest = predictions_dir / "predictions_manifest.jsonl"
    if not predictions_manifest.is_file():
        parser.error(f"missing {predictions_manifest}")

    forwarded = [
        "--predictions-manifest",
        str(predictions_manifest),
        "--output-dir",
        args.output_dir,
    ]
    if args.video_metrics:
        forwarded.append("--video-mode")
    if not args.optional_metrics:
        forwarded.append("--no-optional-metrics")
    if args.max_records is not None:
        forwarded.extend(("--max-records", str(args.max_records)))

    from lumaflux.evaluation.benchmark import main as benchmark_main

    benchmark_main(forwarded)


if __name__ == "__main__":
    main()
