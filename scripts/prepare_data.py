#!/usr/bin/env python3
"""Build a LumaFlux training corpus from locally available source media.

Example spec (YAML)::

    out_dir: data/curated
    peak_nits: 1000
    tmos: [ocio_filmic, bt2446c_gm, hard_clip_gm, bt2446a, reinhard,
           youtube_logc, bt2390_eetf_gm, gamma_clip]
    crf_levels: [23, 31, 39]
    sources:
      - name: chug
        path: data/raw/chug
        category: ugc
        format: hlg2020
        fps: 1
      - name: live_tmhdr
        path: data/raw/live-tmhdr/reference
        category: pgc
        format: pq2020
        fps: 60
        expert_sdr_path: data/raw/live-tmhdr/expert-sdr
      - name: hdrtv1k
        source_dataset: HDRTV1K
        mode: native_pairs
        path: data/raw/hdrtv1k/train-hdr
        sdr_path: data/raw/hdrtv1k/train-sdr

Usage: python scripts/prepare_data.py --spec configs/data/example.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumaflux.data.curate import CurationConfig, SourceSpec, curate  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Build a LumaFlux training corpus from local media.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--spec",
        default="configs/data/example.yaml",
        help="YAML file describing raw SDR and HDR sources",
    )
    p.add_argument(
        "--corpus-output",
        default=None,
        help="training corpus directory; defaults to out_dir in the YAML file",
    )
    p.add_argument("--workers", type=int, default=None, help="number of curation workers")
    args = p.parse_args(argv)
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    sources = [SourceSpec(**source) for source in spec.pop("sources")]
    if args.corpus_output is not None:
        spec["out_dir"] = args.corpus_output
    if args.workers is not None:
        spec["workers"] = args.workers
    cfg = CurationConfig(sources=sources, **spec)
    manifest = curate(cfg)
    print(f"training manifest written: {manifest}")


if __name__ == "__main__":
    main()
