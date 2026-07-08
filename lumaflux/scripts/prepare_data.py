#!/usr/bin/env python3
"""Curate the SDR-HDR training corpus from a source spec (Sec. 5).

Example spec (YAML)::

    out_dir: data/curated
    peak_nits: 1000
    tmos: [ocio_v2, bt2446c_gm, hard_clip_gm, bt2446a, reinhard,
           youtube_logc, bt2390_eetf_gm, gamma_clip]
    crf_levels: [23, 31, 39]
    sources:
      - name: hidrovqa
        path: /data/hidrovqa
        category: pgc
        format: pq2020
        fps: 1
      - name: chug
        path: /data/chug
        category: ugc
        format: hlg2020
        fps: 1
      - name: live_tmhdr
        path: /data/live_tmhdr
        category: pgc
        format: pq2020
        fps: 60
        expert_sdr_path: /data/live_tmhdr_expert_sdr

Usage: python scripts/prepare_data.py --spec spec.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumaflux.data.curate import CurationConfig, SourceSpec, curate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="YAML curation spec")
    args = p.parse_args()
    with open(args.spec) as f:
        spec = yaml.safe_load(f)
    sources = [SourceSpec(**s) for s in spec.pop("sources")]
    cfg = CurationConfig(sources=sources, **spec)
    manifest = curate(cfg)
    print(f"manifest written: {manifest}")


if __name__ == "__main__":
    main()
