"""Training-corpus curation (Sec. 5): frame sampling + SDR variant synthesis.

Builds a JSONL manifest of SDR-HDR pairs from a directory tree of HDR
sources. Sources are declared in a YAML/JSON spec::

    sources:
      - name: hidrovqa            # PGC, 1 fps tier
        path: /data/hidrovqa      # videos or per-video frame dirs
        category: pgc
        format: pq2020
        fps: 1
      - name: chug                # UGC, 1 fps tier
        path: /data/chug
        category: ugc
        format: hlg2020
        fps: 1
      - name: live_tmhdr_expert   # expert-graded SDR available
        path: /data/live_tmhdr
        category: pgc
        format: pq2020
        fps: 60
        expert_sdr_path: /data/live_tmhdr_sdr_expert

Each sampled HDR frame is normalized to PQ/BT.2020 (1,000-nit mastering),
saved as 16-bit PNG, and paired with SDR variants from the TMO x CRF grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from ..utils.io import (
    extract_video_frames,
    ffmpeg_available,
    load_hdr_png16,
    load_sdr_png,
    save_hdr_png16,
    save_sdr_png,
)
from .degrade import CRF_LEVELS, degrade_chain
from .normalize import normalize_to_pq2020
from .tmo import TMO_REGISTRY

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".y4m"}
IMAGE_EXTS = {".png", ".tif", ".tiff", ".exr"}


@dataclass
class SourceSpec:
    name: str
    path: str
    category: str = "pgc"  # pgc | ugc
    format: str = "pq2020"
    fps: float = 1.0
    expert_sdr_path: str | None = None
    max_frames_per_video: int | None = None


@dataclass
class CurationConfig:
    sources: list[SourceSpec]
    out_dir: str = "data/curated"
    peak_nits: float = 1000.0
    tmos: list[str] = field(default_factory=lambda: sorted(TMO_REGISTRY))
    crf_levels: list[int] = field(default_factory=lambda: list(CRF_LEVELS))
    seed: int = 0


def _iter_source_frames(spec: SourceSpec, work_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield (frame_id, png_path) of raw HDR frames for one source."""
    root = Path(spec.path)
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if videos:
        if not ffmpeg_available():
            raise RuntimeError(f"Source {spec.name} contains videos but ffmpeg is missing")
        for vid in videos:
            frames = extract_video_frames(
                vid, work_dir / spec.name / vid.stem, fps=spec.fps,
                bit_depth=16, max_frames=spec.max_frames_per_video,
            )
            for fr in frames:
                yield f"{vid.stem}_{fr.stem}", fr
    else:
        for fr in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            yield f"{fr.parent.name}_{fr.stem}", fr


def curate(cfg: CurationConfig) -> Path:
    """Run the full curation pipeline; returns the manifest path."""
    out = Path(cfg.out_dir)
    (out / "hdr").mkdir(parents=True, exist_ok=True)
    (out / "sdr").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    records = []
    for spec in cfg.sources:
        for frame_id, fr_path in tqdm(
            _iter_source_frames(spec, out / "_work"), desc=f"curate:{spec.name}"
        ):
            x = load_hdr_png16(fr_path)
            x_pq = normalize_to_pq2020(x, source_format=spec.format, peak_nits=cfg.peak_nits)
            hdr_rel = f"hdr/{spec.name}_{frame_id}.png"
            save_hdr_png16(x_pq, out / hdr_rel)
            variants = {}
            for tmo in cfg.tmos:
                for crf in cfg.crf_levels:
                    sdr = degrade_chain(x_pq, tmo, crf=crf, peak_nits=cfg.peak_nits)
                    rel = f"sdr/{spec.name}_{frame_id}__{tmo}_crf{crf}.png"
                    save_sdr_png(sdr, out / rel)
                    variants[f"{tmo}_crf{crf}"] = rel
            if spec.expert_sdr_path is not None:
                expert = Path(spec.expert_sdr_path) / fr_path.name
                if expert.exists():
                    sdr = load_sdr_png(expert)
                    rel = f"sdr/{spec.name}_{frame_id}__expert.png"
                    save_sdr_png(sdr, out / rel)
                    variants["expert"] = rel
            records.append(
                {
                    "id": f"{spec.name}_{frame_id}",
                    "source": spec.name,
                    "category": spec.category,
                    "hdr": hdr_rel,
                    "sdr_variants": variants,
                }
            )
    with open(manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return manifest_path
