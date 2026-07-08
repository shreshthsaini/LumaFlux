"""Benchmark runner: evaluate a method over a curated manifest (Sec. 6).

Walks a manifest (Luma-Eval / HDRTV1K / HDRTV4K curated to the unified
PQ-BT.2020 format), runs the model (or scores precomputed outputs) per
SDR variant, and reports per-TMO/per-CRF breakdowns as CSV + markdown
(the layout of Tables 1-2 in the paper).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from ..data.dataset import load_manifest
from ..utils.io import load_hdr_png16, load_sdr_png
from .deitp import delta_e_itp
from .hdr_lpips import hdr_lpips
from .hdrvdp3 import hdr_vdp3
from .metrics import pu21_psnr, pu21_psnr_y, pu21_ssim

METRIC_COLUMNS = ["psnr", "psnr_y", "ssim", "deitp", "hdr_lpips", "hdr_vdp3"]


def score_pair(pred_pq: torch.Tensor, target_pq: torch.Tensor,
               with_optional: bool = True) -> dict[str, float | None]:
    row: dict[str, float | None] = {
        "psnr": pu21_psnr(pred_pq, target_pq).item(),
        "psnr_y": pu21_psnr_y(pred_pq, target_pq).item(),
        "ssim": pu21_ssim(pred_pq, target_pq).item(),
        "deitp": delta_e_itp(pred_pq, target_pq).item(),
    }
    if with_optional:
        lp = hdr_lpips(pred_pq, target_pq)
        row["hdr_lpips"] = lp.item() if lp is not None else None
        row["hdr_vdp3"] = hdr_vdp3(pred_pq, target_pq)
    else:
        row["hdr_lpips"] = row["hdr_vdp3"] = None
    return row


def evaluate_manifest(
    manifest: str | Path,
    predict_fn,
    variants: list[str] | None = None,
    max_records: int | None = None,
    with_optional: bool = True,
) -> list[dict]:
    """``predict_fn(sdr) -> hdr_pq`` maps a (3,H,W) SDR frame to PQ/BT.2020."""
    root = Path(manifest).parent
    records = load_manifest(manifest)
    if max_records:
        records = records[:max_records]
    rows = []
    for rec in tqdm(records, desc="eval"):
        target = load_hdr_png16(root / rec["hdr"])
        for variant, rel in rec["sdr_variants"].items():
            if variants and variant not in variants:
                continue
            sdr = load_sdr_png(root / rel)
            pred = predict_fn(sdr)
            if pred.shape[-2:] != target.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(0), size=target.shape[-2:], mode="bilinear",
                    align_corners=False).squeeze(0)
            row = {"id": rec["id"], "variant": variant,
                   "category": rec.get("category", "pgc")}
            row.update(score_pair(pred, target, with_optional=with_optional))
            rows.append(row)
    return rows


def aggregate(rows: list[dict], key: str = "variant") -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    groups["ALL"] = rows
    agg = {}
    for name, rs in groups.items():
        agg[name] = {}
        for m in METRIC_COLUMNS:
            vals = [r[m] for r in rs if r.get(m) is not None]
            if vals:
                agg[name][m] = sum(vals) / len(vals)
    return agg


def to_markdown(agg: dict[str, dict[str, float]]) -> str:
    cols = [c for c in METRIC_COLUMNS if any(c in v for v in agg.values())]
    header = "| Variant | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for name in sorted(agg):
        vals = [f"{agg[name][c]:.3f}" if c in agg[name] else "-" for c in cols]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="LumaFlux benchmark evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--adapters", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", default="eval_results")
    p.add_argument("--num-steps", type=int, default=40)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--no-optional-metrics", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    from ..inference.pipeline import LumaFluxPipeline
    from ..models.factory import build_model, load_config

    cfg = load_config(args.config)
    model = build_model(cfg, device=args.device)
    if args.adapters:
        model.load_adapters(args.adapters)
    pipe = LumaFluxPipeline(model, device=args.device)
    mult = 2 * model.vae_scale

    def predict(sdr: torch.Tensor) -> torch.Tensor:
        h, w = sdr.shape[-2:]
        ph, pw = (-h) % mult, (-w) % mult
        if ph or pw:
            sdr = torch.nn.functional.pad(
                sdr.unsqueeze(0), (0, pw, 0, ph), mode="replicate").squeeze(0)
        out = pipe(sdr.unsqueeze(0), num_steps=args.num_steps)
        return out["hdr"][0][:, :h, :w]

    rows = evaluate_manifest(
        args.manifest, predict, variants=args.variants,
        max_records=args.max_records, with_optional=not args.no_optional_metrics,
    )
    out = Path(args.output_dir)
    save_csv(rows, out / "per_frame.csv")
    agg = aggregate(rows)
    md = to_markdown(agg)
    (out / "summary.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
