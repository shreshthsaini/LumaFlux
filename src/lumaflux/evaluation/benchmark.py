"""Benchmark runner for frame and video HDR evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm

from ..data.dataset import (
    SdrHdrShards,
    collate_sdr_hdr,
    is_shard_manifest,
    load_manifest,
)
from ..utils.io import load_hdr_png16, load_sdr_png, save_hdr_png16, save_sdr_png
from .backends import EvalBackendUnavailable
from .deitp import delta_e_itp
from .exposure import score_exposure_calibrated_content
from .fr_hidrovqa import fr_hidrovqa, probe_fr_hidrovqa
from .hdr_lpips import hdr_lpips, probe_hdr_lpips
from .hdrvdp3 import hdr_vdp3, probe_hdrvdp3
from .metrics import pu21_psnr, pu21_psnr_y, pu21_ssim
from .temporal import TemporalAccumulator

CORE_METRICS = ["psnr", "psnr_y", "ssim", "deitp"]
EXPOSURE_CALIBRATED_METRICS = [
    f"exposure_calibrated_{metric}" for metric in CORE_METRICS
]
OPTIONAL_METRICS = ["hdr_lpips", "hdr_vdp3", "fr_hidrovqa"]
TEMPORAL_METRICS = ["flicker_raw", "flicker_reference", "flicker_excess"]
FRAME_METRIC_COLUMNS = EXPOSURE_CALIBRATED_METRICS + CORE_METRICS + OPTIONAL_METRICS
METRIC_COLUMNS = FRAME_METRIC_COLUMNS + TEMPORAL_METRICS

Probe = Callable[[], None]
Availability = dict[str, str | None]


@dataclass
class BenchmarkResult:
    frame_rows: list[dict]
    video_rows: list[dict]
    overall_rows: list[dict]
    availability: Availability


class PredictionWriter:
    """Persist model inputs, references, and PNG16 predictions for rescoring."""

    def __init__(self, prediction_dir: str | Path) -> None:
        self.prediction_dir = Path(prediction_dir)
        self.root = self.prediction_dir.parent
        self.input_dir = self.root / "inputs"
        self.reference_dir = self.root / "references"
        for directory in (self.prediction_dir, self.input_dir, self.reference_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    @staticmethod
    def _safe_variant(variant: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", variant).strip("._") or "variant"

    def write(
        self,
        *,
        index: int,
        row: dict,
        record: dict,
        sdr: torch.Tensor,
        prediction: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        stem = f"{index:08d}_{self._safe_variant(str(row['variant']))}"
        input_path = self.input_dir / f"{stem}.png"
        prediction_path = self.prediction_dir / f"{stem}.png"
        reference_path = self.reference_dir / f"{stem}.png"
        save_sdr_png(sdr, input_path)
        save_hdr_png16(prediction, prediction_path)
        save_hdr_png16(reference, reference_path)
        self.records.append(
            {
                **{
                    key: row[key]
                    for key in ("id", "video", "frame_index", "variant", "category")
                },
                **{
                    key: record[key]
                    for key in ("content_id", "track", "tmo", "crf", "source_dataset")
                    if record.get(key) is not None
                },
                "method": "lumaflux_main",
                "input": str(input_path.relative_to(self.root)),
                "prediction": str(prediction_path.relative_to(self.root)),
                "reference": str(reference_path.relative_to(self.root)),
            }
        )

    def finish(self) -> Path:
        path = self.root / "predictions_manifest.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in self.records)
        )
        temporary.replace(path)
        return path


def probe_metric_backends(
    *,
    include_optional: bool = True,
    video_mode: bool = False,
    probes: dict[str, Probe] | None = None,
) -> Availability:
    """Probe optional metrics once and retain an explicit skip reason."""
    availability: Availability = {name: None for name in CORE_METRICS}
    probe_map = probes or {
        "hdr_lpips": probe_hdr_lpips,
        "hdr_vdp3": probe_hdrvdp3,
        "fr_hidrovqa": probe_fr_hidrovqa,
    }
    for name in OPTIONAL_METRICS:
        if not include_optional:
            availability[name] = "disabled by --no-optional-metrics"
            continue
        try:
            probe_map[name]()
        except EvalBackendUnavailable as exc:
            availability[name] = exc.reason
        else:
            availability[name] = None
    for name in TEMPORAL_METRICS:
        availability[name] = None if video_mode else "requires --video-mode"
    return availability


def score_pair(
    pred_pq: torch.Tensor,
    target_pq: torch.Tensor,
    with_optional: bool = True,
    available_metrics: set[str] | None = None,
) -> dict[str, float | None]:
    """Score one paired PQ/BT.2020 frame without swallowing backend errors."""
    enabled = set(OPTIONAL_METRICS) if available_metrics is None else available_metrics
    row: dict[str, float | None] = {
        "psnr": pu21_psnr(pred_pq, target_pq).item(),
        "psnr_y": pu21_psnr_y(pred_pq, target_pq).item(),
        "ssim": pu21_ssim(pred_pq, target_pq).item(),
        "deitp": delta_e_itp(pred_pq, target_pq).item(),
    }
    optional_scores: dict[str, Callable[[], float]] = {
        "hdr_lpips": lambda: hdr_lpips(pred_pq, target_pq).item(),
        "hdr_vdp3": lambda: hdr_vdp3(pred_pq, target_pq),
        "fr_hidrovqa": lambda: fr_hidrovqa(pred_pq, target_pq).item(),
    }
    for name, scorer in optional_scores.items():
        row[name] = scorer() if with_optional and name in enabled else None
    return row


def _score_core_pair(pred_pq: torch.Tensor, target_pq: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": pu21_psnr(pred_pq, target_pq).item(),
        "psnr_y": pu21_psnr_y(pred_pq, target_pq).item(),
        "ssim": pu21_ssim(pred_pq, target_pq).item(),
        "deitp": delta_e_itp(pred_pq, target_pq).item(),
    }


def _is_expert_track(record: dict, variant: str) -> bool:
    track = record.get("track", record.get("tmo", variant))
    return str(track).lower() == "expert"


def _content_id(record: dict) -> str:
    for field in ("content_id", "video_id", "video", "sequence_id"):
        if record.get(field) is not None:
            return str(record[field])
    return str(record.get("id", "unknown"))


def _add_expert_calibrated_scores(
    groups: dict[tuple[str, str], list[tuple[dict, torch.Tensor, torch.Tensor]]],
) -> None:
    # LIVE-TMHDR expert grades have free exposure, so Luma-Eval fits one gain
    # per content and method. Synthetic tracks never enter this wrapper.
    for entries in groups.values():
        predictions = [prediction for _, prediction, _ in entries]
        targets = [target for _, _, target in entries]
        gain, calibrated = score_exposure_calibrated_content(
            predictions,
            targets,
            _score_core_pair,
        )
        for (row, _, _), scores in zip(entries, calibrated):
            row["exposure_gain"] = gain
            row["headline_protocol"] = "exposure_calibrated"
            row.update(
                {
                    f"exposure_calibrated_{metric}": value
                    for metric, value in scores.items()
                }
            )


def _video_id(record: dict, *, required: bool) -> str:
    for field in ("video_id", "video", "sequence_id"):
        if record.get(field) is not None:
            return str(record[field])
    if required:
        raise ValueError(
            "--video-mode requires video_id, video, or sequence_id on every manifest record"
        )
    return str(record["id"])


def _select_records(
    records: list[dict],
    *,
    subsample: int,
    video_mode: bool,
) -> list[tuple[int, dict]]:
    if subsample < 1:
        raise ValueError("subsample must be at least 1")
    indexed = list(enumerate(records))
    if not video_mode:
        return indexed[::subsample]
    seen: dict[str, int] = defaultdict(int)
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for index, record in indexed:
        video = _video_id(record, required=True)
        if seen[video] % subsample == 0:
            grouped.setdefault(video, []).append((index, record))
        seen[video] += 1
    too_short = [video for video, group in grouped.items() if len(group) < 2]
    if too_short:
        raise ValueError(
            "video mode requires at least two selected frames per video; too short: "
            + ", ".join(too_short)
        )
    selected = []
    for group in grouped.values():
        selected.extend(
            sorted(
                group,
                key=lambda item: int(
                    item[1].get("frame_index", item[1].get("frame", item[0]))
                ),
            )
        )
    return selected


def _shard_variant(record: dict) -> str:
    tmo = str(record["tmo"])
    return tmo if record.get("crf") is None else f"{tmo}_crf{record['crf']}"


def _evaluate_frames(
    manifest: str | Path,
    predict_fn,
    *,
    variants: list[str] | None,
    max_records: int | None,
    availability: Availability,
    subsample: int,
    video_mode: bool,
    prediction_writer: PredictionWriter | None = None,
) -> tuple[list[dict], dict[tuple[str, str], dict[str, float | str | None]]]:
    root = Path(manifest).parent
    records = load_manifest(manifest)
    if max_records is not None:
        records = records[:max_records]
    records_with_indices = _select_records(records, subsample=subsample, video_mode=video_mode)
    available_optional = {
        name for name in OPTIONAL_METRICS if availability.get(name) is None
    }
    rows: list[dict] = []
    temporal_results: dict[tuple[str, str], dict[str, float | str | None]] = {}
    active_temporal: dict[tuple[str, str], TemporalAccumulator] = {}
    expert_groups: dict[
        tuple[str, str], list[tuple[dict, torch.Tensor, torch.Tensor]]
    ] = defaultdict(list)
    active_video: str | None = None
    shard_dataset = (
        SdrHdrShards(manifest, crop_size=None, random_flip=False)
        if is_shard_manifest(manifest)
        else None
    )

    def finish_video() -> None:
        for key, accumulator in active_temporal.items():
            temporal_results[key] = accumulator.compute()
        active_temporal.clear()

    for manifest_index, record in tqdm(records_with_indices, desc="eval"):
        if shard_dataset is not None:
            sample = collate_sdr_hdr([shard_dataset[manifest_index]])
            target = sample["hdr"][0]
            inputs = [(_shard_variant(record), sample["sdr"][0])]
        else:
            target = load_hdr_png16(root / record["hdr"])
            inputs = [
                (variant, load_sdr_png(root / relative_path))
                for variant, relative_path in record["sdr_variants"].items()
            ]
        video = _video_id(record, required=video_mode)
        if video_mode and active_video is not None and video != active_video:
            finish_video()
        active_video = video
        frame_index = int(record.get("frame_index", record.get("frame", manifest_index)))
        for variant, sdr in inputs:
            if variants and variant not in variants:
                continue
            pred = predict_fn(sdr)
            target_for_score = target.to(pred.device)
            if pred.shape[-2:] != target_for_score.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(0),
                    size=target_for_score.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            row = {
                "id": record.get("id", f"{video}:{frame_index}:{variant}"),
                "video": video,
                "frame_index": frame_index,
                "variant": variant,
                "category": record.get(
                    "category", record.get("source_dataset", "pgc")
                ),
                "headline_protocol": "raw",
            }
            row.update(
                score_pair(
                    pred,
                    target_for_score,
                    with_optional=True,
                    available_metrics=available_optional,
                )
            )
            if prediction_writer is not None:
                prediction_writer.write(
                    index=len(rows),
                    row=row,
                    record=record,
                    sdr=sdr,
                    prediction=pred,
                    reference=target_for_score,
                )
            rows.append(row)
            if _is_expert_track(record, variant):
                expert_groups[(_content_id(record), variant)].append(
                    (row, pred.detach().cpu(), target_for_score.detach().cpu())
                )
            if video_mode:
                active_temporal.setdefault((video, variant), TemporalAccumulator()).update(
                    pred, target_for_score
                )
    if video_mode:
        finish_video()
    _add_expert_calibrated_scores(expert_groups)
    return rows, temporal_results


def evaluate_manifest(
    manifest: str | Path,
    predict_fn,
    variants: list[str] | None = None,
    max_records: int | None = None,
    with_optional: bool = True,
    subsample: int = 1,
) -> list[dict]:
    """Evaluate a frame manifest while preserving the original list API."""
    availability = probe_metric_backends(include_optional=with_optional)
    rows, _ = _evaluate_frames(
        manifest,
        predict_fn,
        variants=variants,
        max_records=max_records,
        availability=availability,
        subsample=subsample,
        video_mode=False,
    )
    return rows


def evaluate_precomputed_manifest(
    manifest: str | Path,
    *,
    max_records: int | None = None,
    with_optional: bool = True,
) -> list[dict]:
    """Score a JSONL manifest containing ``prediction`` and ``reference`` paths."""
    return run_precomputed_benchmark(
        manifest,
        max_records=max_records,
        include_optional=with_optional,
        video_mode=False,
    ).frame_rows


def _evaluate_precomputed_frames(
    manifest: str | Path,
    *,
    max_records: int | None,
    availability: Availability,
    video_mode: bool,
) -> tuple[list[dict], dict[tuple[str, str], dict[str, float | str | None]]]:
    root = Path(manifest).parent
    records = load_manifest(manifest)
    if max_records:
        records = records[:max_records]
    available_optional = {
        name for name in OPTIONAL_METRICS if availability.get(name) is None
    }
    rows: list[dict] = []
    temporal_results: dict[tuple[str, str], dict[str, float | str | None]] = {}
    temporal: dict[tuple[str, str], TemporalAccumulator] = {}
    expert_groups: dict[
        tuple[str, str], list[tuple[dict, torch.Tensor, torch.Tensor]]
    ] = defaultdict(list)
    for index, record in enumerate(tqdm(records, desc="eval-precomputed")):
        pred = load_hdr_png16(root / record["prediction"])
        target = load_hdr_png16(root / record["reference"])
        if pred.shape[-2:] != target.shape[-2:]:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        video = _video_id(record, required=video_mode)
        variant = str(record.get("variant", "precomputed"))
        row = {
            "id": record.get("id", str(index)),
            "video": video,
            "frame_index": int(record.get("frame_index", record.get("frame", index))),
            "variant": variant,
            "category": record.get("category", "verification"),
            "headline_protocol": "raw",
        }
        row.update(
            score_pair(
                pred,
                target,
                with_optional=True,
                available_metrics=available_optional,
            )
        )
        rows.append(row)
        if _is_expert_track(record, str(row["variant"])):
            expert_groups[(_content_id(record), str(row["variant"]))].append(
                (row, pred, target)
            )
        if video_mode:
            temporal.setdefault((video, variant), TemporalAccumulator()).update(pred, target)
    _add_expert_calibrated_scores(expert_groups)
    if video_mode:
        for key, accumulator in temporal.items():
            temporal_results[key] = accumulator.compute()
    return rows, temporal_results


def run_precomputed_benchmark(
    manifest: str | Path,
    *,
    max_records: int | None = None,
    include_optional: bool = True,
    video_mode: bool = False,
) -> BenchmarkResult:
    """Score precomputed PNG16 pairs and build frame, video, and overall tables."""
    availability = probe_metric_backends(
        include_optional=include_optional,
        video_mode=video_mode,
    )
    frames, temporal = _evaluate_precomputed_frames(
        manifest,
        max_records=max_records,
        availability=availability,
        video_mode=video_mode,
    )
    videos = aggregate_per_video(frames, temporal if video_mode else None)
    return BenchmarkResult(frames, videos, _overall_rows(videos), availability)


def aggregate(rows: list[dict], key: str = "variant") -> dict[str, dict[str, float]]:
    """Average all populated metric columns by a row key and overall."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    groups["ALL"] = rows
    aggregated: dict[str, dict[str, float]] = {}
    for name, group_rows in groups.items():
        aggregated[name] = {}
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in group_rows if row.get(metric) is not None]
            if values:
                aggregated[name][metric] = sum(values) / len(values)
    return aggregated


def aggregate_per_video(
    frame_rows: list[dict],
    temporal: dict[tuple[str, str], dict[str, float | str | None]] | None = None,
) -> list[dict]:
    """Average frame metrics for each video and variant."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in frame_rows:
        groups[(str(row["video"]), str(row["variant"]))].append(row)
    video_rows = []
    for (video, variant), rows in sorted(groups.items()):
        result = {
            "video": video,
            "variant": variant,
            "category": rows[0]["category"],
            "n_frames": len(rows),
            "headline_protocol": rows[0].get("headline_protocol", "raw"),
        }
        gains = [float(row["exposure_gain"]) for row in rows if "exposure_gain" in row]
        if gains:
            result["exposure_gain"] = gains[0]
        for metric in FRAME_METRIC_COLUMNS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            result[metric] = sum(values) / len(values) if values else None
        if temporal is not None:
            result.update(temporal[(video, variant)])
        video_rows.append(result)
    return video_rows


def _overall_rows(video_rows: list[dict]) -> list[dict]:
    summary = aggregate(video_rows)
    rows = []
    for variant, metrics in sorted(summary.items()):
        count = len(video_rows) if variant == "ALL" else sum(
            row["variant"] == variant for row in video_rows
        )
        row = {"variant": variant, "n_videos": count}
        protocols = {
            str(video_row.get("headline_protocol", "raw"))
            for video_row in video_rows
            if variant == "ALL" or video_row["variant"] == variant
        }
        row["headline_protocol"] = protocols.pop() if len(protocols) == 1 else "mixed"
        row.update(metrics)
        rows.append(row)
    return rows


def run_benchmark(
    manifest: str | Path,
    predict_fn,
    *,
    variants: list[str] | None = None,
    max_records: int | None = None,
    include_optional: bool = True,
    subsample: int = 1,
    video_mode: bool = False,
    availability: Availability | None = None,
    prediction_writer: PredictionWriter | None = None,
) -> BenchmarkResult:
    """Run probing, frame scoring, per-video reduction, and overall reduction."""
    if availability is None:
        availability = probe_metric_backends(
            include_optional=include_optional,
            video_mode=video_mode,
        )
    frames, temporal = _evaluate_frames(
        manifest,
        predict_fn,
        variants=variants,
        max_records=max_records,
        availability=availability,
        subsample=subsample,
        video_mode=video_mode,
        prediction_writer=prediction_writer,
    )
    videos = aggregate_per_video(frames, temporal if video_mode else None)
    return BenchmarkResult(frames, videos, _overall_rows(videos), availability)


def to_markdown(
    aggregated: dict[str, dict[str, float]],
    availability: Availability | None = None,
    columns: list[str] | None = None,
    first_column: str = "Variant",
) -> str:
    """Render an aggregate table with explicit backend availability footers."""
    requested = columns or METRIC_COLUMNS
    shown = [
        name
        for name in requested
        if any(name in values for values in aggregated.values())
        or (availability is not None and availability.get(name) is not None)
    ]
    header = f"| {first_column} | " + " | ".join(shown) + " |"
    separator = "|" + "---|" * (len(shown) + 1)
    lines = [header, separator]
    for name in sorted(aggregated):
        values = [
            f"{aggregated[name][metric]:.3f}" if metric in aggregated[name] else "-"
            for metric in shown
        ]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    if availability is not None:
        unavailable = [
            (name, availability[name])
            for name in requested
            if availability.get(name) is not None
        ]
        if unavailable:
            lines.extend(["", "Metric availability:", ""])
            lines.extend(f"- {name}: unavailable: {reason}" for name, reason in unavailable)
    return "\n".join(lines)


def _rows_markdown(
    rows: list[dict],
    *,
    availability: Availability,
) -> str:
    keyed = {
        f"{row['video']} / {row['variant']}": {
            metric: float(row[metric])
            for metric in METRIC_COLUMNS
            if row.get(metric) is not None
        }
        for row in rows
    }
    return to_markdown(keyed, availability, first_column="Video / variant")


def save_csv(
    rows: list[dict],
    path: str | Path,
    *,
    metric_columns: list[str] | None = None,
    availability: Availability | None = None,
) -> None:
    """Write stable CSV columns and materialize unavailable reasons."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = metric_columns or []
    metadata = []
    for row in rows:
        for field in row:
            if field not in metrics and field not in metadata:
                metadata.append(field)
    fieldnames = metadata + [name for name in metrics if name not in metadata]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for source in rows:
            row = {field: source.get(field) for field in fieldnames}
            if availability is not None:
                for metric in metrics:
                    if row.get(metric) is None and availability.get(metric) is not None:
                        row[metric] = f"unavailable: {availability[metric]}"
            writer.writerow(row)


def save_benchmark_outputs(result: BenchmarkResult, output_dir: Path) -> tuple[str, str]:
    save_csv(
        result.frame_rows,
        output_dir / "per_frame.csv",
        metric_columns=FRAME_METRIC_COLUMNS,
        availability=result.availability,
    )
    save_csv(
        result.video_rows,
        output_dir / "per_video.csv",
        metric_columns=METRIC_COLUMNS,
        availability=result.availability,
    )
    save_csv(
        result.overall_rows,
        output_dir / "overall.csv",
        metric_columns=METRIC_COLUMNS,
        availability=result.availability,
    )
    availability_rows = [
        {
            "metric": metric,
            "status": "available" if reason is None else f"unavailable: {reason}",
        }
        for metric, reason in result.availability.items()
    ]
    save_csv(availability_rows, output_dir / "availability.csv")

    overall = aggregate(result.video_rows)
    summary_markdown = to_markdown(overall, result.availability)
    video_markdown = _rows_markdown(result.video_rows, availability=result.availability)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.md").write_text(summary_markdown + "\n")
    (output_dir / "per_video.md").write_text(video_markdown + "\n")
    return summary_markdown, video_markdown


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LumaFlux benchmark evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapters", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--predictions-manifest",
        default=None,
        help="JSONL pairs with prediction and reference paths",
    )
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="save model inputs, references, PNG16 predictions, and a rescore manifest",
    )
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=1, help="evaluate every Nth frame")
    parser.add_argument("--video-mode", action="store_true")
    parser.add_argument("--no-optional-metrics", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.subsample < 1:
        parser.error("--subsample must be at least 1")

    if args.predictions_manifest:
        if args.manifest or args.config or args.adapters or args.predictions_dir:
            parser.error("--predictions-manifest cannot be combined with model inference arguments")
        result = run_precomputed_benchmark(
            args.predictions_manifest,
            max_records=args.max_records,
            include_optional=not args.no_optional_metrics,
            video_mode=args.video_mode,
        )
        out = Path(args.output_dir)
        summary, _ = save_benchmark_outputs(result, out)
        print(summary)
        return
    if not args.config or not args.manifest:
        parser.error("--config and --manifest are required for model inference")

    availability = probe_metric_backends(
        include_optional=not args.no_optional_metrics,
        video_mode=args.video_mode,
    )

    from ..inference.pipeline import LumaFluxPipeline
    from ..models.factory import build_model, load_config

    config = load_config(args.config)
    model = build_model(config, device=args.device)
    if args.adapters:
        model.load_adapters(args.adapters)
    pipeline = LumaFluxPipeline(model, device=args.device, cfg=config)
    multiple = 2 * model.vae_scale

    def predict(sdr: torch.Tensor) -> torch.Tensor:
        height, width = sdr.shape[-2:]
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h or pad_w:
            sdr = torch.nn.functional.pad(
                sdr.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate"
            ).squeeze(0)
        output = pipeline(sdr.unsqueeze(0), num_steps=args.num_steps)
        return output["hdr"][0][:, :height, :width]

    prediction_writer = PredictionWriter(args.predictions_dir) if args.predictions_dir else None
    result = run_benchmark(
        args.manifest,
        predict,
        variants=args.variants,
        max_records=args.max_records,
        include_optional=not args.no_optional_metrics,
        subsample=args.subsample,
        video_mode=args.video_mode,
        availability=availability,
        prediction_writer=prediction_writer,
    )
    if prediction_writer is not None:
        prediction_writer.finish()
    summary, _ = save_benchmark_outputs(result, Path(args.output_dir))
    print(summary)


if __name__ == "__main__":
    main()
