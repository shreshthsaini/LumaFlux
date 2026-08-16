"""Parallel SDR/HDR curation into safetensors shards.

The primary pipeline decodes each source video once, normalizes its sampled
frames to PQ/BT.2020, and stores native-resolution integer tensors. The older
PNG curator remains available as :func:`curate_png` for compatibility only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import random
import uuid
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import torch
from tqdm import tqdm

from ..utils.io import (
    load_hdr_png16,
    load_sdr_png,
    save_hdr_png16,
    save_sdr_png,
)
from .degrade import CRF_LEVELS, degrade_chain
from .normalize import normalize_to_pq2020
from .shards import DEFAULT_SHARD_BYTES, SHARD_FORMAT, ShardWriter, write_jsonl_atomic
from .tmo import TMO_REGISTRY, TONE_CHAIN_VERSION
from .video_decode import decode_hdr_video, decode_sdr_video, ffmpeg_available

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".y4m"}
IMAGE_EXTS = {".png", ".tif", ".tiff", ".exr"}


@dataclass
class SourceSpec:
    name: str
    path: str
    mode: str = "derived"
    sdr_path: str | None = None
    category: str = "pgc"
    format: str = "pq2020"
    fps: float = 1.0
    expert_sdr_path: str | None = None
    expert_fps: float = 60.0
    frame_offset: int = 0
    max_frames_per_video: int | None = None
    max_pairs: int | None = None
    pair_seed: int = 260402787
    split: str = "train"
    source_dataset: str | None = None
    reference_csv: str | None = None
    reference_hash_column: str = "Video"
    reference_filter_column: str | None = "ref"
    reference_filter_value: str = "1"
    include_video_hashes: list[str] | None = None
    exclude_video_hashes: list[str] = field(default_factory=list)
    holdout_count: int = 0
    holdout_seed: int = 260402787
    enabled: bool = True


@dataclass
class CurationConfig:
    sources: list[SourceSpec]
    out_dir: str = "data/curated"
    peak_nits: float = 1000.0
    tmos: list[str] = field(default_factory=lambda: sorted(TMO_REGISTRY))
    crf_levels: list[int] = field(default_factory=lambda: list(CRF_LEVELS))
    seed: int = 0
    workers: int | None = None
    shard_bytes: int = DEFAULT_SHARD_BYTES


@dataclass(frozen=True)
class VideoTask:
    source: SourceSpec
    video_hash: str
    files: tuple[str, ...]
    expert_files: tuple[str | None, ...]
    is_video: bool
    fps: float
    native_sdr_files: tuple[str, ...] = ()


def _stable_hash(source: str, relative: str) -> str:
    stem = Path(relative).stem.lower()
    if len(stem) >= 16 and all(char in "0123456789abcdef" for char in stem):
        return stem
    return hashlib.sha256(f"{source}:{relative}".encode()).hexdigest()[:32]


def _reference_hashes(spec: SourceSpec) -> set[str] | None:
    if spec.reference_csv is None:
        return None
    with open(spec.reference_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if spec.reference_filter_column:
        matching = [
            row
            for row in rows
            if str(row.get(spec.reference_filter_column, "")).strip()
            == str(spec.reference_filter_value)
        ]
        if matching:
            rows = matching
    return {str(row[spec.reference_hash_column]).strip() for row in rows}


def _expert_match(spec: SourceSpec, source_root: Path, source_file: Path) -> Path | None:
    if spec.expert_sdr_path is None:
        return None
    expert_root = Path(spec.expert_sdr_path)
    candidates = [expert_root / source_file.relative_to(source_root), expert_root / source_file.name]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def deterministic_video_selection(
    candidates: Iterable[str], count: int, seed: int, source_name: str
) -> list[str]:
    """Sample deterministically from sorted hashes for a named source."""
    ordered = sorted(candidates)
    if len(ordered) < count:
        raise ValueError(f"{source_name} has {len(ordered)} videos, but {count} are required")
    rng = random.Random(f"{seed}:{source_name}")
    return sorted(rng.sample(ordered, count))


def _apply_holdout(spec: SourceSpec, tasks: list[VideoTask]) -> list[VideoTask]:
    if not spec.holdout_count:
        return tasks
    held_out = set(
        deterministic_video_selection(
            (task.video_hash for task in tasks),
            spec.holdout_count,
            spec.holdout_seed,
            spec.name,
        )
    )
    return [task for task in tasks if task.video_hash not in held_out]


def _discover_source_tasks(spec: SourceSpec, *, apply_holdout: bool = True) -> list[VideoTask]:
    if not spec.enabled:
        return []
    root = Path(spec.path)
    if not root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    if spec.mode == "native_pairs":
        return _discover_native_pair_tasks(spec, root)
    allowed = _reference_hashes(spec)
    if spec.include_video_hashes is not None:
        included = set(spec.include_video_hashes)
        allowed = included if allowed is None else allowed & included
    excluded = set(spec.exclude_video_hashes)

    videos = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTS)
    tasks: list[VideoTask] = []
    for video in videos:
        video_hash = _stable_hash(spec.name, video.relative_to(root).as_posix())
        if allowed is not None and video_hash not in allowed:
            continue
        if video_hash in excluded:
            continue
        expert = _expert_match(spec, root, video)
        tasks.append(
            VideoTask(
                source=spec,
                video_hash=video_hash,
                files=(str(video),),
                expert_files=((None,) if expert is None else (str(expert),)),
                is_video=True,
                fps=spec.expert_fps if expert is not None else spec.fps,
            )
        )
    if videos:
        return _apply_holdout(spec, tasks) if apply_holdout else tasks

    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    groups: dict[Path, list[Path]] = {}
    for image in images:
        groups.setdefault(image.parent, []).append(image)
    for parent, frames in sorted(groups.items(), key=lambda item: str(item[0])):
        relative = parent.relative_to(root).as_posix()
        video_hash = _stable_hash(spec.name, relative)
        if allowed is not None and video_hash not in allowed:
            continue
        if video_hash in excluded:
            continue
        expert_files = tuple(
            None if (match := _expert_match(spec, root, frame)) is None else str(match)
            for frame in frames
        )
        tasks.append(
            VideoTask(
                source=spec,
                video_hash=video_hash,
                files=tuple(str(frame) for frame in frames),
                expert_files=expert_files,
                is_video=False,
                fps=spec.expert_fps if any(expert_files) else spec.fps,
            )
        )
    return _apply_holdout(spec, tasks) if apply_holdout else tasks


def _discover_native_pair_tasks(spec: SourceSpec, hdr_root: Path) -> list[VideoTask]:
    if spec.sdr_path is None:
        raise ValueError(f"Native-pair source {spec.name} requires sdr_path")
    sdr_root = Path(spec.sdr_path)
    if not sdr_root.is_dir():
        raise FileNotFoundError(f"Native SDR directory does not exist: {sdr_root}")
    hdr_by_relative = {
        path.relative_to(hdr_root): path
        for path in hdr_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    sdr_by_relative = {
        path.relative_to(sdr_root): path
        for path in sdr_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }
    if not hdr_by_relative:
        raise ValueError(f"Native HDR directory contains no supported images: {hdr_root}")
    missing_sdr = sorted(str(path) for path in hdr_by_relative.keys() - sdr_by_relative.keys())
    missing_hdr = sorted(str(path) for path in sdr_by_relative.keys() - hdr_by_relative.keys())
    if missing_sdr or missing_hdr:
        raise ValueError(
            f"Native-pair filenames do not match for {spec.name}: "
            f"missing SDR={missing_sdr[:5]}, missing HDR={missing_hdr[:5]}"
        )
    relatives = sorted(hdr_by_relative)
    if spec.max_pairs is not None:
        if spec.max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        if len(relatives) < spec.max_pairs:
            raise ValueError(
                f"Native-pair source {spec.name} has {len(relatives)} pairs, "
                f"but {spec.max_pairs} are required"
            )
        rng = random.Random(f"{spec.pair_seed}:{spec.name}:native_pairs")
        relatives = sorted(rng.sample(relatives, spec.max_pairs))
    return [
        VideoTask(
            source=spec,
            video_hash=_stable_hash(spec.name, relative.as_posix()),
            files=(str(hdr_by_relative[relative]),),
            expert_files=(None,),
            is_video=False,
            fps=spec.fps,
            native_sdr_files=(str(sdr_by_relative[relative]),),
        )
        for relative in relatives
    ]


def discover_video_hashes(spec: SourceSpec) -> list[str]:
    """Return all sorted hashes before the configured training holdout is removed."""
    enabled_spec = replace(spec, enabled=True)
    return sorted(
        task.video_hash
        for task in _discover_source_tasks(enabled_spec, apply_holdout=False)
    )


def _iter_hdr_task(task: VideoTask) -> Iterable[tuple[int, torch.Tensor]]:
    if task.is_video:
        yield from decode_hdr_video(
            task.files[0],
            fps=task.fps,
            max_frames=task.source.max_frames_per_video,
        )
        return
    limit = task.source.max_frames_per_video
    for frame_idx, path in enumerate(task.files[:limit]):
        frame = load_hdr_png16(path)
        yield frame_idx, (frame.clamp(0, 1) * 65535.0).round().to(torch.uint16)


def _iter_expert_task(task: VideoTask) -> Iterable[torch.Tensor | None]:
    if not any(task.expert_files):
        return
    if task.is_video:
        assert task.expert_files[0] is not None
        for _, frame in decode_sdr_video(
            task.expert_files[0],
            fps=task.fps,
            max_frames=task.source.max_frames_per_video,
        ):
            yield frame
        return
    limit = task.source.max_frames_per_video
    for path in task.expert_files[:limit]:
        if path is None:
            yield None
            continue
        frame = load_sdr_png(path)
        yield (frame.clamp(0, 1) * 255.0).round().to(torch.uint8)


def _load_native_png(path: str, expected_dtype: torch.dtype) -> torch.Tensor:
    """Decode a native PNG to CHW without changing its integer code values."""
    import cv2
    import numpy as np

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"failed to read {path}")
    numpy_dtype = np.uint8 if expected_dtype == torch.uint8 else np.uint16
    if image.dtype != numpy_dtype:
        raise ValueError(
            f"Native PNG {path} has dtype {image.dtype}, expected {numpy_dtype.__name__}"
        )
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Native PNG {path} must have at least three color channels")
    rgb = np.ascontiguousarray(image[..., :3][..., ::-1])
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy())


def _curate_worker(
    worker_index: int,
    tasks: list[VideoTask],
    staging_root: str,
    peak_nits: float,
    tmos: list[str],
    crf_levels: list[int],
    shard_bytes: int,
) -> dict:
    worker_root = Path(staging_root) / f"worker-{worker_index:03d}"
    hdr_writer = ShardWriter(
        worker_root,
        f"hdr-w{worker_index:03d}",
        target_bytes=shard_bytes,
        metadata={"kind": "hdr", "dtype": "uint16", "color": "PQ BT.2020 RGB"},
    )
    sdr_writer = ShardWriter(
        worker_root,
        f"sdr-w{worker_index:03d}",
        target_bytes=shard_bytes,
        metadata={"kind": "sdr", "dtype": "uint8", "color": "BT.709 RGB"},
    )
    records: list[dict] = []
    for task in tasks:
        if task.source.mode == "native_pairs":
            if len(task.files) != 1 or len(task.native_sdr_files) != 1:
                raise RuntimeError("Native-pair tasks must contain exactly one SDR/HDR pair")
            hdr_uint16 = _load_native_png(task.files[0], torch.uint16)
            sdr_uint8 = _load_native_png(task.native_sdr_files[0], torch.uint8)
            if hdr_uint16.shape != sdr_uint8.shape:
                raise ValueError(
                    f"Native pair shape mismatch for {task.files[0]}: "
                    f"SDR {tuple(sdr_uint8.shape)} vs HDR {tuple(hdr_uint16.shape)}"
                )
            hdr_key = f"hdr/{task.video_hash}/0"
            sdr_key = f"sdr/{task.video_hash}/0/native/none"
            records.append(
                {
                    "hdr_shard": hdr_writer.add(hdr_key, hdr_uint16),
                    "hdr_key": hdr_key,
                    "sdr_shard": sdr_writer.add(sdr_key, sdr_uint8),
                    "sdr_key": sdr_key,
                    "video": task.video_hash,
                    "frame": 0,
                    "tmo": "native",
                    "crf": None,
                    "split": task.source.split,
                    "source_dataset": task.source.source_dataset or task.source.name,
                }
            )
            continue
        expert_frames = iter(_iter_expert_task(task))
        for frame_idx, raw_hdr in _iter_hdr_task(task):
            hdr_float = raw_hdr.to(torch.float32).div_(65535.0)
            normalized = normalize_to_pq2020(
                hdr_float,
                source_format=task.source.format,
                peak_nits=peak_nits,
            )
            hdr_uint16 = (normalized.clamp(0, 1) * 65535.0).round().to(torch.uint16)
            hdr_key = f"hdr/{task.video_hash}/{frame_idx}"
            hdr_shard = hdr_writer.add(hdr_key, hdr_uint16)
            for tmo in tmos:
                for crf in crf_levels:
                    sdr = degrade_chain(normalized, tmo, crf=crf, peak_nits=peak_nits)
                    sdr_uint8 = (sdr.clamp(0, 1) * 255.0).round().to(torch.uint8)
                    sdr_key = f"sdr/{task.video_hash}/{frame_idx}/{tmo}/{crf}"
                    sdr_shard = sdr_writer.add(sdr_key, sdr_uint8)
                    records.append(
                        {
                            "hdr_shard": hdr_shard,
                            "hdr_key": hdr_key,
                            "sdr_shard": sdr_shard,
                            "sdr_key": sdr_key,
                            "video": task.video_hash,
                            "frame": frame_idx,
                            "tmo": tmo,
                            "crf": crf,
                            "split": task.source.split,
                            "source_dataset": task.source.source_dataset or task.source.name,
                        }
                    )
            expert = (
                next(expert_frames, None)
                if frame_idx >= task.source.frame_offset
                else None
            )
            if expert is not None:
                sdr_key = f"sdr/{task.video_hash}/{frame_idx}/expert/none"
                sdr_shard = sdr_writer.add(sdr_key, expert.to(torch.uint8))
                records.append(
                    {
                        "hdr_shard": hdr_shard,
                        "hdr_key": hdr_key,
                        "sdr_shard": sdr_shard,
                        "sdr_key": sdr_key,
                        "video": task.video_hash,
                        "frame": frame_idx,
                        "tmo": "expert",
                        "crf": None,
                        "split": task.source.split,
                        "source_dataset": task.source.source_dataset or task.source.name,
                    }
                )
    hdr_paths = hdr_writer.close()
    sdr_paths = sdr_writer.close()
    part_manifest = write_jsonl_atomic(worker_root / "manifest.jsonl", records)
    return {
        "manifest": str(part_manifest),
        "hdr_paths": [str(path) for path in hdr_paths],
        "sdr_paths": [str(path) for path in sdr_paths],
    }


def _load_pair_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                continue
            if "hdr_shard" not in record:
                raise ValueError(f"Cannot mix legacy PNG records with shard records in {path}")
            records.append(record)
    return records


def _load_header(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            return record if record.get("type") == "header" else None
    return None


def _curation_header(cfg: CurationConfig) -> dict:
    return {
        "format": SHARD_FORMAT,
        "tone_chain_version": TONE_CHAIN_VERSION,
        "hdr_encoding": "PQ-coded BT.2020 RGB, absolute ST 2084",
        "hdr_mastering_peak_nits": float(cfg.peak_nits),
        "sdr_encoding": "BT.709 RGB, display light encoded by inverse BT.1886",
        "gamut_mapping": "linear BT.2020 to BT.709 matrix plus hard clip",
    }


def _validate_resume_header(path: Path, cfg: CurationConfig, records: list[dict]) -> None:
    if not records:
        return
    actual = _load_header(path)
    expected = _curation_header(cfg)
    if actual is None:
        raise RuntimeError(
            f"Cannot resume legacy curation manifest without a convention header: {path}. "
            "Use a new output directory so old and corrected tone chains are not mixed."
        )
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Cannot resume curation with a different luminance convention: {mismatches}"
        )


def _next_shard_index(root: Path, kind: str) -> int:
    indices = []
    for path in root.glob(f"{kind}-*.safetensors"):
        suffix = path.stem.removeprefix(f"{kind}-")
        if suffix.isdigit():
            indices.append(int(suffix))
    return max(indices, default=-1) + 1


def curate(cfg: CurationConfig) -> Path:
    """Curate source videos into resumable safetensors shards and a pair manifest."""
    invalid_offsets = {
        source.name: source.frame_offset
        for source in cfg.sources
        if source.frame_offset < 0
    }
    if invalid_offsets:
        raise ValueError(f"frame_offset must be non-negative: {invalid_offsets}")
    unknown_modes = sorted({source.mode for source in cfg.sources} - {"derived", "native_pairs"})
    if unknown_modes:
        raise ValueError(f"Unknown curation source modes: {unknown_modes}")
    if sorted(cfg.tmos) != sorted(set(cfg.tmos)):
        raise ValueError("TMO names must be unique")
    unknown = sorted(set(cfg.tmos) - set(TMO_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown TMOs: {unknown}")
    if not ffmpeg_available() and any(
        source.enabled and any(
            path.suffix.lower() in VIDEO_EXTS for path in Path(source.path).rglob("*")
        )
        for source in cfg.sources
        if Path(source.path).is_dir()
    ):
        raise RuntimeError("The vendored high-bit-depth ffmpeg stack is required for video sources")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    existing_records = _load_pair_records(manifest_path)
    _validate_resume_header(manifest_path, cfg, existing_records)
    header = _curation_header(cfg)
    completed_videos = {record["video"] for record in existing_records}
    tasks = [
        task
        for source in cfg.sources
        for task in _discover_source_tasks(source)
        if task.video_hash not in completed_videos
    ]
    if not tasks:
        if not manifest_path.exists():
            write_jsonl_atomic(manifest_path, [], header=header)
        return manifest_path

    default_workers = min(16, os.cpu_count() or 1)
    worker_limit = default_workers if cfg.workers is None else cfg.workers
    worker_count = min(len(tasks), worker_limit)
    if worker_count <= 0:
        raise ValueError("workers must be positive")
    task_batches = [tasks[index::worker_count] for index in range(worker_count)]
    staging = out / "_staging" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=False)
    arguments = [
        (
            index,
            batch,
            str(staging),
            cfg.peak_nits,
            cfg.tmos,
            cfg.crf_levels,
            cfg.shard_bytes,
        )
        for index, batch in enumerate(task_batches)
    ]
    if worker_count == 1:
        results = [_curate_worker(*arguments[0])]
    else:
        with mp.get_context("spawn").Pool(worker_count) as pool:
            results = pool.starmap(_curate_worker, arguments)

    shard_mapping: dict[str, str] = {}
    for kind in ("hdr", "sdr"):
        source_paths = sorted(
            Path(path)
            for result in results
            for path in result[f"{kind}_paths"]
        )
        next_index = _next_shard_index(out, kind)
        for offset, source_path in enumerate(source_paths):
            destination = out / f"{kind}-{next_index + offset:05d}.safetensors"
            source_path.replace(destination)
            shard_mapping[source_path.name] = destination.name

    new_records = []
    for result in results:
        with open(result["manifest"]) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["hdr_shard"] = shard_mapping[record["hdr_shard"]]
                record["sdr_shard"] = shard_mapping[record["sdr_shard"]]
                new_records.append(record)
    new_records.sort(
        key=lambda record: (
            record["source_dataset"],
            record["video"],
            record["frame"],
            record["tmo"],
            -1 if record["crf"] is None else record["crf"],
        )
    )
    write_jsonl_atomic(manifest_path, existing_records + new_records, header=header)
    return manifest_path


def _iter_source_frames(spec: SourceSpec, work_dir: Path) -> Iterable[tuple[str, Path]]:
    """Yield legacy PNG inputs. This exists only for old synthetic tests and users."""
    root = Path(spec.path)
    videos = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTS)
    if videos:
        if not ffmpeg_available():
            raise RuntimeError(f"Source {spec.name} contains videos but ffmpeg is missing")
        for video in videos:
            frame_dir = work_dir / spec.name / video.stem
            frame_dir.mkdir(parents=True, exist_ok=True)
            frames = decode_hdr_video(
                video,
                fps=spec.fps,
                max_frames=spec.max_frames_per_video,
            )
            for frame_idx, frame in frames:
                frame_path = frame_dir / f"frame_{frame_idx:06d}.png"
                save_hdr_png16(frame.to(torch.float32).div_(65535.0), frame_path)
                yield f"{video.stem}_{frame_path.stem}", frame_path
    else:
        for frame in sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTS):
            yield f"{frame.parent.name}_{frame.stem}", frame


def curate_png(cfg: CurationConfig) -> Path:
    """Deprecated PNG curator retained for backward compatibility."""
    warnings.warn(
        "curate_png is deprecated; use curate for safetensors shards",
        DeprecationWarning,
        stacklevel=2,
    )
    out = Path(cfg.out_dir)
    (out / "hdr").mkdir(parents=True, exist_ok=True)
    (out / "sdr").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    records = []
    for spec in cfg.sources:
        if not spec.enabled:
            continue
        for frame_id, frame_path in tqdm(
            _iter_source_frames(spec, out / "_work"), desc=f"curate-png:{spec.name}"
        ):
            frame = load_hdr_png16(frame_path)
            normalized = normalize_to_pq2020(
                frame, source_format=spec.format, peak_nits=cfg.peak_nits
            )
            hdr_relative = f"hdr/{spec.name}_{frame_id}.png"
            save_hdr_png16(normalized, out / hdr_relative)
            variants = {}
            for tmo in cfg.tmos:
                for crf in cfg.crf_levels:
                    sdr = degrade_chain(normalized, tmo, crf=crf, peak_nits=cfg.peak_nits)
                    relative = f"sdr/{spec.name}_{frame_id}__{tmo}_crf{crf}.png"
                    save_sdr_png(sdr, out / relative)
                    variants[f"{tmo}_crf{crf}"] = relative
            if spec.expert_sdr_path is not None:
                expert = Path(spec.expert_sdr_path) / frame_path.name
                if expert.exists():
                    sdr = load_sdr_png(expert)
                    relative = f"sdr/{spec.name}_{frame_id}__expert.png"
                    save_sdr_png(sdr, out / relative)
                    variants["expert"] = relative
            records.append(
                {
                    "id": f"{spec.name}_{frame_id}",
                    "source": spec.name,
                    "category": spec.category,
                    "hdr": hdr_relative,
                    "sdr_variants": variants,
                }
            )
    write_jsonl_atomic(manifest_path, records)
    return manifest_path


__all__ = [
    "CurationConfig",
    "SHARD_FORMAT",
    "SourceSpec",
    "curate",
    "curate_png",
    "deterministic_video_selection",
    "discover_video_hashes",
]
