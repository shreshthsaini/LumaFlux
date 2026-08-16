import json
import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import cv2
import numpy as np
from safetensors import safe_open

from lumaflux.data.curate import (
    CurationConfig,
    SourceSpec,
    VideoTask,
    _curate_worker,
    curate,
    deterministic_video_selection,
)
from lumaflux.data.dataset import (
    SdrHdrShards,
    collate_sdr_hdr,
    load_manifest,
    load_manifest_header,
)
from lumaflux.data.shards import SHARD_FORMAT

@pytest.fixture(scope="module")
def shard_manifest(synthetic_sources, tmp_path_factory):
    output = tmp_path_factory.mktemp("shards")
    config = CurationConfig(
        sources=[
            SourceSpec(name="pgc_demo", path=str(synthetic_sources / "pgc"), format="pq2020"),
            SourceSpec(name="ugc_demo", path=str(synthetic_sources / "ugc"), format="pq2020"),
        ],
        out_dir=str(output),
        tmos=["reinhard", "bt2446c_gm"],
        crf_levels=[31],
        workers=2,
        shard_bytes=20_000,
    )
    return curate(config), config


def test_shard_manifest_and_tensor_format(shard_manifest):
    manifest, _ = shard_manifest
    records = load_manifest(manifest)
    assert len(records) == 8
    required = {
        "hdr_shard",
        "hdr_key",
        "sdr_shard",
        "sdr_key",
        "video",
        "frame",
        "tmo",
        "crf",
        "split",
        "source_dataset",
    }
    assert all(set(record) == required for record in records)
    assert {record["tmo"] for record in records} == {"reinhard", "bt2446c_gm"}
    assert {record["crf"] for record in records} == {31}

    record = records[0]
    with safe_open(manifest.parent / record["hdr_shard"], framework="pt") as handle:
        hdr = handle.get_tensor(record["hdr_key"])
        assert handle.metadata()["format"] == SHARD_FORMAT
    with safe_open(manifest.parent / record["sdr_shard"], framework="pt") as handle:
        sdr = handle.get_tensor(record["sdr_key"])
    assert hdr.dtype == torch.uint16
    assert sdr.dtype == torch.uint8
    assert hdr.shape == sdr.shape == (3, 64, 64)


def test_shard_dataset_crops_integer_then_collate_scales(shard_manifest):
    manifest, _ = shard_manifest
    dataset = SdrHdrShards(manifest, crop_size=32, random_flip=False)
    first, second = dataset[0], dataset[1]
    assert first["sdr"].dtype == torch.uint8
    assert first["hdr"].dtype == torch.uint16
    assert first["sdr"].shape == first["hdr"].shape == (3, 32, 32)

    batch = collate_sdr_hdr([first, second])
    assert batch["sdr"].dtype == batch["hdr"].dtype == torch.float32
    assert batch["sdr"].shape == batch["hdr"].shape == (2, 3, 32, 32)
    assert 0 <= batch["sdr"].min() <= batch["sdr"].max() <= 1
    assert 0 <= batch["hdr"].min() <= batch["hdr"].max() <= 1


def test_curation_resume_skips_committed_videos(shard_manifest):
    manifest, config = shard_manifest
    before_manifest = manifest.read_bytes()
    before_shards = sorted(path.name for path in manifest.parent.glob("*.safetensors"))
    assert curate(config) == manifest
    assert manifest.read_bytes() == before_manifest
    assert sorted(path.name for path in manifest.parent.glob("*.safetensors")) == before_shards


def test_default_grid_is_eight_tmos_by_three_crfs(tmp_path, monkeypatch):
    root = tmp_path / "source" / "scene"
    root.mkdir(parents=True)
    tensor = torch.linspace(0, 1, 3 * 8 * 8).reshape(3, 8, 8)
    from lumaflux.utils.io import save_hdr_png16

    save_hdr_png16(tensor, root / "frame.png")

    def no_codec_degrade(x_pq, tmo, crf, peak_nits):
        del tmo, crf, peak_nits
        return x_pq.clamp(0, 1)

    curate_module = importlib.import_module("lumaflux.data.curate")
    monkeypatch.setattr(curate_module, "degrade_chain", no_codec_degrade)
    config = CurationConfig(
        sources=[SourceSpec(name="grid", path=str(root.parent))],
        out_dir=str(tmp_path / "output"),
        workers=1,
    )
    manifest = curate(config)
    records = load_manifest(manifest)
    assert len(config.tmos) == 8
    assert config.crf_levels == [23, 31, 39]
    assert len(records) == 24
    assert {(record["tmo"], record["crf"]) for record in records} == {
        (tmo, crf) for tmo in config.tmos for crf in config.crf_levels
    }


def test_manifest_is_plain_jsonl_pairs(shard_manifest):
    manifest, _ = shard_manifest
    with open(manifest) as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    assert lines[0] == load_manifest_header(manifest)
    assert (
        lines[0]["tone_chain_version"]
        == "pq1000-bt1886-bt2407-matrix-clip-decode-matrix"
    )
    assert lines[1:] == load_manifest(manifest)


def test_legacy_manifest_cannot_be_resumed(shard_manifest, tmp_path):
    manifest, config = shard_manifest
    records = load_manifest(manifest)
    legacy_manifest = tmp_path / "manifest.jsonl"
    legacy_manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    legacy_config = replace(config, out_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="legacy curation manifest"):
        curate(legacy_config)


def test_holdout_selection_is_seeded_and_order_independent():
    hashes = [f"{index:032x}" for index in range(30)]
    selected = deterministic_video_selection(hashes, 10, 260402787, "chug")
    assert selected == deterministic_video_selection(
        list(reversed(hashes)), 10, 260402787, "chug"
    )
    assert len(selected) == len(set(selected)) == 10


def test_expert_reference_frame_offset_pairs_reference_n_plus_one(tmp_path, monkeypatch):
    source = SourceSpec(
        name="live_tmhdr",
        path="unused",
        expert_sdr_path="unused",
        frame_offset=1,
    )
    task = VideoTask(
        source=source,
        video_hash="clip",
        files=("reference.mp4",),
        expert_files=("expert.mp4",),
        is_video=True,
        fps=60.0,
    )
    hdr_frames = [
        (index, torch.full((3, 2, 2), value, dtype=torch.uint16))
        for index, value in enumerate((1000, 2000, 3000))
    ]
    expert_frames = [
        torch.full((3, 2, 2), value, dtype=torch.uint8) for value in (11, 22)
    ]
    curate_module = importlib.import_module("lumaflux.data.curate")
    monkeypatch.setattr(curate_module, "_iter_hdr_task", lambda _: iter(hdr_frames))
    monkeypatch.setattr(curate_module, "_iter_expert_task", lambda _: iter(expert_frames))
    monkeypatch.setattr(curate_module, "normalize_to_pq2020", lambda frame, **_: frame)

    result = _curate_worker(
        0,
        [task],
        str(tmp_path),
        1000.0,
        [],
        [],
        1_000_000,
    )
    records = load_manifest(result["manifest"])
    assert [record["frame"] for record in records] == [1, 2]

    worker_root = Path(result["manifest"]).parent
    paired_values = []
    for record in records:
        with safe_open(worker_root / record["hdr_shard"], framework="pt") as handle:
            hdr = handle.get_tensor(record["hdr_key"])
        with safe_open(worker_root / record["sdr_shard"], framework="pt") as handle:
            sdr = handle.get_tensor(record["sdr_key"])
        paired_values.append((hdr[0, 0, 0].item(), sdr[0, 0, 0].item()))
    assert paired_values == [(2000, 11), (3000, 22)]


def test_native_pairs_preserve_png_code_values(tmp_path, monkeypatch):
    sdr_root = tmp_path / "sdr"
    hdr_root = tmp_path / "hdr"
    sdr_root.mkdir()
    hdr_root.mkdir()
    sdr = np.arange(3 * 8 * 9, dtype=np.uint8).reshape(8, 9, 3)
    hdr = (np.arange(3 * 8 * 9, dtype=np.uint16).reshape(8, 9, 3) * 257)
    assert cv2.imwrite(str(sdr_root / "pair.png"), sdr)
    assert cv2.imwrite(str(hdr_root / "pair.png"), hdr)

    curate_module = importlib.import_module("lumaflux.data.curate")

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("native pairs must bypass normalization and degradation")

    monkeypatch.setattr(curate_module, "normalize_to_pq2020", forbidden)
    monkeypatch.setattr(curate_module, "degrade_chain", forbidden)
    manifest = curate(
        CurationConfig(
            sources=[
                SourceSpec(
                    name="native_demo",
                    path=str(hdr_root),
                    mode="native_pairs",
                    sdr_path=str(sdr_root),
                    source_dataset="HDRTV1K",
                )
            ],
            out_dir=str(tmp_path / "output"),
            workers=1,
        )
    )
    records = load_manifest(manifest)
    assert len(records) == 1
    assert records[0]["tmo"] == "native"
    assert records[0]["crf"] is None

    dataset = SdrHdrShards(manifest, crop_size=None, random_flip=False)
    item = dataset[0]
    assert item["variant"] == "native"
    assert torch.equal(item["sdr"], torch.from_numpy(sdr[..., ::-1].copy()).permute(2, 0, 1))
    assert torch.equal(item["hdr"], torch.from_numpy(hdr[..., ::-1].copy()).permute(2, 0, 1))


def test_native_pairs_require_matching_names_and_bit_depth(tmp_path):
    sdr_root = tmp_path / "sdr"
    hdr_root = tmp_path / "hdr"
    sdr_root.mkdir()
    hdr_root.mkdir()
    image8 = np.zeros((4, 4, 3), dtype=np.uint8)
    assert cv2.imwrite(str(sdr_root / "sdr_only.png"), image8)
    assert cv2.imwrite(str(hdr_root / "hdr_only.png"), image8)
    config = CurationConfig(
        sources=[
            SourceSpec(
                name="bad_native",
                path=str(hdr_root),
                mode="native_pairs",
                sdr_path=str(sdr_root),
            )
        ],
        out_dir=str(tmp_path / "output"),
        workers=1,
    )
    with pytest.raises(ValueError, match="filenames do not match"):
        curate(config)

    (sdr_root / "sdr_only.png").rename(sdr_root / "hdr_only.png")
    with pytest.raises(ValueError, match="expected uint16"):
        curate(config)
