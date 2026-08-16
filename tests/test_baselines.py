from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lumaflux.baselines.conversions import (
    anchor_relative_luminance,
    relative_linear_709_to_pq2020,
)
from lumaflux.color.gamut import M_709_TO_2020
from lumaflux.color.transfer import pq_eotf_nits, pq_oetf_nits
from lumaflux.evaluation.benchmark import run_precomputed_benchmark, save_benchmark_outputs
from lumaflux.utils.io import save_hdr_png16


def test_nit_anchor_maps_relative_neutral_percentile_to_1000_nits() -> None:
    relative = torch.full((3, 8, 8), 2.0)
    anchored, scale = anchor_relative_luminance(relative)
    assert scale.item() == pytest.approx(500.0)
    torch.testing.assert_close(anchored, torch.full_like(anchored, 1000.0))


def test_nit_anchor_black_is_finite_and_unchanged() -> None:
    anchored, scale = anchor_relative_luminance(torch.zeros(3, 4, 4))
    assert scale.item() == 1.0
    assert torch.count_nonzero(anchored) == 0


def test_relative_709_to_2020_pq_preserves_neutral_1000_nits() -> None:
    prediction, scale = relative_linear_709_to_pq2020(torch.full((3, 4, 4), 0.25))
    expected = pq_oetf_nits(torch.tensor(1000.0))
    assert scale.item() == pytest.approx(4000.0)
    torch.testing.assert_close(prediction, torch.full_like(prediction, expected), atol=2e-5, rtol=0)
    torch.testing.assert_close(
        pq_eotf_nits(prediction),
        torch.full_like(prediction, 1000.0),
        atol=0.2,
        rtol=0,
    )


def test_relative_709_to_2020_pq_applies_primary_matrix_before_pq() -> None:
    red = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    target = 1000.0
    prediction, scale = relative_linear_709_to_pq2020(
        red,
        percentile=100.0,
        target_nits=target,
    )
    expected_nits = (M_709_TO_2020[:, 0] * target).view(3, 1, 1).clamp(0.0, target)
    assert scale.item() == pytest.approx(target / 0.2126, rel=1e-6)
    torch.testing.assert_close(pq_eotf_nits(prediction), expected_nits, atol=0.2, rtol=1e-4)


def test_precomputed_video_benchmark_writes_all_csvs(tmp_path: Path) -> None:
    records = []
    for index, value in enumerate((0.2, 0.3)):
        frame = torch.full((3, 16, 16), value)
        prediction = tmp_path / f"prediction_{index}.png"
        reference = tmp_path / f"reference_{index}.png"
        save_hdr_png16(frame, prediction)
        save_hdr_png16(frame, reference)
        records.append(
            {
                "id": f"clip:{index}",
                "video": "clip",
                "frame_index": index,
                "variant": "reinhard_crf23",
                "prediction": prediction.name,
                "reference": reference.name,
            }
        )
    manifest = tmp_path / "predictions.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))

    result = run_precomputed_benchmark(
        manifest,
        include_optional=False,
        video_mode=True,
    )
    assert len(result.frame_rows) == 2
    assert len(result.video_rows) == 1
    assert result.video_rows[0]["n_frames"] == 2
    assert "flicker_raw" in result.video_rows[0]
    save_benchmark_outputs(result, tmp_path / "scores")
    assert {
        "availability.csv",
        "overall.csv",
        "per_frame.csv",
        "per_video.csv",
    } <= {path.name for path in (tmp_path / "scores").iterdir()}


def test_precomputed_single_frame_skips_temporal_metrics(tmp_path: Path) -> None:
    frame = torch.full((3, 16, 16), 0.25)
    prediction = tmp_path / "prediction.png"
    reference = tmp_path / "reference.png"
    save_hdr_png16(frame, prediction)
    save_hdr_png16(frame, reference)
    manifest = tmp_path / "predictions.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "single:0",
                "video": "single",
                "frame_index": 0,
                "variant": "expert",
                "prediction": prediction.name,
                "reference": reference.name,
            }
        )
        + "\n"
    )

    result = run_precomputed_benchmark(
        manifest,
        include_optional=False,
        video_mode=True,
    )
    assert len(result.frame_rows) == 1
    assert result.frame_rows[0]["psnr"] is not None
    assert len(result.video_rows) == 1
    video_row = result.video_rows[0]
    assert video_row["psnr"] == result.frame_rows[0]["psnr"]
    assert video_row["flicker_raw"] is None
    assert video_row["flicker_reference"] is None
    assert video_row["flicker_excess"] is None
    assert video_row["temporal"] == "skipped (single frame)"
