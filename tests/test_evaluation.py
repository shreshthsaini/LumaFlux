from __future__ import annotations

import json
import importlib

import pytest
import torch
from torch import nn

from lumaflux.color.spaces import pu21_encode
from lumaflux.color.transfer import pq_oetf_nits
from lumaflux.evaluation.backends import EvalBackendUnavailable
from lumaflux.evaluation.diagnostics import diagnostic_ref_max_rescale, pq_code_statistics
from lumaflux.evaluation.benchmark import (
    OPTIONAL_METRICS,
    aggregate,
    evaluate_precomputed_manifest,
    probe_metric_backends,
    run_benchmark,
    save_benchmark_outputs,
    save_csv,
    to_markdown,
)
from lumaflux.evaluation.exposure import (
    apply_exposure_gain,
    fit_exposure_gain,
    score_exposure_calibrated_content,
)
from lumaflux.evaluation.fr_hidrovqa import (
    HIDROFeatureExtractor,
    build_hidrovqa_model,
    fr_hidrovqa,
)
from lumaflux.evaluation.hdrvdp3 import hdr_vdp3
from lumaflux.evaluation.temporal import (
    TemporalAccumulator,
    pu21_flicker_energy,
    temporal_consistency,
)
from lumaflux.utils.io import save_hdr_png16, save_sdr_png


@pytest.fixture
def random_hidro_backbone() -> HIDROFeatureExtractor:
    torch.manual_seed(7)
    encoder = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    )
    return HIDROFeatureExtractor(encoder, n_features=8, projection_dim=4).eval()


def test_pu21_official_banding_glare_reference() -> None:
    luminance = torch.tensor([0.005, 100.0, 10_000.0], dtype=torch.float64)
    expected = torch.tensor([0.0, 256.3839, 595.3939], dtype=torch.float64)
    torch.testing.assert_close(pu21_encode(luminance), expected, atol=0.05, rtol=0.0)


def test_pq_output_statistics_and_diagnostic_rescale() -> None:
    prediction = torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1)
    reference = prediction * 2.0
    statistics = pq_code_statistics(prediction)
    rescaled, factor = diagnostic_ref_max_rescale(prediction, reference)
    assert statistics["mean_pq_code"] == pytest.approx(0.2)
    assert statistics["max_pq_code"] == pytest.approx(0.3)
    assert factor == pytest.approx(2.0)
    assert torch.allclose(rescaled, reference)


def test_exposure_calibration_recovers_gain_clamps_and_preserves_identity() -> None:
    prediction_nits = torch.tensor(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )
    prediction = pq_oetf_nits(prediction_nits)
    target = pq_oetf_nits(prediction_nits * 4.0)

    assert fit_exposure_gain([prediction], [target]) == pytest.approx(4.0, rel=1e-5)
    torch.testing.assert_close(apply_exposure_gain(prediction, 4.0), target)
    assert fit_exposure_gain(
        [prediction], [pq_oetf_nits(prediction_nits * 100.0)]
    ) == pytest.approx(8.0)
    assert fit_exposure_gain(
        [prediction], [pq_oetf_nits(prediction_nits * 0.01)]
    ) == pytest.approx(0.25)
    assert fit_exposure_gain([prediction], [prediction]) == pytest.approx(1.0)
    assert torch.equal(apply_exposure_gain(prediction, 1.0), prediction)

    gain, scores = score_exposure_calibrated_content(
        [prediction],
        [target],
        lambda calibrated, reference: {
            "mse": (calibrated - reference).square().mean().item()
        },
    )
    assert gain == pytest.approx(4.0, rel=1e-5)
    assert scores[0]["mse"] == pytest.approx(0.0, abs=1e-10)


def test_exposure_calibration_is_reported_for_expert_track_only(tmp_path) -> None:
    prediction_nits = torch.full((3, 16, 16), 25.0)
    prediction = pq_oetf_nits(prediction_nits)
    reference = pq_oetf_nits(prediction_nits * 2.0)
    save_hdr_png16(prediction, tmp_path / "prediction.png")
    save_hdr_png16(reference, tmp_path / "reference.png")
    records = [
        {
            "id": "expert-frame",
            "content_id": "clip-a",
            "track": "expert",
            "variant": "method",
            "prediction": "prediction.png",
            "reference": "reference.png",
        },
        {
            "id": "synthetic-frame",
            "content_id": "clip-b",
            "track": "synthetic",
            "variant": "method",
            "prediction": "prediction.png",
            "reference": "reference.png",
        },
    ]
    manifest = tmp_path / "pairs.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    rows = evaluate_precomputed_manifest(manifest, with_optional=False)

    expert, synthetic = rows
    assert expert["headline_protocol"] == "exposure_calibrated"
    assert expert["exposure_gain"] == pytest.approx(2.0, rel=2e-3)
    assert "exposure_calibrated_psnr" in expert
    assert synthetic["headline_protocol"] == "raw"
    assert "exposure_calibrated_psnr" not in synthetic


def test_fr_hidrovqa_with_random_backbone(random_hidro_backbone) -> None:
    torch.manual_seed(11)
    reference = torch.rand(2, 3, 24, 24)
    identical = fr_hidrovqa(reference, reference.clone(), model=random_hidro_backbone)
    distorted = fr_hidrovqa(
        (reference + 0.15 * torch.randn_like(reference)).clamp(0, 1),
        reference,
        model=random_hidro_backbone,
    )
    assert identical.item() == pytest.approx(0.0, abs=1e-8)
    assert distorted.item() > 0.0


def test_fr_hidrovqa_missing_checkpoint_raises(tmp_path, random_hidro_backbone) -> None:
    with pytest.raises(EvalBackendUnavailable, match="checkpoint missing"):
        build_hidrovqa_model(
            tmp_path / "missing.tar",
            encoder=random_hidro_backbone.encoder,
            n_features=8,
            projection_dim=4,
        )


def test_hdr_lpips_missing_cached_backbone_raises(monkeypatch, tmp_path) -> None:
    module = importlib.import_module("lumaflux.evaluation.hdr_lpips")
    module._lpips_models.clear()
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    frame = torch.full((3, 64, 64), 0.5)
    with pytest.raises(EvalBackendUnavailable, match="weights are not cached"):
        module.hdr_lpips(frame, frame)


def test_temporal_flicker_raw_reference_and_excess() -> None:
    reference = torch.stack(
        [torch.full((3, 8, 8), value) for value in (0.1, 0.3, 0.2)]
    )
    prediction = torch.full_like(reference, 0.2)
    result = temporal_consistency(prediction, reference)
    assert result["flicker_raw"].item() == pytest.approx(0.0, abs=1e-8)
    assert result["flicker_reference"].item() == pytest.approx(
        pu21_flicker_energy(reference).item()
    )
    assert result["flicker_excess"].item() == pytest.approx(
        -result["flicker_reference"].item()
    )


def test_temporal_streaming_matches_batch() -> None:
    torch.manual_seed(3)
    prediction = torch.rand(4, 3, 8, 8)
    reference = torch.rand(4, 3, 8, 8)
    expected = temporal_consistency(prediction, reference)
    accumulator = TemporalAccumulator()
    for pred_frame, ref_frame in zip(prediction, reference):
        accumulator.update(pred_frame, ref_frame)
    actual = accumulator.compute()
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value.item(), rel=2e-6, abs=1e-5)


def test_probe_records_one_explicit_skip_and_markdown_footer(tmp_path) -> None:
    calls = {name: 0 for name in OPTIONAL_METRICS}

    def available(name):
        def probe():
            calls[name] += 1

        return probe

    def unavailable():
        calls["hdr_vdp3"] += 1
        raise EvalBackendUnavailable("hdr_vdp3", "no MATLAB or Octave")

    availability = probe_metric_backends(
        video_mode=True,
        probes={
            "hdr_lpips": available("hdr_lpips"),
            "hdr_vdp3": unavailable,
            "fr_hidrovqa": available("fr_hidrovqa"),
        },
    )
    assert calls == {name: 1 for name in OPTIONAL_METRICS}
    assert availability["hdr_vdp3"] == "no MATLAB or Octave"

    markdown = to_markdown(
        {"ALL": {"psnr": 42.0}},
        availability,
        columns=["psnr", "hdr_vdp3"],
    )
    assert "hdr_vdp3: unavailable: no MATLAB or Octave" in markdown

    output = tmp_path / "summary.csv"
    save_csv(
        [{"variant": "ALL", "psnr": 42.0, "hdr_vdp3": None}],
        output,
        metric_columns=["psnr", "hdr_vdp3"],
        availability=availability,
    )
    assert "unavailable: no MATLAB or Octave" in output.read_text()


def test_hdrvdp3_missing_executable_raises(monkeypatch, tmp_path) -> None:
    toolbox = tmp_path / "hdrvdp3"
    (toolbox / "npy-matlab").mkdir(parents=True)
    (toolbox / "hdrvdp3.m").touch()
    (toolbox / "npy-matlab/readNPY.m").touch()
    monkeypatch.delenv("HDRVDP3_BACKEND", raising=False)
    monkeypatch.setenv("HDRVDP3_PATH", str(toolbox))
    monkeypatch.setattr("lumaflux.evaluation.hdrvdp3.shutil.which", lambda _: None)
    frame = torch.full((3, 8, 8), 0.5)
    with pytest.raises(EvalBackendUnavailable, match="neither matlab nor octave"):
        hdr_vdp3(frame, frame)


def test_video_benchmark_subsamples_per_video(tmp_path) -> None:
    records = []
    for index, value in enumerate((0.1, 0.2, 0.3, 0.4)):
        hdr_path = tmp_path / f"hdr_{index}.png"
        sdr_path = tmp_path / f"sdr_{index}.png"
        frame = torch.full((3, 16, 16), value)
        save_hdr_png16(frame, hdr_path)
        save_sdr_png(frame, sdr_path)
        records.append(
            {
                "id": f"clip_frame_{index}",
                "video_id": "clip",
                "frame_index": index,
                "category": "pgc",
                "hdr": hdr_path.name,
                "sdr_variants": {"identity": sdr_path.name},
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    availability = probe_metric_backends(include_optional=False, video_mode=True)

    result = run_benchmark(
        manifest,
        lambda sdr: sdr,
        include_optional=False,
        subsample=2,
        video_mode=True,
        availability=availability,
    )
    assert len(result.frame_rows) == 2
    assert [row["frame_index"] for row in result.frame_rows] == [0, 2]
    assert len(result.video_rows) == 1
    assert result.video_rows[0]["n_frames"] == 2
    assert result.video_rows[0]["flicker_raw"] >= 0.0
    assert "ALL" in aggregate(result.video_rows)

    output_dir = tmp_path / "reports"
    save_benchmark_outputs(result, output_dir)
    expected = {
        "availability.csv",
        "overall.csv",
        "per_frame.csv",
        "per_video.csv",
        "per_video.md",
        "summary.md",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert "unavailable: disabled by --no-optional-metrics" in (
        output_dir / "overall.csv"
    ).read_text()
