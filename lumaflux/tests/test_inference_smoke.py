import torch

from lumaflux.evaluation import aggregate, delta_e_itp, pu21_psnr, pu21_ssim, score_pair, to_markdown
from lumaflux.inference.pipeline import LumaFluxPipeline
from lumaflux.models.factory import build_model

TINY = {
    "model": {
        "backbone": "tiny", "siglip": "tiny", "rank": 4, "num_knots": 8,
        "phys_channels": 8, "stats_dim": 8, "num_bands": 4,
        "num_null_tokens": 4, "modulation_hidden": 32,
    }
}


def _pipe():
    torch.manual_seed(0)
    return LumaFluxPipeline(build_model(TINY))


def test_pipeline_end_to_end():
    pipe = _pipe()
    sdr = torch.rand(1, 3, 64, 64)
    gen = torch.Generator().manual_seed(0)
    out = pipe(sdr, num_steps=4, generator=gen)
    hdr = out["hdr"]
    assert hdr.shape == (1, 3, 64, 64)
    assert torch.isfinite(hdr).all()
    assert hdr.min() >= 0.0 and hdr.max() <= 1.0  # valid PQ signal


def test_pipeline_deterministic_with_seed():
    pipe = _pipe()
    sdr = torch.rand(1, 3, 64, 64)
    a = pipe(sdr, num_steps=2, generator=torch.Generator().manual_seed(7))["hdr"]
    b = pipe(sdr, num_steps=2, generator=torch.Generator().manual_seed(7))["hdr"]
    assert torch.allclose(a, b)


def test_pipeline_rejects_bad_size():
    pipe = _pipe()
    try:
        pipe(torch.rand(1, 3, 50, 50), num_steps=1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_metrics_sane():
    x = torch.rand(3, 32, 32)
    assert pu21_psnr(x, x).item() > 60
    assert delta_e_itp(x, x).item() < 1e-3
    assert pu21_ssim(x, x).item() > 0.99
    noisy = (x + 0.05 * torch.randn_like(x)).clamp(0, 1)
    assert pu21_psnr(x, noisy).item() < pu21_psnr(x, x).item()
    assert delta_e_itp(x, noisy).item() > delta_e_itp(x, x).item()


def test_score_and_aggregate():
    x = torch.rand(3, 16, 16)
    row = score_pair(x, x.clone(), with_optional=False)
    rows = [dict(id="a", variant="reinhard_crf31", category="pgc", **row)]
    agg = aggregate(rows)
    md = to_markdown(agg)
    assert "reinhard_crf31" in md and "ALL" in md


def test_hdr_video_writer(tmp_path):
    from lumaflux.utils.io import ffmpeg_available

    frames = [torch.rand(3, 64, 64) for _ in range(3)]
    if ffmpeg_available():
        import pytest

        from lumaflux.inference.video import write_hdr_video, x265_supports_10bit

        if not x265_supports_10bit():
            # 8-bit-only x265 builds must be rejected, not silently downgraded.
            with pytest.raises(RuntimeError, match="10-bit"):
                write_hdr_video(frames, tmp_path / "out.mp4", fps=24)
            return
        out = write_hdr_video(frames, tmp_path / "out.mp4", fps=24)
        assert out.exists() and out.stat().st_size > 0
        # Verify HDR signaling + true 10-bit encoding in the bitstream.
        import subprocess

        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_streams", str(out)],
            capture_output=True, text=True,
        ).stdout
        assert "smpte2084" in probe and "bt2020" in probe
        assert "yuv420p10le" in probe
    else:
        from lumaflux.inference.video import write_hdr_frames

        paths = write_hdr_frames(frames, tmp_path / "frames")
        assert len(paths) == 3


def test_demo_app_builds():
    import importlib.util

    if importlib.util.find_spec("gradio") is None:
        import pytest

        pytest.skip("gradio not installed")
    from lumaflux.demo.app import build_demo

    pipe = _pipe()
    demo = build_demo(pipe, mult=2 * pipe.model.vae_scale)
    assert demo is not None
