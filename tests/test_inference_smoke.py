import inspect

import torch
import pytest

from lumaflux.evaluation import aggregate, delta_e_itp, pu21_psnr, pu21_ssim, score_pair, to_markdown
from lumaflux.inference.pipeline import LumaFluxPipeline
from lumaflux.models.factory import build_model
from lumaflux.models.luma_flux import pack_latents

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


def test_transport_starts_at_training_matched_noisy_sdr_at_t_one(monkeypatch):
    bridge_noise = 0.125
    pipe = LumaFluxPipeline(build_model(TINY), cfg={"train": {"bridge_noise": bridge_noise}})
    sdr = torch.rand(1, 3, 64, 64)
    z_sdr = pipe.model.encode_image(sdr)
    seed = 17
    eps = torch.randn(z_sdr.shape, generator=torch.Generator().manual_seed(seed))
    seen_start = None
    seen_times = []

    def zero_velocity(z, t, cond, latent_hw):
        nonlocal seen_start
        if seen_start is None:
            seen_start = z.clone()
        seen_times.append(t.clone())
        return torch.zeros_like(z)

    monkeypatch.setattr(pipe.model, "velocity", zero_velocity)
    out = pipe(
        sdr,
        num_steps=4,
        generator=torch.Generator().manual_seed(seed),
        return_intermediate=True,
    )
    expected_start = pack_latents(z_sdr + bridge_noise * eps)
    assert torch.equal(seen_start, expected_start)
    assert torch.equal(out["bridge_noise"], eps)
    assert torch.equal(out["trajectory"][0], expected_start)
    assert torch.equal(
        torch.stack([time[0] for time in seen_times]),
        torch.tensor([1.0, 0.75, 0.5, 0.25]),
    )


def test_strength_is_removed_from_public_api_with_deprecation_error():
    pipe = _pipe()
    parameters = inspect.signature(pipe.__call__).parameters
    assert "strength" not in parameters
    assert parameters["num_steps"].default == 8
    with pytest.raises(TypeError, match="strength was removed"):
        pipe(torch.rand(1, 3, 64, 64), strength=0.4)


def test_explicit_bridge_noise_can_be_shared():
    pipe = _pipe()
    sdr = torch.rand(1, 3, 64, 64)
    first = pipe(sdr, generator=torch.Generator().manual_seed(7))
    second = pipe(
        sdr,
        noise=first["bridge_noise"],
        generator=torch.Generator().manual_seed(999),
    )
    assert torch.equal(first["hdr"], second["hdr"])


def test_pipeline_returns_applied_spline_params():
    pipe = _pipe()
    out = pipe(torch.rand(1, 3, 64, 64), num_steps=2)
    assert set(out["spline_params"]) == {"widths", "heights", "derivs"}
    assert out["spline_params"]["widths"].shape == (1, 8)
    assert out["spline_params"]["derivs"].shape == (1, 9)


def test_rqs_parameter_ema_returns_actually_applied_values():
    pipe = _pipe()
    sdr = torch.rand(1, 3, 64, 64)
    noise = torch.zeros_like(pipe.model.encode_image(sdr))
    current = pipe(sdr, noise=noise, num_steps=2)["spline_params"]
    previous = {name: torch.full_like(value, 2.0) for name, value in current.items()}
    smoothed = pipe(
        sdr,
        noise=noise,
        num_steps=2,
        previous_spline_params=previous,
        spline_ema=0.8,
    )["spline_params"]
    for name in current:
        expected = 0.8 * previous[name] + 0.2 * current[name]
        assert torch.allclose(smoothed[name], expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_steps": 0}, "num_steps"),
        ({"spline_ema": 1.0}, "spline_ema"),
    ],
)
def test_pipeline_rejects_invalid_sampling_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _pipe()(torch.rand(1, 3, 64, 64), **kwargs)


def test_video_writer_consumes_frames_incrementally(monkeypatch, tmp_path):
    import lumaflux.inference.video as video

    events = []

    class FakeWriter:
        def __init__(self, path, height, width, fps=24.0, crf=16, peak_nits=1000.0):
            assert peak_nits == 1000.0
            events.append("open")

        def __enter__(self):
            return self

        def write(self, frame):
            events.append(f"write{int(frame[0, 0, 0])}")

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("close")

    def frames():
        events.append("yield0")
        yield torch.zeros(3, 2, 2)
        events.append("yield1")
        yield torch.ones(3, 2, 2)

    monkeypatch.setattr(video, "HDRVideoWriter", FakeWriter)
    video.write_hdr_video(frames(), tmp_path / "out.mp4")
    assert events == ["yield0", "open", "write0", "yield1", "write1", "close"]


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
