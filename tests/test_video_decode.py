import subprocess
import importlib

import numpy as np
import pytest
import torch

from lumaflux.data.video_decode import (
    FFMPEG,
    FFMPEG_THREADS,
    _decode_command,
    decode_hdr_video,
    decode_sdr_video,
    ffmpeg_available,
    ffmpeg_env,
)


def test_constructed_decode_args_use_defaults_and_explicit_tags(monkeypatch):
    untagged = {"color_space": "unknown", "color_range": "unknown"}
    sdr_command = _decode_command(
        "untagged.mp4",
        fps=1.0,
        max_frames=2,
        pix_fmt="rgb24",
        metadata=untagged,
        in_color_matrix="bt709",
    )
    assert sdr_command[sdr_command.index("-vf") + 1] == (
        "fps=1,scale=in_color_matrix=bt709:in_range=limited"
    )

    tagged_command = _decode_command(
        "tagged.mp4",
        fps=None,
        max_frames=None,
        pix_fmt="rgb48le",
        metadata={"color_space": "bt2020nc", "color_range": "pc"},
        in_color_matrix="bt709",
    )
    assert tagged_command[tagged_command.index("-vf") + 1] == (
        "scale=in_color_matrix=bt2020:in_range=full"
    )

    calls = []
    module = importlib.import_module("lumaflux.data.video_decode")

    def fake_decode(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(())

    monkeypatch.setattr(module, "_decode_raw", fake_decode)
    list(decode_sdr_video("sdr.mp4"))
    list(decode_hdr_video("hdr.mp4"))
    assert calls[0][1]["in_color_matrix"] == "bt709"
    assert calls[1][1]["in_color_matrix"] == "bt2020"


@pytest.mark.skipif(not ffmpeg_available(), reason="high-bit-depth ffmpeg is unavailable")
def test_rgb48le_pipe_preserves_more_than_eight_bits(tmp_path):
    width, height = 512, 16
    ramp = np.linspace(0, 65535, width, dtype=np.uint16)
    rgb = np.broadcast_to(ramp[None, :, None], (height, width, 3)).copy()
    raw = tmp_path / "ramp.rgb48le"
    raw.write_bytes(rgb.astype("<u2").tobytes())
    clip = tmp_path / "ramp.mkv"
    command = [
        str(FFMPEG),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(FFMPEG_THREADS),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb48le",
        "-s",
        f"{width}x{height}",
        "-r",
        "1",
        "-i",
        str(raw),
        "-frames:v",
        "1",
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv444p10le",
        "-threads",
        str(FFMPEG_THREADS),
        "-x265-params",
        "lossless=1:pools=1:frame-threads=1",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
        str(clip),
    ]
    subprocess.run(command, env=ffmpeg_env(), check=True, timeout=30)

    frames = list(decode_hdr_video(clip))
    assert len(frames) == 1
    decoded = frames[0][1]
    assert decoded.dtype == torch.uint16
    values = decoded[0].to(torch.int32)
    assert values.max().item() > 255
    assert torch.unique(values).numel() > 256
    assert torch.any(values.remainder(257) != 0)
