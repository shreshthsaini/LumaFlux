"""HDR video I/O: write 10-bit PQ/BT.2020 HEVC with mastering metadata.

Frames are PQ/BT.2020 signals in [0,1]. Output follows the paper's
delivery format: yuv420p10le HEVC tagged BT.2020 primaries / SMPTE-2084
transfer with 1,000-nit mastering display metadata.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from ..utils.io import ffmpeg_available, save_hdr_png16

# SMPTE ST 2086 mastering metadata (BT.2020 primaries, D65, 1000 nits).
_MASTER_DISPLAY = (
    "master-display="
    "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,1)"
    ":max-cll=1000,400"
)


@functools.lru_cache(maxsize=1)
def x265_supports_10bit() -> bool:
    """True if the ffmpeg on PATH has a 10-bit-capable libx265.

    8-bit-only x265 builds exist (ffmpeg then silently downgrades the encode
    to yuv420p, which would band badly under PQ), so the writer refuses to
    proceed rather than produce an 8-bit file tagged as HDR.
    """
    if not ffmpeg_available():
        return False
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", "encoder=libx265"],
        capture_output=True, text=True,
    ).stdout
    return "yuv420p10le" in out


def write_hdr_video(
    frames: Iterable[torch.Tensor] | Sequence[torch.Tensor],
    path: str | Path,
    fps: float = 24.0,
    crf: int = 16,
    peak_nits: int = 1000,
) -> Path:
    """Encode PQ/BT.2020 frames ((3,H,W) in [0,1]) to a 10-bit HEVC video."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required for HDR video export; "
                           "use save_hdr_png16 for frame output instead")
    if not x265_supports_10bit():
        raise RuntimeError(
            "the ffmpeg on PATH lacks a 10-bit libx265 (or libx265 entirely); "
            "it would silently encode 8-bit yuv420p, which bands badly under "
            "PQ. Install a 10-bit-capable build (e.g. `pip install "
            "imageio-ffmpeg` and put its binary first on PATH), or use "
            "write_hdr_frames for 16-bit PNG output."
        )
    frames = list(frames)
    if not frames:
        raise ValueError("no frames to encode")
    _, h, w = frames[0].shape
    he, we = h - h % 2, w - w % 2
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{we}x{he}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx265", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p10le",
        "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
        "-x265-params",
        f"colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:{_MASTER_DISPLAY}",
        "-tag:v", "hvc1",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fr in frames:
            arr = (fr[:, :he, :we].clamp(0, 1) * 65535.0).round().cpu().numpy()
            arr = arr.astype("<u2").transpose(1, 2, 0)
            proc.stdin.write(np.ascontiguousarray(arr).tobytes())
    finally:
        proc.stdin.close()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")
    return path


def write_hdr_frames(frames: Sequence[torch.Tensor], out_dir: str | Path) -> list[Path]:
    """Fallback writer: 16-bit PNG frames holding the PQ signal."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fr in enumerate(frames):
        p = out_dir / f"hdr_{i:06d}.png"
        save_hdr_png16(fr, p)
        paths.append(p)
    return paths
