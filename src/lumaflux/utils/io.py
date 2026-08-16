"""Image and video I/O helpers.

HDR frames are stored as 16-bit PNGs holding the PQ/BT.2020 signal
(code values = signal * 65535); SDR frames as 8-bit PNGs holding the
BT.709 signal. This keeps the dataset codec-independent and lossless.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


def save_sdr_png(x: torch.Tensor, path: str | Path) -> None:
    """(3,H,W) [0,1] BT.709 signal -> 8-bit PNG."""
    arr = (x.clamp(0, 1) * 255.0).round().byte().cpu().numpy().transpose(1, 2, 0)
    iio.imwrite(Path(path), arr)


def load_sdr_png(path: str | Path) -> torch.Tensor:
    arr = iio.imread(Path(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr[..., :3].transpose(2, 0, 1)).float() / 255.0


def save_hdr_png16(x_pq: torch.Tensor, path: str | Path) -> None:
    """(3,H,W) [0,1] PQ/BT.2020 signal -> 16-bit PNG (via OpenCV; PIL has no
    16-bit RGB support)."""
    import cv2

    arr = (x_pq.clamp(0, 1) * 65535.0).round().cpu().numpy().astype(np.uint16)
    bgr = arr.transpose(1, 2, 0)[..., ::-1]
    if not cv2.imwrite(str(path), np.ascontiguousarray(bgr)):
        raise IOError(f"failed to write {path}")


def load_hdr_png16(path: str | Path) -> torch.Tensor:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise IOError(f"failed to read {path}")
    arr = bgr[..., :3][..., ::-1].astype(np.float32)
    scale = 65535.0 if bgr.dtype == np.uint16 else 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()) / scale


@functools.lru_cache(maxsize=1)
def ffmpeg_available() -> bool:
    """True if an ffmpeg binary is on PATH *and* actually runs (a present but
    broken install - e.g. missing shared libraries - counts as unavailable)."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        return subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, timeout=20
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def extract_video_frames(
    video: str | Path,
    out_dir: str | Path,
    fps: float | None = None,
    bit_depth: int = 16,
    max_frames: int | None = None,
) -> list[Path]:
    """Decode a video to PNG frames with ffmpeg (16-bit for HDR sources)."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required to decode videos")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video)]
    filters = []
    if fps is not None:
        filters.append(f"fps={fps}")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    pix = "rgb48be" if bit_depth == 16 else "rgb24"
    cmd += ["-pix_fmt", pix, str(out_dir / "frame_%06d.png")]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.png"))
