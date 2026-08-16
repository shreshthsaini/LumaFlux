"""HDR video I/O: write 10-bit PQ/BT.2020 HEVC with mastering metadata.

Frames are PQ/BT.2020 signals in [0,1]. Output follows the paper's
delivery format: yuv420p10le HEVC tagged BT.2020 primaries / SMPTE-2084
transfer with 1,000-nit mastering display metadata.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..color.transfer import CORPUS_PEAK_NITS, CORPUS_PEAK_PQ
from ..utils.io import ffmpeg_available, save_hdr_png16


def _master_display(peak_nits: float) -> str:
    maximum = int(round(peak_nits * 10_000.0))
    max_fall = min(int(round(peak_nits)), 400)
    return (
        "master-display="
        f"G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L({maximum},1)"
        f":max-cll={int(round(peak_nits))},{max_fall}"
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


class HDRVideoWriter:
    """Streaming ffmpeg writer for PQ/BT.2020 frames."""

    def __init__(
        self,
        path: str | Path,
        height: int,
        width: int,
        fps: float = 24.0,
        crf: int = 16,
        peak_nits: float = CORPUS_PEAK_NITS,
    ) -> None:
        if peak_nits != CORPUS_PEAK_NITS:
            raise ValueError(
                f"LumaFlux output mastering peak is fixed at {CORPUS_PEAK_NITS:.0f} cd/m2"
            )
        self.path = Path(path)
        self.height = height - height % 2
        self.width = width - width % 2
        self._closed = False
        self._proc: subprocess.Popen | None = None

        self._validate_ffmpeg()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb48le",
            "-s", f"{self.width}x{self.height}", "-r", str(fps),
            "-i", "-", "-threads", "4",
            "-c:v", "libx265", "-crf", str(crf), "-preset", "medium",
            "-pix_fmt", "yuv420p10le",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc", "-x265-params",
            "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
            f"{_master_display(peak_nits)}",
            "-tag:v", "hvc1", str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    @staticmethod
    def _validate_ffmpeg() -> None:
        if not ffmpeg_available():
            raise RuntimeError(
                "ffmpeg is required for HDR video export; use save_hdr_png16 "
                "for frame output instead"
            )
        if not x265_supports_10bit():
            raise RuntimeError(
                "the ffmpeg on PATH lacks a 10-bit libx265 (or libx265 entirely); "
                "it would silently encode 8-bit yuv420p, which bands badly under "
                "PQ. Install a 10-bit-capable build, or use write_hdr_frames for "
                "16-bit PNG output."
            )

    def write(self, frame: torch.Tensor) -> None:
        if self._closed or self._proc is None or self._proc.stdin is None:
            raise RuntimeError("cannot write to a closed HDR video writer")
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(f"expected a (3,H,W) frame, got {tuple(frame.shape)}")
        if frame.shape[-2] < self.height or frame.shape[-1] < self.width:
            raise ValueError(
                f"frame is smaller than writer size {(self.height, self.width)}: "
                f"{tuple(frame.shape[-2:])}"
            )
        arr = (
            frame[:, : self.height, : self.width].clamp(0, CORPUS_PEAK_PQ) * 65535.0
        ).round().cpu().numpy()
        arr = arr.astype("<u2").transpose(1, 2, 0)
        self._proc.stdin.write(np.ascontiguousarray(arr).tobytes())

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        if self._proc is None or self._proc.stdin is None:
            return self.path
        self._proc.stdin.close()
        ret = self._proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}")
        return self.path

    def __enter__(self) -> "HDRVideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def write_hdr_video(
    frames: Iterable[torch.Tensor],
    path: str | Path,
    fps: float = 24.0,
    crf: int = 16,
    peak_nits: int = int(CORPUS_PEAK_NITS),
) -> Path:
    """Stream PQ/BT.2020 frames ((3,H,W) in [0,1]) to 10-bit HEVC."""
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("no frames to encode") from exc

    _, height, width = first.shape
    with HDRVideoWriter(
        path,
        height,
        width,
        fps=fps,
        crf=crf,
        peak_nits=peak_nits,
    ) as writer:
        writer.write(first)
        for frame in iterator:
            writer.write(frame)
    return Path(path)


def write_hdr_frames(frames: Iterable[torch.Tensor], out_dir: str | Path) -> list[Path]:
    """Write frames as they arrive to 16-bit PNG files holding the PQ signal."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        path = out_dir / f"hdr_{i:06d}.png"
        save_hdr_png16(frame, path)
        paths.append(path)
    return paths
