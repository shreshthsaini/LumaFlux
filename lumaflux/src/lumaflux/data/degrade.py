"""Composite SDR degradation chain (Eq. 19): TMO -> gamut -> codec.

The codec stage Q_codec simulates broadcast/UGC encodings at CRF levels
{23, 31, 39}. We use ffmpeg/libx264 when available; otherwise we fall back
to a JPEG round-trip with a quality factor mapped from CRF, which produces
comparable block/ringing artifacts for single frames.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .tmo import apply_tmo

CRF_LEVELS = (23, 31, 39)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) [0,1] float -> (H,W,3) uint8."""
    arr = (x.clamp(0, 1) * 255.0).round().byte().cpu().numpy()
    return np.transpose(arr, (1, 2, 0))


def _from_uint8(arr: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    t = torch.from_numpy(np.transpose(arr, (2, 0, 1))).to(like.device, like.dtype)
    return t / 255.0


def _jpeg_roundtrip(arr: np.ndarray, crf: int) -> np.ndarray:
    from PIL import Image

    # CRF 23/31/39 -> JPEG quality ~ 75/45/20.
    quality = int(np.clip(np.interp(crf, [18, 23, 31, 39, 45], [90, 75, 45, 20, 12]), 5, 95))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _x264_roundtrip(frames: Sequence[np.ndarray], crf: int) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_in = td / "in.rgb"
        with open(raw_in, "wb") as f:
            for fr in frames:
                f.write(fr.tobytes())
        mp4 = td / "out.mp4"
        common = ["ffmpeg", "-y", "-loglevel", "error"]
        subprocess.run(
            common + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "24",
                      "-i", str(raw_in), "-c:v", "libx264", "-crf", str(crf),
                      "-pix_fmt", "yuv420p", str(mp4)],
            check=True,
        )
        out = subprocess.run(
            common + ["-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, stdout=subprocess.PIPE,
        ).stdout
    n = len(frames)
    arr = np.frombuffer(out, dtype=np.uint8)
    arr = arr[: n * h * w * 3].reshape(n, h, w, 3)
    return [arr[i].copy() for i in range(n)]


def codec_degrade(x_sdr: torch.Tensor, crf: int) -> torch.Tensor:
    """Quantize to 8-bit and run the codec round-trip.

    Args:
        x_sdr: ``(3,H,W)`` or ``(T,3,H,W)`` BT.709 signal in [0,1].
        crf: x264 constant-rate-factor (23 high / 31 mid / 39 low quality).
    """
    single = x_sdr.dim() == 3
    frames = x_sdr.unsqueeze(0) if single else x_sdr
    arrs = [_to_uint8(f) for f in frames]
    if ffmpeg_available():
        # x264 needs even dimensions for yuv420p.
        h, w = arrs[0].shape[:2]
        he, we = h - h % 2, w - w % 2
        cropped = [a[:he, :we] for a in arrs]
        outs = _x264_roundtrip(cropped, crf)
        outs = [np.pad(o, ((0, h - he), (0, w - we), (0, 0)), mode="edge") for o in outs]
    else:
        outs = [_jpeg_roundtrip(a, crf) for a in arrs]
    res = torch.stack([_from_uint8(o, x_sdr) for o in outs])
    return res[0] if single else res


def degrade_chain(
    x_pq: torch.Tensor,
    tmo: str,
    crf: int | None = None,
    peak_nits: float = 1000.0,
) -> torch.Tensor:
    """Full Eq. 19 chain: PQ/BT.2020 HDR -> degraded SDR BT.709 in [0,1]."""
    sdr = apply_tmo(tmo, x_pq, peak_nits=peak_nits)
    if crf is not None:
        sdr = codec_degrade(sdr, crf)
    return sdr
