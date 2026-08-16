"""High-bit-depth video decoding for HDR curation with system ffmpeg."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = Path(shutil.which("ffprobe") or "ffprobe")
FFMPEG_THREADS = 4

_UNKNOWN_COLOR_METADATA = {"", "unknown", "unspecified", "reserved"}
_COLOR_MATRIX_ALIASES = {
    "bt470bg": "bt601",
    "smpte170m": "bt601",
    "bt2020nc": "bt2020",
}
_COLOR_RANGE_ALIASES = {
    "tv": "limited",
    "mpeg": "limited",
    "pc": "full",
    "jpeg": "full",
}


def ffmpeg_env() -> dict[str, str]:
    """Return the environment used for ffmpeg subprocesses."""
    return os.environ.copy()


def ffmpeg_available() -> bool:
    """Return whether the system ffmpeg and ffprobe executables are runnable."""
    if shutil.which(str(FFMPEG)) is None or shutil.which(str(FFPROBE)) is None:
        return False
    try:
        return subprocess.run(
            [str(FFMPEG), "-version"],
            env=ffmpeg_env(),
            capture_output=True,
            timeout=20,
            check=False,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def probe_video(path: str | Path) -> dict:
    """Read the first video stream metadata with ffprobe."""
    command = [
        str(FFPROBE),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=width,height,pix_fmt,color_space,color_transfer,color_primaries,"
            "color_range,r_frame_rate"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        env=ffmpeg_env(),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {path}")
    return streams[0]


def _input_color_matrix(metadata: dict, untagged_default: str) -> str:
    """Resolve ffmpeg's input matrix while preserving explicit stream metadata."""
    # swscale otherwise treats untagged LIVE-TMHDR media as BT.601.
    tagged = str(metadata.get("color_space") or "").lower()
    if tagged in _UNKNOWN_COLOR_METADATA:
        return untagged_default
    return _COLOR_MATRIX_ALIASES.get(tagged, tagged)


def _input_color_range(metadata: dict) -> str:
    """Resolve ffmpeg's input range, treating untagged YUV video as limited range."""
    tagged = str(metadata.get("color_range") or "").lower()
    if tagged in _UNKNOWN_COLOR_METADATA:
        return "limited"
    return _COLOR_RANGE_ALIASES.get(tagged, tagged)


def _decode_command(
    path: str | Path,
    *,
    fps: float | None,
    max_frames: int | None,
    pix_fmt: str,
    metadata: dict,
    in_color_matrix: str,
) -> list[str]:
    filters = []
    if fps is not None:
        filters.append(f"fps={fps:g}")
    matrix = _input_color_matrix(metadata, in_color_matrix)
    color_range = _input_color_range(metadata)
    filters.append(f"scale=in_color_matrix={matrix}:in_range={color_range}")
    command = [
        str(FFMPEG),
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        str(FFMPEG_THREADS),
        "-i",
        str(path),
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(filters),
    ]
    if max_frames is not None:
        command += ["-frames:v", str(max_frames)]
    command += [
        "-threads",
        str(FFMPEG_THREADS),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "pipe:1",
    ]
    return command


def _decode_raw(
    path: str | Path,
    *,
    fps: float | None,
    max_frames: int | None,
    pix_fmt: str,
    dtype: np.dtype,
    in_color_matrix: str,
) -> Iterator[tuple[int, torch.Tensor]]:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg and ffprobe must be available on PATH")
    metadata = probe_video(path)
    width, height = int(metadata["width"]), int(metadata["height"])
    bytes_per_value = np.dtype(dtype).itemsize
    frame_bytes = width * height * 3 * bytes_per_value
    command = _decode_command(
        path,
        fps=fps,
        max_frames=max_frames,
        pix_fmt=pix_fmt,
        metadata=metadata,
        in_color_matrix=in_color_matrix,
    )
    process = subprocess.Popen(
        command,
        env=ffmpeg_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        frame_idx = 0
        while True:
            raw = process.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError(
                    f"Short ffmpeg frame for {path}: got {len(raw)} of {frame_bytes} bytes"
                )
            array = np.frombuffer(raw, dtype=dtype).reshape(height, width, 3).copy()
            yield frame_idx, torch.from_numpy(array).permute(2, 0, 1).contiguous()
            frame_idx += 1
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait(timeout=30)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg decode failed for {path}: {stderr.strip()}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def decode_hdr_video(
    path: str | Path,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield PQ/BT.2020 RGB frames as full-range ``uint16`` CHW tensors."""
    yield from _decode_raw(
        path,
        fps=fps,
        max_frames=max_frames,
        pix_fmt="rgb48le",
        dtype=np.dtype("<u2"),
        in_color_matrix="bt2020",
    )


def decode_sdr_video(
    path: str | Path,
    *,
    fps: float | None = None,
    max_frames: int | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield BT.709 RGB frames as ``uint8`` CHW tensors."""
    yield from _decode_raw(
        path,
        fps=fps,
        max_frames=max_frames,
        pix_fmt="rgb24",
        dtype=np.dtype("u1"),
        in_color_matrix="bt709",
    )
