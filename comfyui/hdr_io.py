"""HDR helpers with no ComfyUI dependency: a 16-bit PNG writer that tags the
file as PQ/BT.2020, a matching reader for tests, padding, and an SDR preview
tone map for the PQ output."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import torch

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# PNG cICP chunk (PNG third edition): colour primaries 9 (BT.2020), transfer
# characteristics 16 (PQ, SMPTE ST 2084), matrix coefficients 0 (RGB), full range.
CICP_PQ_BT2020 = bytes([9, 16, 0, 1])


def _chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def write_png16(path, rgb16: np.ndarray, tag_pq: bool = True, compress_level: int = 6) -> None:
    """Write an (H, W, 3) uint16 array as a 16-bit RGB PNG.

    With ``tag_pq`` the file carries a cICP chunk declaring PQ / BT.2020, so a
    viewer that reads it (Chrome, recent macOS) renders the picture as HDR rather
    than as a flat SDR image. Viewers that ignore cICP still open the file.
    """
    if rgb16.ndim != 3 or rgb16.shape[2] != 3 or rgb16.dtype != np.uint16:
        raise ValueError(f"expected an (H, W, 3) uint16 array, got {rgb16.shape} {rgb16.dtype}")
    h, w, _ = rgb16.shape
    rows = np.ascontiguousarray(rgb16.astype(">u2")).view(np.uint8).reshape(h, w * 6)
    filtered = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    parts = [PNG_SIGNATURE, _chunk(b"IHDR", ihdr)]
    if tag_pq:
        parts.append(_chunk(b"cICP", CICP_PQ_BT2020))
    parts.append(_chunk(b"IDAT", zlib.compress(filtered, compress_level)))
    parts.append(_chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(b"".join(parts))


def read_png16(path) -> tuple[np.ndarray, bytes | None]:
    """Read a 16-bit RGB PNG written by :func:`write_png16` (filter type 0 only).

    Returns the (H, W, 3) uint16 array and the raw cICP payload, if any.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG file")
    pos, idat, cicp, width, height = 8, [], None, None, None
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", body[:10])
            if depth != 16 or colour != 2:
                raise ValueError("only 16-bit RGB PNGs are supported")
        elif tag == b"cICP":
            cicp = body
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(height, 1 + width * 6)
    if (raw[:, 0] != 0).any():
        raise ValueError("only filter type 0 rows are supported")
    rgb = raw[:, 1:].copy().view(">u2").reshape(height, width, 3).astype(np.uint16)
    return rgb, cicp


def pad_to_multiple(x: torch.Tensor, mult: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Replicate-pad a (3, H, W) frame so both sides divide ``mult``; return the original size."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % mult, (-w) % mult
    if ph or pw:
        x = torch.nn.functional.pad(x.unsqueeze(0), (0, pw, 0, ph), mode="replicate").squeeze(0)
    return x, (h, w)


def pq_to_sdr_preview(
    hdr_pq: torch.Tensor,
    white_nits: float = 203.0,
    peak_nits: float = 1000.0,
    method: str = "reinhard",
) -> torch.Tensor:
    """(B, H, W, 3) PQ/BT.2020 signal -> (B, H, W, 3) BT.709 display signal for an SDR monitor.

    Decodes PQ to absolute luminance, converts BT.2020 to BT.709 primaries, scales so
    that ``white_nits`` lands on SDR white, compresses what lies above it (extended
    Reinhard on luminance, or a plain clip), and re-encodes with the BT.1886 inverse
    EOTF. This is a viewing aid only: everything above SDR white is squeezed or lost.
    """
    from lumaflux.color import M_2020_TO_709, apply_matrix, bt1886_inverse_eotf, pq_eotf_nits

    x = hdr_pq.movedim(-1, 1).float()
    lin = apply_matrix(pq_eotf_nits(x), M_2020_TO_709).clamp(min=0.0) / float(white_nits)
    if method == "reinhard":
        lw = max(float(peak_nits) / float(white_nits), 1.0)
        y = 0.2126 * lin[:, 0:1] + 0.7152 * lin[:, 1:2] + 0.0722 * lin[:, 2:3]
        y_tm = y * (1.0 + y / (lw * lw)) / (1.0 + y)
        lin = lin * (y_tm / y.clamp(min=1e-6))
    elif method != "clip":
        raise ValueError(f"unknown preview method: {method}")
    return bt1886_inverse_eotf(lin.clamp(0.0, 1.0)).movedim(1, -1).contiguous()
