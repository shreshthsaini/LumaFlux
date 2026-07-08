"""Tone-mapping operators (TMOs) for synthesizing SDR from HDR (Sec. 5).

Every operator maps a *PQ-encoded BT.2020* HDR frame (``(..., 3, H, W)`` in
[0,1], mastered at ``peak_nits``) to an SDR BT.709 signal in [0,1]
(BT.709 OETF-encoded, before quantization/codec degradation). The composite
degradation chain of Eq. 19 is::

    x_sdr = Q_codec( M_2020->709( TMO(x_pq) ) )

The operator family follows HDRTV4K (Guo et al., 2023) as referenced by the
paper: OCIOv2-style filmic, BT.2446c+GM, hard-clip+GM, BT.2446a, Reinhard,
YouTube-LogC-style, BT.2390-EETF+GM, plus a plain gamma/clip fallback.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from ..color.gamut import gamut_compress_2020_to_709
from ..color.spaces import luma_2020
from ..color.transfer import bt709_oetf, pq_eotf_nits

SDR_PEAK_NITS = 100.0


def _scale_by_luma(x_nits: torch.Tensor, l_in: torch.Tensor, l_out: torch.Tensor) -> torch.Tensor:
    """Apply a luminance-domain tone curve to RGB via ratio scaling."""
    ratio = l_out / l_in.clamp(min=1e-6)
    return x_nits * ratio


def _finish(linear2020_sdr: torch.Tensor) -> torch.Tensor:
    """Gamut-map linear BT.2020 SDR (normalized to [0,1]) and OETF-encode."""
    lin709 = gamut_compress_2020_to_709(linear2020_sdr.clamp(0.0, 1.0))
    return bt709_oetf(lin709)


def tmo_reinhard(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """Photographic tone reproduction (Reinhard et al.) on luminance."""
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    y = luma_2020(nits)
    yn = y / SDR_PEAK_NITS
    ywhite = peak_nits / SDR_PEAK_NITS
    yout = yn * (1.0 + yn / ywhite**2) / (1.0 + yn)
    sdr = _scale_by_luma(nits / SDR_PEAK_NITS, yn, yout)
    return _finish(sdr)


def tmo_bt2446a(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """ITU-R BT.2446 Method A (simplified luminance form)."""
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    y = luma_2020(nits).clamp(min=0.0)
    # Normalized, gamma-domain luma as specified by Method A.
    yp = (y / peak_nits).clamp(min=1e-6).pow(1.0 / 2.4)
    rho = 1.0 + 32.0 * (peak_nits / 10_000.0) ** (1.0 / 2.4)
    yc = torch.log1p((rho - 1.0) * yp) / torch.log(torch.tensor(rho, dtype=y.dtype, device=y.device))
    # Piecewise tone curve on y_c.
    y_sdr = torch.where(
        yc <= 0.7399,
        1.0770 * yc,
        torch.where(
            yc < 0.9909,
            -1.1510 * yc.pow(2) + 2.7811 * yc - 0.6302,
            0.5 * yc + 0.5,
        ),
    )
    rho_sdr = 1.0 + 32.0 * (SDR_PEAK_NITS / 10_000.0) ** (1.0 / 2.4)
    yt = (torch.tensor(rho_sdr, dtype=y.dtype, device=y.device).pow(y_sdr) - 1.0) / (rho_sdr - 1.0)
    yout_nits = yt.clamp(min=0.0).pow(2.4) * SDR_PEAK_NITS
    sdr = _scale_by_luma(nits / SDR_PEAK_NITS, y / SDR_PEAK_NITS, yout_nits / SDR_PEAK_NITS)
    return _finish(sdr)


def tmo_bt2446c_gm(x_pq: torch.Tensor, peak_nits: float = 1000.0, alpha: float = 0.10) -> torch.Tensor:
    """ITU-R BT.2446 Method C (crosstalk + knee) followed by gamut mapping."""
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    x = nits / peak_nits
    # Crosstalk de-saturates highlights before the knee.
    r, g, b = x.unbind(dim=-3)
    cs = 1.0 - 2.0 * alpha
    rc = cs * r + alpha * g + alpha * b
    gc = alpha * r + cs * g + alpha * b
    bc = alpha * r + alpha * g + cs * b
    xc = torch.stack([rc, gc, bc], dim=-3)
    # Knee in a log-like domain: linear below k1, soft-compressed above.
    k1, k3 = 0.75, 0.75
    inv_k4 = (peak_nits / SDR_PEAK_NITS) * k1 / k3 - k1
    y = xc.clamp(min=0.0) * (peak_nits / SDR_PEAK_NITS)
    yc = torch.where(y < k1, y, k1 + (1.0 - k1) * torch.tanh((y - k1) / inv_k4))
    # Inverse crosstalk.
    a2 = alpha / (1.0 - 3.0 * alpha + 1e-9)
    s2 = 1.0 + 2.0 * a2
    r2, g2, b2 = yc.unbind(dim=-3)
    ro = s2 * r2 - a2 * (g2 + b2) / (1.0 - alpha)
    go = s2 * g2 - a2 * (r2 + b2) / (1.0 - alpha)
    bo = s2 * b2 - a2 * (r2 + g2) / (1.0 - alpha)
    sdr = torch.stack([ro, go, bo], dim=-3)
    return _finish(sdr)


def tmo_hard_clip_gm(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """Hard clip at SDR peak + gamut map ("HC+GM")."""
    nits = pq_eotf_nits(x_pq)
    sdr = (nits / SDR_PEAK_NITS).clamp(0.0, 1.0)
    return _finish(sdr)


def tmo_bt2390_eetf_gm(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """BT.2390 EETF roll-off applied in PQ space, retargeted to 100 nits."""
    from ..color.transfer import pq_oetf_nits

    e = x_pq.clamp(0.0, 1.0)
    src_max = pq_oetf_nits(torch.tensor(peak_nits)).item()
    dst_max = pq_oetf_nits(torch.tensor(SDR_PEAK_NITS)).item()
    e1 = e / src_max
    max_lum = dst_max / src_max
    ks = 1.5 * max_lum - 0.5
    t = (e1 - ks) / (1.0 - ks + 1e-9)
    p = ((2.0 * t**3 - 3.0 * t**2 + 1.0) * ks
         + (t**3 - 2.0 * t**2 + t) * (1.0 - ks)
         + (-2.0 * t**3 + 3.0 * t**2) * max_lum)
    e2 = torch.where(e1 < ks, e1, p)
    nits = pq_eotf_nits((e2 * src_max).clamp(0.0, 1.0))
    sdr = (nits / SDR_PEAK_NITS).clamp(0.0, 1.0)
    return _finish(sdr)


def tmo_ocio_filmic(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """OCIOv2-style filmic (ACES-like rational sigmoid) view transform."""
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    x = nits / (0.18 * peak_nits / 4.0)  # mid-gray anchored exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    sdr = (x * (a * x + b)) / (x * (c * x + d) + e)
    return _finish(sdr.clamp(0.0, 1.0))


def tmo_youtube_logc(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """LogC-style global log curve approximating YouTube's HDR->SDR look."""
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    x = nits / SDR_PEAK_NITS
    cut, slope, off = 0.011, 5.37, 0.093
    a, b, c, d = 5.555556, 0.052272, 0.247190, 0.385537
    lo = slope * x + off
    hi = c * torch.log10((a * x + b).clamp(min=1e-6)) + d
    sdr = torch.where(x < cut, lo, hi).clamp(0.0, 1.0)
    # Curve is already display-light-like; undo the final OETF double-bend by
    # treating it as linear SDR.
    return _finish(sdr)


def tmo_gamma_clip(x_pq: torch.Tensor, peak_nits: float = 1000.0, gamma: float = 1.4) -> torch.Tensor:
    """Plain exposure+gamma clip; the crudest member of the family."""
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    sdr = (nits / peak_nits).clamp(min=0.0).pow(1.0 / gamma)
    return _finish(sdr)


TMO_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "ocio_v2": tmo_ocio_filmic,
    "bt2446c_gm": tmo_bt2446c_gm,
    "hard_clip_gm": tmo_hard_clip_gm,
    "bt2446a": tmo_bt2446a,
    "reinhard": tmo_reinhard,
    "youtube_logc": tmo_youtube_logc,
    "bt2390_eetf_gm": tmo_bt2390_eetf_gm,
    "gamma_clip": tmo_gamma_clip,
}


def apply_tmo(name: str, x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    if name not in TMO_REGISTRY:
        raise KeyError(f"Unknown TMO '{name}'. Available: {sorted(TMO_REGISTRY)}")
    return TMO_REGISTRY[name](x_pq, peak_nits=peak_nits).clamp(0.0, 1.0)
