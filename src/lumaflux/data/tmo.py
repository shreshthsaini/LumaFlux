"""Display-referred HDR to SDR tone-mapping operators.

Every operator maps PQ-coded BT.2020 RGB, mastered at ``peak_nits``, to a
full-range BT.709 SDR signal. PQ is decoded to absolute cd/m2 first. Tone and
gamut operations run in linear display light, and the final 100 cd/m2 BT.709
RGB is encoded with the inverse BT.1886 EOTF.

The common degradation chain is::

    x_sdr = codec(BT1886^-1(clip(M_2020_to_709(TMO(PQ_EOTF(x_pq))))))

The selected gamut operation is the simple conversion in Report ITU-R
BT.2407-0 Sections 2.1 through 2.4: a linear BT.2020 to BT.709 matrix followed
by hard clipping. The clip is the only non-invertible color-volume step.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

from ..color.gamut import (
    M_2020_TO_XYZ,
    M_XYZ_TO_2020,
    apply_matrix,
    gamut_compress_2020_to_709,
)
from ..color.spaces import luma_2020
from ..color.transfer import bt1886_inverse_eotf, pq_eotf_nits

SDR_PEAK_NITS = 100.0
TONE_CHAIN_VERSION = "pq1000-bt1886-bt2407-matrix-clip-decode-matrix"

# Report ITU-R BT.2446-1 Section 6.1.4, Equations 5, 6, and 10.
BT2446C_K1 = 0.83802
BT2446C_K2 = 15.09968
BT2446C_K3 = 0.74204
BT2446C_K4 = 78.99439
BT2446C_INFLECTION_NITS = 58.5 / BT2446C_K1
DEFAULT_ALPHA = 0.10


def _validate_peak(peak_nits: float) -> None:
    if not 0.0 < peak_nits <= 10_000.0:
        raise ValueError("peak_nits must be in (0, 10000]")


def _scale_by_luma(
    x: torch.Tensor,
    l_in: torch.Tensor,
    l_out: torch.Tensor,
) -> torch.Tensor:
    """Preserve chromaticity while replacing luminance."""
    scaled = x * (l_out / l_in.clamp(min=1e-8))
    return torch.where(l_in > 0.0, scaled, torch.zeros_like(scaled))


def _finish(linear2020_sdr: torch.Tensor) -> torch.Tensor:
    """Map linear BT.2020 to BT.709, hard clip, then inverse-BT.1886 encode.

    Report ITU-R BT.2446-1 Section 6.1.7 specifies inverse BT.1886 for Method
    C output. Its Section 6.1 introduction permits a BT.2407 gamut conversion
    after HDR to SDR conversion.
    """
    lin709 = gamut_compress_2020_to_709(linear2020_sdr)
    return bt1886_inverse_eotf(lin709)


def method_c_crosstalk_matrix(
    alpha: float,
    *,
    like: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """BT.2446-1 Section 6.1.2 Equation 2 and Section 6.1.6 Equation 14."""
    if not 0.0 <= alpha < 1.0 / 3.0:
        raise ValueError("alpha must be in [0, 1/3)")
    matrix = torch.tensor(
        [
            [1.0 - 2.0 * alpha, alpha, alpha],
            [alpha, 1.0 - 2.0 * alpha, alpha],
            [alpha, alpha, 1.0 - 2.0 * alpha],
        ],
        dtype=like.dtype,
        device=like.device,
    )
    return torch.linalg.inv(matrix) if inverse else matrix


def _bt2446c_reference_curve(hdr_nits: torch.Tensor) -> torch.Tensor:
    """Unscaled BT.2446-1 Section 6.1.4 Equation 5 curve in cd/m2."""
    linear = BT2446C_K1 * hdr_nits
    log_argument = (hdr_nits / BT2446C_INFLECTION_NITS - BT2446C_K3).clamp(min=1e-12)
    highlight = BT2446C_K2 * torch.log(log_argument) + BT2446C_K4
    return torch.where(hdr_nits < BT2446C_INFLECTION_NITS, linear, highlight)


def method_c_output_scale(peak_nits: float, *, like: torch.Tensor) -> torch.Tensor:
    """Return the fixed post-curve scale that maps ``peak_nits`` to SDR white.

    Report ITU-R BT.2446-1 Section 6.1.4, Equation 5 defines the reference
    Method C curve and notes that production intent determines its parameters.
    LumaFlux evaluates that reference curve unchanged, then applies the
    project-fixed normalization ``100 / f(peak_nits)``. This preserves its
    shape while anchoring the configured corpus maximum to 100 cd/m2. The
    normalization is explicit and content-independent, not an ITU parameter
    set or an input-statistics estimate.
    """
    _validate_peak(peak_nits)
    peak = like.new_tensor(peak_nits)
    return like.new_tensor(SDR_PEAK_NITS) / _bt2446c_reference_curve(peak)


def bt2446c_forward_luminance(
    hdr_nits: torch.Tensor,
    peak_nits: float = 1000.0,
) -> torch.Tensor:
    """Peak-anchored Method C luminance curve from HDR nits to SDR nits."""
    scale = method_c_output_scale(peak_nits, like=hdr_nits)
    return _bt2446c_reference_curve(hdr_nits.clamp(0.0, peak_nits)) * scale


def reinhard_forward_luminance(
    hdr_nits: torch.Tensor,
    peak_nits: float = 1000.0,
) -> torch.Tensor:
    """White-point Reinhard curve from HDR nits to 100 cd/m2 SDR nits."""
    _validate_peak(peak_nits)
    x = hdr_nits.clamp(0.0, peak_nits) / SDR_PEAK_NITS
    white = peak_nits / SDR_PEAK_NITS
    return SDR_PEAK_NITS * x * (1.0 + x / white**2) / (1.0 + x)


def tmo_reinhard(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """Reinhard et al. white-point photographic operator on absolute luminance."""
    nits = pq_eotf_nits(x_pq).clamp(0.0, peak_nits)
    y_in = luma_2020(nits)
    y_out = reinhard_forward_luminance(y_in, peak_nits)
    return _finish(_scale_by_luma(nits, y_in, y_out) / SDR_PEAK_NITS)


def tmo_bt2446a(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """ITU-R BT.2446 Method A simplified luminance form."""
    _validate_peak(peak_nits)
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    y = luma_2020(nits).clamp(min=0.0)
    # Report ITU-R BT.2446-1 Section 4.1 defines the gamma-domain luminance,
    # rho transform, piecewise tone curve, and inverse rho transform below.
    yp = (y / peak_nits).clamp(min=1e-6).pow(1.0 / 2.4)
    # Report ITU-R BT.2446-1 Section 4.1 uses 10,000 cd/m2 only as the PQ
    # reference in the Method A rho transform, not as the content peak.
    rho = 1.0 + 32.0 * (peak_nits / 10_000.0) ** (1.0 / 2.4)
    yc = torch.log1p((rho - 1.0) * yp) / torch.log(
        torch.tensor(rho, dtype=y.dtype, device=y.device)
    )
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
    yt = (
        torch.tensor(rho_sdr, dtype=y.dtype, device=y.device).pow(y_sdr) - 1.0
    ) / (rho_sdr - 1.0)
    yout_nits = yt.clamp(min=0.0).pow(2.4) * SDR_PEAK_NITS
    return _finish(_scale_by_luma(nits, y, yout_nits) / SDR_PEAK_NITS)


def tmo_bt2446c_gm(
    x_pq: torch.Tensor,
    peak_nits: float = 1000.0,
    alpha: float = DEFAULT_ALPHA,
) -> torch.Tensor:
    """Peak-anchored BT.2446-1 Method C followed by BT.2407 matrix and clip."""
    _validate_peak(peak_nits)
    nits = pq_eotf_nits(x_pq).clamp(0.0, peak_nits)

    # Report ITU-R BT.2446-1 Sections 6.1.2 and 6.1.3, Equations 2 through 4.
    crosstalk = method_c_crosstalk_matrix(alpha, like=nits)
    rgb_x_hdr = apply_matrix(nits, crosstalk)
    xyz_hdr = apply_matrix(rgb_x_hdr, M_2020_TO_XYZ)
    y_hdr = xyz_hdr[..., 1:2, :, :]

    # Report ITU-R BT.2446-1 Section 6.1.4, Equations 5 through 10.
    y_sdr = bt2446c_forward_luminance(y_hdr, peak_nits)
    xyz_sdr = _scale_by_luma(xyz_hdr, y_hdr, y_sdr)

    # Report ITU-R BT.2446-1 Sections 6.1.5 and 6.1.6, Equations 11 through 14.
    rgb_x_sdr = apply_matrix(xyz_sdr, M_XYZ_TO_2020)
    inverse_crosstalk = method_c_crosstalk_matrix(alpha, like=nits, inverse=True)
    rgb_sdr_nits = apply_matrix(rgb_x_sdr, inverse_crosstalk)
    return _finish(rgb_sdr_nits / SDR_PEAK_NITS)


def tmo_hard_clip_gm(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """Hard clip absolute display light at 100 cd/m2, then gamut map."""
    _validate_peak(peak_nits)
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    return _finish(nits / SDR_PEAK_NITS)


def tmo_bt2390_eetf_gm(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """BT.2390 EETF roll-off in PQ code space, retargeted to 100 cd/m2."""
    from ..color.transfer import pq_oetf_nits

    _validate_peak(peak_nits)
    # Report ITU-R BT.2390-12 Section 5.4 defines mastering-display PQ
    # normalization, KS = 1.5 * maxLum - 0.5, and the Hermite EETF below.
    e = x_pq.clamp(0.0, pq_oetf_nits(x_pq.new_tensor(peak_nits)))
    src_max = pq_oetf_nits(x_pq.new_tensor(peak_nits))
    dst_max = pq_oetf_nits(x_pq.new_tensor(SDR_PEAK_NITS))
    e1 = e / src_max
    max_lum = dst_max / src_max
    ks = 1.5 * max_lum - 0.5
    t = (e1 - ks) / (1.0 - ks + 1e-9)
    p = (
        (2.0 * t**3 - 3.0 * t**2 + 1.0) * ks
        + (t**3 - 2.0 * t**2 + t) * (1.0 - ks)
        + (-2.0 * t**3 + 3.0 * t**2) * max_lum
    )
    e2 = torch.where(e1 < ks, e1, p)
    nits = pq_eotf_nits((e2 * src_max).clamp(0.0, dst_max))
    return _finish(nits / SDR_PEAK_NITS)


def tmo_ocio_filmic(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """ACES-like rational filmic approximation after absolute PQ decoding."""
    _validate_peak(peak_nits)
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    x = nits / (0.18 * peak_nits / 4.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    sdr = (x * (a * x + b)) / (x * (c * x + d) + e)
    return _finish(sdr.clamp(0.0, 1.0))


def tmo_youtube_logc(x_pq: torch.Tensor, peak_nits: float = 1000.0) -> torch.Tensor:
    """LogC-style global curve after absolute PQ decoding."""
    _validate_peak(peak_nits)
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    x = nits / SDR_PEAK_NITS
    cut, slope, off = 0.011, 5.37, 0.093
    a, b, c, d = 5.555556, 0.052272, 0.247190, 0.385537
    lo = slope * x + off
    hi = c * torch.log10((a * x + b).clamp(min=1e-6)) + d
    return _finish(torch.where(x < cut, lo, hi).clamp(0.0, 1.0))


def tmo_gamma_clip(
    x_pq: torch.Tensor,
    peak_nits: float = 1000.0,
    gamma: float = 1.4,
) -> torch.Tensor:
    """Normalize by the mastering peak, apply a power curve, then hard clip."""
    _validate_peak(peak_nits)
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    nits = pq_eotf_nits(x_pq).clamp(max=peak_nits)
    sdr = (nits / peak_nits).clamp(min=0.0).pow(1.0 / gamma)
    return _finish(sdr)


TMO_REGISTRY: Dict[str, Callable[..., torch.Tensor]] = {
    "ocio_filmic": tmo_ocio_filmic,
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
