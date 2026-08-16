"""Analytic inverses for LumaFlux synthetic SDR operators.

The Method C path follows the inverse direction permitted by Report ITU-R
BT.2446-1 Section 6.2. SDR BT.709 is decoded with BT.1886, the selected
BT.2407 matrix is inverted, the Method C operations from Sections 6.1.2
through 6.1.6 are reversed, and absolute BT.2020 light is PQ encoded.

``hdr_peak_nits`` is always explicit and defaults to the fixed corpus value of
1,000 cd/m2. No image-statistics peak estimation is implemented.
"""

from __future__ import annotations

import torch

from ..color.gamut import (
    M_2020_TO_XYZ,
    M_XYZ_TO_2020,
    apply_matrix,
    gamut_expand_709_to_2020,
)
from ..color.spaces import luma_2020
from ..color.transfer import bt1886_eotf, pq_oetf_nits
from ..data.tmo import (
    BT2446C_INFLECTION_NITS,
    BT2446C_K1,
    BT2446C_K2,
    BT2446C_K3,
    BT2446C_K4,
    DEFAULT_ALPHA,
    SDR_PEAK_NITS,
    method_c_crosstalk_matrix,
    method_c_output_scale,
)

DEFAULT_HDR_PEAK_NITS = 1000.0


def _validate_input(sdr_bt709: torch.Tensor, hdr_peak_nits: float) -> None:
    if sdr_bt709.shape[-3] != 3:
        raise ValueError(f"expected three RGB channels, got shape {tuple(sdr_bt709.shape)}")
    if not 0.0 < hdr_peak_nits <= 10_000.0:
        raise ValueError("hdr_peak_nits must be in (0, 10000]")


def _scale_by_luma(
    x: torch.Tensor,
    l_in: torch.Tensor,
    l_out: torch.Tensor,
) -> torch.Tensor:
    scaled = x * (l_out / l_in.clamp(min=1e-8))
    return torch.where(l_in > 0.0, scaled, torch.zeros_like(scaled))


def bt2446c_inverse_luminance(
    sdr_nits: torch.Tensor,
    hdr_peak_nits: float = DEFAULT_HDR_PEAK_NITS,
) -> torch.Tensor:
    """Invert the peak-anchored Method C curve from SDR nits to HDR nits.

    This is the algebraic inverse of Report ITU-R BT.2446-1 Section 6.1.4,
    Equation 5, including the fixed post-curve normalization used by our
    forward operator to map ``hdr_peak_nits`` to SDR white.
    """
    scale = method_c_output_scale(hdr_peak_nits, like=sdr_nits)
    reference_sdr = sdr_nits.clamp(min=0.0) / scale
    linear = reference_sdr / BT2446C_K1
    highlight = BT2446C_INFLECTION_NITS * (
        torch.exp((reference_sdr - BT2446C_K4) / BT2446C_K2) + BT2446C_K3
    )
    hdr = torch.where(reference_sdr < 58.5, linear, highlight)
    return hdr.clamp(0.0, hdr_peak_nits)


def bt2446c_inverse(
    sdr_bt709: torch.Tensor,
    *,
    alpha: float = DEFAULT_ALPHA,
    hdr_peak_nits: float = DEFAULT_HDR_PEAK_NITS,
) -> torch.Tensor:
    """Invert our BT.2446c+GM forward operator to PQ-coded BT.2020 RGB."""
    _validate_input(sdr_bt709, hdr_peak_nits)

    # Reverse Report ITU-R BT.2446-1 Section 6.1.7 and the subsequent
    # BT.2407 Section 2 matrix conversion. A prior hard clip cannot be undone.
    sdr_709_nits = bt1886_eotf(sdr_bt709.clamp(0.0, 1.0)) * SDR_PEAK_NITS
    sdr_2020_nits = gamut_expand_709_to_2020(sdr_709_nits)

    # Reverse BT.2446-1 Sections 6.1.6 through 6.1.3, Equations 14 through 3.
    crosstalk = method_c_crosstalk_matrix(alpha, like=sdr_2020_nits)
    rgb_x_sdr = apply_matrix(sdr_2020_nits, crosstalk)
    xyz_sdr = apply_matrix(rgb_x_sdr, M_2020_TO_XYZ)
    y_sdr = xyz_sdr[..., 1:2, :, :]

    # BT.2446-1 Section 6.2 permits the inverse of the Section 6.1 conversion.
    y_hdr = bt2446c_inverse_luminance(y_sdr, hdr_peak_nits)
    xyz_hdr = _scale_by_luma(xyz_sdr, y_sdr, y_hdr)

    rgb_x_hdr = apply_matrix(xyz_hdr, M_XYZ_TO_2020)
    inverse_crosstalk = method_c_crosstalk_matrix(alpha, like=rgb_x_hdr, inverse=True)
    rgb_hdr_nits = apply_matrix(rgb_x_hdr, inverse_crosstalk)
    return pq_oetf_nits(rgb_hdr_nits.clamp(0.0, hdr_peak_nits))


def reinhard_inverse_luminance(
    sdr_nits: torch.Tensor,
    hdr_peak_nits: float = DEFAULT_HDR_PEAK_NITS,
) -> torch.Tensor:
    """Invert the white-point Reinhard curve used by ``tmo_reinhard``."""
    if not 0.0 < hdr_peak_nits <= 10_000.0:
        raise ValueError("hdr_peak_nits must be in (0, 10000]")
    output = sdr_nits.clamp(0.0, SDR_PEAK_NITS) / SDR_PEAK_NITS
    white = hdr_peak_nits / SDR_PEAK_NITS
    a = 1.0 / white**2
    b = 1.0 - output
    x = (-b + torch.sqrt(b**2 + 4.0 * a * output)) / (2.0 * a)
    return (x * SDR_PEAK_NITS).clamp(0.0, hdr_peak_nits)


def reinhard_inverse(
    sdr_bt709: torch.Tensor,
    *,
    hdr_peak_nits: float = DEFAULT_HDR_PEAK_NITS,
) -> torch.Tensor:
    """Invert our Reinhard+GM forward operator to PQ-coded BT.2020 RGB."""
    _validate_input(sdr_bt709, hdr_peak_nits)
    sdr_709_nits = bt1886_eotf(sdr_bt709.clamp(0.0, 1.0)) * SDR_PEAK_NITS
    sdr_2020_nits = gamut_expand_709_to_2020(sdr_709_nits)
    y_sdr = luma_2020(sdr_2020_nits)
    y_hdr = reinhard_inverse_luminance(y_sdr, hdr_peak_nits)
    hdr_2020_nits = _scale_by_luma(sdr_2020_nits, y_sdr, y_hdr)
    return pq_oetf_nits(hdr_2020_nits.clamp(0.0, hdr_peak_nits))
