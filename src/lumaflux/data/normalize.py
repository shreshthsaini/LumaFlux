"""HDR source normalization to PQ/BT.2020 with fixed mastering peak (Eq. 20).

    x_pq = PQ_OETF( M_{p_i -> 2020}( Gamma_i^{-1}(x) ) )

Linear luminance is clipped to 10^4 cd/m^2 and sources are conformed to a
single mastering peak (default 1,000 nits) so supervision is physically
consistent across heterogeneous datasets (HIDROVQA / CHUG / LIVE-TMHDR).
"""

from __future__ import annotations

import torch

from ..color.gamut import rgb709_to_rgb2020, rgbP3_to_rgb2020
from ..color.transfer import (
    PQ_PEAK_NITS,
    bt1886_eotf,
    hlg_inverse_oetf,
    pq_eotf_nits,
    pq_oetf_nits,
)

SUPPORTED_FORMATS = ("pq2020", "hlg2020", "pq_p3d65", "sdr709")


def normalize_to_pq2020(
    x: torch.Tensor,
    source_format: str = "pq2020",
    peak_nits: float = 1000.0,
    hlg_system_gamma: float = 1.2,
) -> torch.Tensor:
    """Conform an HDR frame to PQ-encoded BT.2020 at ``peak_nits`` mastering.

    Args:
        x: ``(..., 3, H, W)`` signal in [0,1] in ``source_format``.
        source_format: one of ``pq2020 | hlg2020 | pq_p3d65 | sdr709``.
        peak_nits: mastering peak the output is clipped to.
    """
    if source_format == "pq2020":
        nits = pq_eotf_nits(x)
    elif source_format == "pq_p3d65":
        nits = rgbP3_to_rgb2020(pq_eotf_nits(x)).clamp(min=0.0)
    elif source_format == "hlg2020":
        scene = hlg_inverse_oetf(x)
        # HLG OOTF: display light = scene light ^ gamma, 1000-nit nominal peak.
        nits = scene.clamp(min=0.0).pow(hlg_system_gamma) * 1000.0
    elif source_format == "sdr709":
        nits = rgb709_to_rgb2020(bt1886_eotf(x)).clamp(min=0.0) * 100.0
    else:
        raise ValueError(f"source_format must be one of {SUPPORTED_FORMATS}")
    nits = nits.clamp(min=0.0, max=min(peak_nits, PQ_PEAK_NITS))
    return pq_oetf_nits(nits)
