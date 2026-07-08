"""Color gamut (primaries) conversion matrices and helpers.

Matrices map *linear-light* RGB column vectors between primary sets.
Tensors are channel-first ``(..., 3, H, W)`` or channel-last ``(..., 3)``;
:func:`apply_matrix` handles both via the ``channel_dim`` argument.

Derived from ITU-R BT.2087 / BT.2407 and SMPTE RP 177 using D65 white.
"""

from __future__ import annotations

import torch

# Linear BT.709 -> linear BT.2020 (ITU-R BT.2087 Table 2).
M_709_TO_2020 = torch.tensor(
    [
        [0.6274, 0.3293, 0.0433],
        [0.0691, 0.9195, 0.0114],
        [0.0164, 0.0880, 0.8956],
    ]
)

# Linear BT.2020 -> linear BT.709 (inverse of the above, BT.2407 Annex 1).
M_2020_TO_709 = torch.tensor(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ]
)

# Linear DCI-P3 (D65) -> linear BT.2020.
M_P3D65_TO_2020 = torch.tensor(
    [
        [0.7530, 0.1986, 0.0484],
        [0.0457, 0.9418, 0.0125],
        [-0.0012, 0.0176, 0.9836],
    ]
)

# Linear RGB (BT.2020) -> CIE 1931 XYZ.
M_2020_TO_XYZ = torch.tensor(
    [
        [0.6370, 0.1446, 0.1689],
        [0.2627, 0.6780, 0.0593],
        [0.0000, 0.0281, 1.0610],
    ]
)


def apply_matrix(x: torch.Tensor, m: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """Apply a 3x3 color matrix along ``channel_dim`` (size 3)."""
    m = m.to(dtype=x.dtype, device=x.device)
    x = x.movedim(channel_dim, -1)
    y = x @ m.T
    return y.movedim(-1, channel_dim)


def rgb709_to_rgb2020(x: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    return apply_matrix(x, M_709_TO_2020, channel_dim)


def rgb2020_to_rgb709(x: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    return apply_matrix(x, M_2020_TO_709, channel_dim)


def rgbP3_to_rgb2020(x: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    return apply_matrix(x, M_P3D65_TO_2020, channel_dim)


def gamut_compress_2020_to_709(x: torch.Tensor, channel_dim: int = -3) -> torch.Tensor:
    """BT.2020 -> BT.709 with simple hard clipping of out-of-gamut values.

    This models the lossy ``M_{2020->709}`` stage of the SDR formation chain
    (Eq. 1/19 of the paper): wide-gamut colors outside BT.709 are clipped,
    which is what discards chroma information that ITM must hallucinate back.
    """
    return rgb2020_to_rgb709(x, channel_dim).clamp(0.0, 1.0)
