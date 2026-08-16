import torch

from lumaflux.color import (
    M_2020_TO_709,
    M_709_TO_2020,
    bt709_eotf,
    bt709_oetf,
    hlg_inverse_oetf,
    hlg_oetf,
    pq_eotf,
    pq_eotf_nits,
    pq_oetf,
    pq_oetf_nits,
    pu21_encode,
    pu21_decode,
    rgb2020_to_rgb709,
    rgb709_to_rgb2020,
    rgb_to_ycbcr2020,
    ycbcr2020_to_rgb,
    pq2020_to_ictcp,
)
from lumaflux.color.spaces import PU21_PEAK


def test_pq_roundtrip():
    # Note: PQ OETF(0) is ~7.3e-7 (not exactly 0) per the ST-2084 formula,
    # and gradient-safety clamps flatten the curves below signal ~1e-3
    # (linear ~1e-10), so the round-trip is tested away from the origin.
    x = torch.linspace(1e-3, 1, 1001, dtype=torch.float64)
    assert torch.allclose(pq_eotf(pq_oetf(x)), x, atol=1e-9)
    assert torch.allclose(pq_oetf(pq_eotf(x)), x, atol=1e-9)
    x32 = x.float()
    assert torch.allclose(pq_eotf(pq_oetf(x32)), x32, atol=2e-4)


def test_pq_known_values():
    # ST 2084 anchors: 100 nits -> ~0.508, 1000 nits -> ~0.7518.
    assert abs(pq_oetf_nits(torch.tensor(100.0)).item() - 0.5081) < 1e-3
    assert abs(pq_oetf_nits(torch.tensor(1000.0)).item() - 0.7518) < 1e-3
    assert abs(pq_eotf_nits(torch.tensor(1.0)).item() - 10000.0) < 1e-2


def test_bt709_roundtrip():
    x = torch.linspace(0, 1, 1001)
    assert torch.allclose(bt709_eotf(bt709_oetf(x)), x, atol=1e-4)


def test_hlg_roundtrip():
    x = torch.linspace(0, 1, 1001)
    assert torch.allclose(hlg_inverse_oetf(hlg_oetf(x)), x, atol=1e-4)


def test_gamut_matrices_inverse():
    eye = M_2020_TO_709 @ M_709_TO_2020
    assert torch.allclose(eye, torch.eye(3), atol=2e-3)


def test_gamut_roundtrip_in_gamut():
    x = torch.rand(2, 3, 8, 8) * 0.5 + 0.25  # well inside both gamuts
    y = rgb2020_to_rgb709(rgb709_to_rgb2020(x))
    assert torch.allclose(x, y, atol=5e-3)


def test_white_preserved():
    white = torch.ones(1, 3, 1, 1)
    assert torch.allclose(rgb709_to_rgb2020(white), white, atol=2e-3)


def test_ycbcr_roundtrip():
    x = torch.rand(2, 3, 8, 8)
    y = ycbcr2020_to_rgb(rgb_to_ycbcr2020(x))
    assert torch.allclose(x, y, atol=1e-5)


def test_ictcp_neutral_axis():
    # Achromatic input -> Ct = Cp = 0.
    gray = torch.full((1, 3, 4, 4), 0.5)
    ictcp = pq2020_to_ictcp(gray)
    assert ictcp[:, 1:].abs().max() < 1e-4


def test_pu21_monotone_and_anchor():
    nits = torch.logspace(-2, 4, 200)
    pu = pu21_encode(nits)
    assert (pu[1:] >= pu[:-1] - 1e-6).all()
    # 100 cd/m^2 encodes near 256 by construction of PU21.
    assert abs(pu21_encode(torch.tensor(100.0)).item() - 256.0) < 0.5


def test_pu21_peak_matches_fixed_encoder():
    encoded_peak = pu21_encode(torch.tensor(10_000.0, dtype=torch.float64)).item()
    assert PU21_PEAK == encoded_peak
    assert abs(PU21_PEAK - 595.3939195) < 1e-6


def test_pu21_roundtrip():
    nits = torch.logspace(-2, 4, 200, dtype=torch.float64)
    assert torch.allclose(pu21_decode(pu21_encode(nits)), nits, rtol=1e-8, atol=1e-8)
