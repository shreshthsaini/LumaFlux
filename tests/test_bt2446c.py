import pytest
import torch

from lumaflux.baselines.bt2446c import (
    bt2446c_inverse,
    bt2446c_inverse_luminance,
    reinhard_inverse,
)
from lumaflux.color.gamut import M_709_TO_2020, apply_matrix
from lumaflux.color.transfer import bt1886_eotf, pq_eotf_nits, pq_oetf_nits
from lumaflux.data.degrade import codec_degrade
from lumaflux.data.tmo import (
    bt2446c_forward_luminance,
    tmo_bt2446c_gm,
    tmo_reinhard,
)
from lumaflux.evaluation.deitp import delta_e_itp
from lumaflux.evaluation.metrics import pu21_psnr


def _synthetic_hdr_gradient(size: int = 256) -> torch.Tensor:
    """A colored 1,000-nit gradient kept safely inside the BT.709 gamut."""
    x = torch.linspace(0.01, 1.0, size).view(1, 1, size).expand(1, size, size)
    y = torch.linspace(0.1, 0.9, size).view(1, size, 1).expand(1, size, size)
    linear_709 = torch.cat(
        [0.7 * x + 0.2 * y, 0.6 * x + 0.25 * y, 0.5 * x + 0.3 * y]
    ).clamp(0.0, 1.0)
    linear_2020_nits = apply_matrix(linear_709 * 1000.0, M_709_TO_2020)
    return pq_oetf_nits(linear_2020_nits.clamp(0.0, 1000.0))


@pytest.mark.parametrize(
    ("hdr_nits", "expected_sdr_nits"),
    [(0.0, 0.0), (1000.0, 100.0)],
)
def test_bt2446c_peak_anchored_luminance_roundtrip(hdr_nits, expected_sdr_nits):
    hdr = torch.tensor(hdr_nits, dtype=torch.float64)
    sdr = bt2446c_forward_luminance(hdr, peak_nits=1000.0)
    reconstructed = bt2446c_inverse_luminance(sdr, hdr_peak_nits=1000.0)
    assert sdr.item() == pytest.approx(expected_sdr_nits, abs=1e-8)
    assert reconstructed.item() == pytest.approx(hdr_nits, abs=1e-7)


def test_bt2446c_white_signal_maps_to_fixed_1000_nits():
    sdr = torch.ones((1, 3, 1, 1), dtype=torch.float64)
    output_nits = pq_eotf_nits(bt2446c_inverse(sdr))[0, :, 0, 0]
    assert output_nits.mean().item() == pytest.approx(1000.0, abs=0.1)
    assert (output_nits.max() - output_nits.min()).item() < 0.1


def test_bt2446c_luminance_curve_is_continuous_at_knee():
    hdr = torch.tensor([69.8073, 69.8074, 69.8075], dtype=torch.float64)
    sdr = bt2446c_forward_luminance(hdr)
    reconstructed = bt2446c_inverse_luminance(sdr)
    assert torch.allclose(reconstructed, hdr, atol=1e-3)
    assert (sdr[1:] >= sdr[:-1]).all()


def test_bt2446c_uses_bt1886_sdr_decode():
    signal = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    sdr_nits = bt1886_eotf(signal) * 100.0
    expected = bt2446c_inverse_luminance(sdr_nits)
    gray = signal.view(-1, 1, 1, 1).expand(-1, 3, 1, 1)
    actual = pq_eotf_nits(bt2446c_inverse(gray)).mean(dim=1).flatten()
    assert torch.allclose(actual, expected, atol=0.1)


@pytest.mark.parametrize(
    ("forward", "inverse"),
    [(tmo_bt2446c_gm, bt2446c_inverse), (tmo_reinhard, reinhard_inverse)],
)
def test_no_codec_roundtrip_exceeds_acceptance_bar(forward, inverse):
    reference = _synthetic_hdr_gradient()
    prediction = inverse(forward(reference), hdr_peak_nits=1000.0)
    assert pu21_psnr(prediction, reference).item() >= 35.0
    assert delta_e_itp(prediction, reference).item() <= 8.0


@pytest.mark.parametrize(
    ("forward", "inverse"),
    [(tmo_bt2446c_gm, bt2446c_inverse), (tmo_reinhard, reinhard_inverse)],
)
def test_crf23_roundtrip_has_documented_codec_floor(forward, inverse):
    """Assert the protocol floor after 8-bit 4:2:0 CRF23 degradation.

    Measured on this fixture: BT.2446c is 38.19 dB and 7.00 DeltaE-ITP;
    Reinhard is 41.42 dB and 4.92 DeltaE-ITP. Codec implementations can vary,
    so the portable acceptance threshold is the requested 28 dB floor.
    """
    reference = _synthetic_hdr_gradient()
    sdr = codec_degrade(forward(reference), crf=23)
    prediction = inverse(sdr, hdr_peak_nits=1000.0)
    assert pu21_psnr(prediction, reference).item() >= 28.0
