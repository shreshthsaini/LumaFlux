import pytest
import torch

from lumaflux.color.spaces import luma_2020
from lumaflux.color.transfer import pq_oetf_nits
from lumaflux.data.tmo import TMO_REGISTRY, apply_tmo


@pytest.fixture(scope="module")
def hdr_frame():
    torch.manual_seed(0)
    nits = torch.rand(1, 3, 32, 32) * 1000.0
    return pq_oetf_nits(nits)


@pytest.mark.parametrize("name", sorted(TMO_REGISTRY))
def test_tmo_output_range(name, hdr_frame):
    sdr = apply_tmo(name, hdr_frame)
    assert sdr.shape == hdr_frame.shape
    assert torch.isfinite(sdr).all()
    assert sdr.min() >= 0.0 and sdr.max() <= 1.0


@pytest.mark.parametrize("name", sorted(TMO_REGISTRY))
def test_tmo_luminance_monotone(name):
    """A gray ramp must map to a (weakly) monotone SDR luma ramp."""
    ramp_nits = torch.logspace(0, 3, 64).view(1, 1, 64, 1).expand(1, 3, 64, 1)
    x_pq = pq_oetf_nits(ramp_nits)
    sdr = apply_tmo(name, x_pq)
    y = luma_2020(sdr).flatten()
    assert (y[1:] >= y[:-1] - 1e-3).all(), f"{name} not monotone"


def test_unknown_tmo_raises(hdr_frame):
    with pytest.raises(KeyError):
        apply_tmo("nope", hdr_frame)
