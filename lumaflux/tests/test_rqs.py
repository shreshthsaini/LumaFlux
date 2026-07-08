import torch

from lumaflux.models.rqs import RQSToneFieldDecoder, rqs_apply, spline_smoothness_loss


def _random_params(b=3, k=8, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    widths = torch.randn(b, k, generator=g) * scale
    heights = torch.randn(b, k, generator=g) * scale
    derivs = torch.randn(b, k + 1, generator=g) * scale
    return widths, heights, derivs


def test_rqs_endpoints():
    w, h, d = _random_params()
    y = torch.tensor([[0.0, 1.0]]).expand(3, 2)
    out = rqs_apply(y, w, h, d)
    assert torch.allclose(out[:, 0], torch.zeros(3), atol=1e-5)
    assert torch.allclose(out[:, 1], torch.ones(3), atol=1e-5)


def test_rqs_strictly_monotone():
    w, h, d = _random_params(scale=2.0, seed=1)
    y = torch.linspace(0, 1, 513).unsqueeze(0).expand(3, -1)
    out = rqs_apply(y, w, h, d)
    assert (out[:, 1:] > out[:, :-1] - 1e-7).all()


def test_rqs_range():
    w, h, d = _random_params(scale=3.0, seed=2)
    y = torch.rand(3, 1, 16, 16)
    out = rqs_apply(y, w, h, d)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.shape == y.shape


def test_rqs_differentiable():
    w, h, d = _random_params()
    w.requires_grad_(True)
    y = torch.rand(3, 64)
    out = rqs_apply(y, w, h, d)
    out.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


def test_decoder_identity_at_init():
    dec = RQSToneFieldDecoder(latent_channels=4, num_knots=8)
    x = torch.rand(2, 3, 16, 16)
    z = torch.randn(2, 4, 4, 4)
    out = dec(x, z)
    # Zero-init head -> uniform knots + unit derivatives -> identity spline;
    # identity-init chroma conv -> output equals input.
    assert torch.allclose(out["hdr"], x, atol=1e-4)


def test_decoder_strength_zero_keeps_luma():
    dec = RQSToneFieldDecoder(latent_channels=4, num_knots=8, strength=0.0)
    torch.nn.init.normal_(dec.head[-1].weight, std=0.5)
    x = torch.rand(2, 3, 16, 16)
    z = torch.randn(2, 4, 4, 4)
    out = dec(x, z)["hdr"]
    from lumaflux.color.spaces import rgb_to_ycbcr2020

    y_in = rgb_to_ycbcr2020(x)[:, :1]
    y_out = rgb_to_ycbcr2020(out)[:, :1]
    assert torch.allclose(y_in, y_out, atol=1e-4)


def test_smoothness_loss_zero_for_identity():
    b, k = 2, 8
    loss = spline_smoothness_loss(torch.zeros(b, k), torch.zeros(b, k), torch.zeros(b, k + 1))
    assert loss.item() < 1e-10
