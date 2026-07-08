import pytest
import torch

from lumaflux.models.factory import build_model
from lumaflux.models.luma_flux import make_img_ids, pack_latents, unpack_latents

TINY = {
    "model": {
        "backbone": "tiny",
        "siglip": "tiny",
        "rank": 4,
        "num_knots": 8,
        "phys_channels": 8,
        "stats_dim": 8,
        "num_bands": 4,
        "num_null_tokens": 4,
        "modulation_hidden": 32,
    }
}


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(TINY)


def test_pack_unpack_roundtrip():
    z = torch.randn(2, 4, 8, 8)
    packed = pack_latents(z)
    assert packed.shape == (2, 16, 16)
    assert torch.allclose(unpack_latents(packed, 8, 8), z)


def test_img_ids_shape():
    ids = make_img_ids(8, 8, "cpu", torch.float32)
    assert ids.shape == (16, 3)


def test_backbone_frozen(model):
    # Frozen: every original Flux / VAE / SigLIP weight (all adapter modules
    # live outside the ``.inner.`` / ``vae.`` / ``siglip.`` namespaces).
    for name, p in model.named_parameters():
        if any(k in name for k in (".inner.", "vae.", "siglip.")):
            assert not p.requires_grad, f"{name} should be frozen"


def test_trainable_fraction_small(model):
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert 0 < n_train < 0.5 * n_total


def test_velocity_forward_and_identity_at_init(model):
    """At init all adapters are zero-initialized residuals, so the wrapped
    transformer must reproduce the frozen backbone's output exactly."""
    torch.manual_seed(1)
    x_sdr = torch.rand(2, 3, 64, 64)
    z = model.encode_image(x_sdr)
    latent_hw = z.shape[-2:]
    z_packed = pack_latents(z)
    t = torch.full((2,), 0.7)
    cond = model.prepare_condition(x_sdr)

    v_adapted = model.velocity(z_packed, t, cond, latent_hw)
    assert v_adapted.shape == z_packed.shape
    assert torch.isfinite(v_adapted).all()

    # Same call with adapters inactive (raw frozen path).
    model.context["active"] = False
    img_ids = make_img_ids(*latent_hw, z_packed.device, z_packed.dtype)
    txt_ids = torch.zeros(cond["ctx_tokens"].shape[1], 3)
    v_frozen = model.transformer(
        hidden_states=z_packed,
        encoder_hidden_states=cond["ctx_tokens"],
        pooled_projections=cond["pooled"],
        timestep=t,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )[0]
    assert torch.allclose(v_adapted, v_frozen, atol=1e-5)


def test_gradients_reach_adapters(model):
    torch.manual_seed(2)
    x_sdr = torch.rand(1, 3, 64, 64)
    z = model.encode_image(x_sdr)
    cond = model.prepare_condition(x_sdr)
    v = model.velocity(pack_latents(z), torch.tensor([0.5]), cond, z.shape[-2:])
    v.pow(2).mean().backward()
    pga_grads = [
        p.grad for n, p in model.named_parameters()
        if p.requires_grad and "pga" in n and p.grad is not None
    ]
    assert pga_grads and any(g.abs().sum() > 0 for g in pga_grads)


def test_decode_hdr_shapes(model):
    z = torch.randn(1, 4, 8, 8)
    out = model.decode_hdr(pack_latents(z), (8, 8))
    assert out["hdr"].shape == (1, 3, 64, 64)
    assert out["hdr"].min() >= 0 and out["hdr"].max() <= 1


def test_adapter_save_load_roundtrip(model, tmp_path):
    path = tmp_path / "adapters.safetensors"
    model.save_adapters(path)
    state = model.trainable_state_dict()
    assert state, "no trainable parameters captured"
    assert not any(k.startswith(("vae.", "siglip.")) for k in state)

    model2 = build_model(TINY)
    model2.load_adapters(path)
    for k, v in model2.trainable_state_dict().items():
        assert torch.allclose(v, state[k]), k
