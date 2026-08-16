import pytest
import torch
from safetensors.torch import save_file

from lumaflux.models.coupler import HDRResidualCoupler
from lumaflux.models.factory import build_model, resolve_torch_dtype
from lumaflux.models.luma_flux import make_img_ids, pack_latents, unpack_latents
from lumaflux.models.lora import LoRALinear
from lumaflux.models.pcm import PCMModulator, PerceptualConnector

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


def test_pcm_bottleneck_shape_and_zero_identity():
    pcm = PCMModulator(model_dim=32, bottleneck=5)
    assert pcm.mlp[0].weight.shape == (5, 32)
    assert pcm.mlp[-1].weight.shape == (64, 5)
    assert torch.count_nonzero(pcm.mlp[-1].weight) == 0
    hidden = torch.randn(2, 7, 32)
    perceptual = torch.randn(2, 4, 32)
    out = pcm(hidden, perceptual, torch.ones(2), torch.zeros(2))
    assert torch.equal(out, hidden)


def test_coupler_bottlenecks_are_independent_and_zero_identity():
    coupler = HDRResidualCoupler(phys_channels=6, model_dim=32, bottleneck=5)
    assert coupler.w_p[0].weight.shape == (5, 6)
    assert coupler.w_c[0].weight.shape == (5, 32)
    assert coupler.w_p[-1] is not coupler.w_c[-1]
    assert torch.count_nonzero(coupler.w_p[-1].weight) == 0
    assert torch.count_nonzero(coupler.w_c[-1].weight) == 0
    residual = torch.randn(2, 4, 32)
    out = coupler(
        residual,
        torch.randn(2, 4, 6),
        torch.randn(2, 4, 32),
        torch.ones(2),
        (2, 2),
    )
    assert torch.equal(out, residual)


def test_perceptual_connector_uses_configured_bottleneck():
    connector = PerceptualConnector(siglip_dim=24, model_dim=32, bottleneck=5)
    assert connector.proj[0].weight.shape == (5, 24)
    assert connector.proj[-1].weight.shape == (32, 5)


def test_model_dtype_keeps_trainable_adapters_fp32():
    cfg = {"model": {**TINY["model"], "dtype": "bf16"}}
    typed_model = build_model(cfg)
    frozen_dtypes = {
        parameter.dtype
        for name, parameter in typed_model.named_parameters()
        if ".inner." in name
    }
    assert frozen_dtypes == {torch.bfloat16}
    assert {parameter.dtype for parameter in typed_model.trainable_parameters()} == {torch.float32}


def test_gradient_checkpointing_config_flag():
    cfg = {"model": {**TINY["model"], "gradient_checkpointing": True}}
    checkpointed = build_model(cfg)
    assert checkpointed.transformer.is_gradient_checkpointing


def test_dtype_name_validation():
    assert resolve_torch_dtype("bf16") is torch.bfloat16
    assert resolve_torch_dtype("fp32") is torch.float32
    with pytest.raises(ValueError, match="Unsupported model dtype"):
        resolve_torch_dtype("tf32")


def test_adapter_load_rejects_missing_keys(tmp_path):
    source = build_model(TINY)
    state = source.trainable_state_dict()
    removed = sorted(state).pop()
    state.pop(removed)
    path = tmp_path / "missing.safetensors"
    save_file(state, path)

    with pytest.raises(RuntimeError, match=f"missing keys:.*{removed}"):
        build_model(TINY).load_adapters(path)


def test_adapter_load_rejects_unexpected_keys(tmp_path):
    source = build_model(TINY)
    state = source.trainable_state_dict()
    state["not_an_adapter"] = torch.zeros(1)
    path = tmp_path / "unexpected.safetensors"
    save_file(state, path)

    with pytest.raises(RuntimeError, match="unexpected keys:.*not_an_adapter"):
        build_model(TINY).load_adapters(path)


def test_attention_lora_ablation_disables_lumaflux_blocks_and_rqs():
    cfg = {
        "model": {
            **TINY["model"],
            "enable_pga": False,
            "enable_pcm": False,
            "enable_coupler": False,
            "rqs_mode": "disabled",
            "lora_mode": "attention",
        }
    }
    adapted = build_model(cfg)
    assert adapted.block_wrappers == []
    assert adapted.rqs is None
    lora_names = [name for name, module in adapted.named_modules() if isinstance(module, LoRALinear)]
    assert lora_names
    assert all(".attn." in name for name in lora_names)


def test_full_backbone_lora_includes_mlp_projections():
    cfg = {
        "model": {
            **TINY["model"],
            "enable_pga": False,
            "enable_pcm": False,
            "enable_coupler": False,
            "lora_mode": "attention_mlp",
        }
    }
    adapted = build_model(cfg)
    lora_names = [name for name, module in adapted.named_modules() if isinstance(module, LoRALinear)]
    assert any(".attn." in name for name in lora_names)
    assert any(
        marker in name
        for name in lora_names
        for marker in (".ff.", ".ff_context.", ".proj_mlp", ".proj_out")
    )


def test_component_and_static_modulation_switches():
    cfg = {
        "model": {
            **TINY["model"],
            "enable_pga": True,
            "pga_spectral": False,
            "enable_pcm": False,
            "enable_coupler": False,
            "time_layer_modulation": False,
        }
    }
    adapted = build_model(cfg)
    assert adapted.block_wrappers
    assert all(wrapper.pga is not None for wrapper in adapted.block_wrappers)
    assert all(wrapper.pga.w_r is None for wrapper in adapted.block_wrappers)
    assert all(wrapper.pcm is None for wrapper in adapted.block_wrappers)
    assert all(wrapper.coupler is None for wrapper in adapted.block_wrappers)
    first = adapted.modulation(torch.tensor([0.1]), 0)
    second = adapted.modulation(torch.tensor([0.9]), 3)
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_linear_tone_curve_mode_and_disabled_decode():
    linear = build_model({"model": {**TINY["model"], "rqs_mode": "linear"}})
    assert linear.rqs.curve == "linear"

    disabled = build_model({"model": {**TINY["model"], "rqs_mode": "disabled"}})
    z = torch.randn(1, 4, 8, 8)
    decoded = disabled.decode_hdr(pack_latents(z), (8, 8))
    assert set(decoded) == {"hdr", "vae_out"}
    assert torch.equal(decoded["hdr"], decoded["vae_out"])
