"""Pure shape arithmetic for the full FLUX.1-dev trainable parameter budget."""


def linear(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def test_flux_dev_trainable_parameter_budget():
    # FLUX.1-dev and SigLIP SO400M dimensions. No model is instantiated here.
    dim = 3072
    context_dim = 4096
    pooled_dim = 768
    siglip_dim = 1152
    latent_channels = 16
    heads = 24
    blocks = 19 + 38

    rank = 8
    bottleneck = 64
    phys_channels = 32
    stats_dim = 16
    num_bands = 8
    num_null_tokens = 8
    modulation_hidden = 128
    num_knots = 8

    # PhysicalEncoder.
    physical = phys_channels * 3 * 3 * 3 + phys_channels
    physical += linear(4, stats_dim) + linear(stats_dim, stats_dim)

    # Two PerceptualConnector bottlenecks.
    connectors = linear(siglip_dim, bottleneck) + linear(bottleneck, dim)
    connectors += linear(siglip_dim, bottleneck) + linear(bottleneck, context_dim)

    prompt_context = num_null_tokens * context_dim + pooled_dim
    prompt_context += linear(stats_dim, pooled_dim)

    # Shared TimestepLayerModulation.
    modulation = linear(modulation_hidden, modulation_hidden)
    modulation += blocks * modulation_hidden
    modulation += linear(modulation_hidden, modulation_hidden)
    modulation += linear(modulation_hidden, 6)

    # PGAValueAdapter, PCMModulator, and both coupler paths in each block.
    pga = linear(dim, rank, bias=False) + linear(rank, dim, bias=False)
    pga += linear(phys_channels + stats_dim, heads) + linear(num_bands, heads)
    pcm = linear(dim, bottleneck) + linear(bottleneck, 2 * dim)
    coupler = linear(phys_channels, bottleneck, bias=False)
    coupler += linear(bottleneck, dim, bias=False)
    coupler += linear(dim, bottleneck, bias=False)
    coupler += linear(bottleneck, dim, bias=False)

    # RQSToneFieldDecoder, including its 2x2 chroma point projection.
    rqs_hidden = 128
    rqs = linear(latent_channels, rqs_hidden)
    rqs += linear(rqs_hidden, rqs_hidden)
    rqs += linear(rqs_hidden, 3 * num_knots + 1)
    rqs += 2 * 2 + 2

    total = physical + connectors + prompt_context + modulation
    total += blocks * (pga + pcm + coupler) + rqs

    assert total == 71_315_893
    assert 40_000_000 <= total <= 80_000_000
    assert total < 100_000_000
