from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[1]


def test_registered_training_configs():
    main = yaml.safe_load((REPO / "configs/train/main.yaml").read_text())
    hdrtv1k = yaml.safe_load((REPO / "configs/train/hdrtv1k.yaml").read_text())

    assert main["model"]["backbone"] == "black-forest-labs/FLUX.1-dev"
    assert main["data"]["manifest"] == "data/curated/manifest.jsonl"
    assert main["train"]["lr"] == 2e-4
    assert main["train"]["batch_size"] == 2
    assert main["train"]["grad_accum"] == 2
    assert main["train"]["max_steps"] == 100_000
    assert main["train"]["warmup_steps"] == 5_000
    assert main["train"]["scheduler"] == "cosine"
    assert main["train"]["ckpt_every"] == 5_000
    assert main["train"]["ckpt_keep"] == 3
    assert main["train"]["run_name"] == "lumaflux-main"

    assert hdrtv1k["data"]["manifest"] == "data/hdrtv1k/manifest.jsonl"
    assert hdrtv1k["train"]["max_steps"] == 50_000
    assert hdrtv1k["train"]["run_name"] == "lumaflux-hdrtv1k"


@pytest.fixture(scope="module")
def curated(synthetic_sources, tmp_path_factory):
    from lumaflux.data.curate import CurationConfig, SourceSpec, curate

    out = tmp_path_factory.mktemp("training-curated")
    cfg = CurationConfig(
        sources=[
            SourceSpec(name="demo", path=str(synthetic_sources / "pgc"), format="pq2020")
        ],
        out_dir=str(out),
        tmos=["reinhard"],
        crf_levels=[31],
        workers=1,
    )
    return curate(cfg)


def test_train_resume_is_exactly_equivalent(curated, tmp_path, monkeypatch):
    """Four steps, resume for four, and an uninterrupted eight must match."""
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path / "trackio"))
    with open(REPO / "configs/model/tiny_debug.yaml") as handle:
        cfg = yaml.safe_load(handle)
    cfg["train"].update(
        max_steps=8,
        ckpt_every=1,
        log_every=1,
        recon_every=2,
        batch_size=2,
        crop_size=64,
        num_workers=0,
        run_name="resume-equivalence",
    )

    from lumaflux.training.train import train

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = train(cfg, str(curated), str(uninterrupted_dir))

    resumed_dir = tmp_path / "resumed"
    first_half = train(cfg, str(curated), str(resumed_dir), stop_at_step=4)
    checkpoint = resumed_dir / "checkpoint_final.pt"
    resumed = train(cfg, str(curated), str(resumed_dir), resume=str(checkpoint))

    assert first_half["global_step"] == 4
    assert resumed["global_step"] == uninterrupted["global_step"] == 8
    assert resumed["completed_optimizer_steps"] == 8
    assert resumed["lr"] == uninterrupted["lr"]

    resumed_weights = load_file(resumed_dir / "adapters_final.safetensors")
    uninterrupted_weights = load_file(uninterrupted_dir / "adapters_final.safetensors")
    assert resumed_weights.keys() == uninterrupted_weights.keys()
    for name in resumed_weights:
        assert torch.equal(resumed_weights[name], uninterrupted_weights[name]), name

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["global_step"] == state["completed_optimizer_steps"] == 8
    assert set(state) >= {"adapters", "optimizer", "scheduler", "rng"}

    periodic = sorted(uninterrupted_dir.glob("checkpoint_step*.pt"))
    assert [path.stem for path in periodic] == [
        f"checkpoint_step{step:07d}" for step in range(1, 9)
    ]
    assert (uninterrupted_dir / "checkpoint_final.pt").exists()

    trackio_dir = tmp_path / "trackio"
    assert trackio_dir.exists() and any(trackio_dir.iterdir())


def test_step_counter_counts_optimizer_steps_under_accumulation(
    curated, tmp_path, monkeypatch
):
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path / "trackio-accum"))
    with open(REPO / "configs/model/tiny_debug.yaml") as handle:
        cfg = yaml.safe_load(handle)
    cfg["train"].update(
        max_steps=2,
        grad_accum=2,
        batch_size=1,
        recon_every=0,
        ckpt_every=0,
        log_every=1,
        crop_size=64,
        num_workers=0,
        run_name="optimizer-step-count",
    )

    from lumaflux.models.luma_flux import LumaFluxModel
    from lumaflux.training.train import train

    calls = 0
    original_forward = LumaFluxModel.forward

    def counted_forward(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(LumaFluxModel, "forward", counted_forward)
    history = []
    result = train(
        cfg,
        str(curated),
        str(tmp_path / "accum"),
        metrics_callback=lambda step, metrics: history.append((step, metrics)),
    )
    assert result["global_step"] == result["completed_optimizer_steps"] == 2
    assert calls == 4
    assert [step for step, _ in history] == [1, 2]
    assert all(
        {"loss/velocity", "loss/total", "lr", "train/step"} <= metrics.keys()
        for _, metrics in history
    )
