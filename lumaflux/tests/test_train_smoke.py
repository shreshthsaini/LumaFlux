import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def curated(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    subprocess.run(
        [sys.executable, str(REPO / "scripts/make_synthetic_data.py"),
         "--out-dir", str(root), "--num-scenes", "1",
         "--frames-per-scene", "2", "--size", "64"],
        check=True,
    )
    from lumaflux.data.curate import CurationConfig, SourceSpec, curate

    out = tmp_path_factory.mktemp("curated")
    cfg = CurationConfig(
        sources=[SourceSpec(name="demo", path=str(root / "pgc"), format="pq2020")],
        out_dir=str(out),
        tmos=["reinhard"],
        crf_levels=[31],
    )
    return curate(cfg)


def test_train_smoke_and_resume(curated, tmp_path, monkeypatch):
    """A few optimizer steps on the tiny stack: loss finite, trackio logs,
    checkpoint written, resume works."""
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path / "trackio"))
    with open(REPO / "configs/model/tiny_debug.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["train"].update(max_steps=4, ckpt_every=2, log_every=1, recon_every=2,
                        batch_size=2, crop_size=64, num_workers=0)

    from lumaflux.training.train import train

    result = train(cfg, str(curated), str(tmp_path / "run1"))
    assert result["steps"] == 4
    assert result["final_loss"] == result["final_loss"]  # not NaN
    final = tmp_path / "run1/adapters_final.safetensors"
    assert final.exists()
    assert (tmp_path / "run1/adapters_step0000002.safetensors").exists()

    # Resume from the final checkpoint.
    result2 = train(cfg, str(curated), str(tmp_path / "run2"), resume=str(final))
    assert result2["steps"] == 4

    # trackio wrote its local run database.
    trackio_dir = tmp_path / "trackio"
    assert trackio_dir.exists() and any(trackio_dir.iterdir())
