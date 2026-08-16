import pytest
import torch

from lumaflux.color.transfer import pq_oetf_nits
from lumaflux.data.curate import CurationConfig, SourceSpec, curate_png
from lumaflux.data.dataset import SdrHdrPairs, load_manifest, to_hf_dataset
from lumaflux.data.degrade import codec_degrade, degrade_chain
from lumaflux.data.normalize import normalize_to_pq2020

@pytest.fixture(scope="module")
def curated(synthetic_sources, tmp_path_factory):
    out = tmp_path_factory.mktemp("curated")
    cfg = CurationConfig(
        sources=[
            SourceSpec(name="pgc_demo", path=str(synthetic_sources / "pgc"),
                       category="pgc", format="pq2020"),
            SourceSpec(name="ugc_demo", path=str(synthetic_sources / "ugc"),
                       category="ugc", format="pq2020"),
        ],
        out_dir=str(out),
        tmos=["reinhard", "bt2446c_gm"],
        crf_levels=[31],
    )
    return curate_png(cfg)


def test_normalize_formats():
    x = torch.rand(1, 3, 16, 16)
    for fmt in ("pq2020", "hlg2020", "pq_p3d65", "sdr709"):
        out = normalize_to_pq2020(x, source_format=fmt)
        assert out.shape == x.shape
        assert out.min() >= 0 and out.max() <= 1


def test_normalize_clips_to_peak():
    bright = pq_oetf_nits(torch.full((1, 3, 4, 4), 5000.0))
    out = normalize_to_pq2020(bright, source_format="pq2020", peak_nits=1000.0)
    assert out.max() <= pq_oetf_nits(torch.tensor(1000.0)) + 1e-4


def test_codec_degrade_changes_pixels():
    torch.manual_seed(0)
    x = torch.rand(3, 64, 64)
    out = codec_degrade(x, crf=39)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)  # compression must actually degrade
    assert (out - x).abs().mean() < 0.25  # but stay recognizable


def test_degrade_chain_full():
    x_pq = pq_oetf_nits(torch.rand(3, 64, 64) * 1000)
    sdr = degrade_chain(x_pq, "reinhard", crf=31)
    assert sdr.shape == x_pq.shape
    assert sdr.min() >= 0 and sdr.max() <= 1


def test_curation_manifest(curated):
    records = load_manifest(curated)
    assert len(records) == 4  # 2 scenes x 2 frames
    cats = {r["category"] for r in records}
    assert cats == {"pgc", "ugc"}
    for r in records:
        assert len(r["sdr_variants"]) == 2  # 2 TMOs x 1 CRF
        for rel in [r["hdr"], *r["sdr_variants"].values()]:
            assert (curated.parent / rel).exists()


def test_paired_dataset(curated):
    ds = SdrHdrPairs(curated, crop_size=32)
    item = ds[0]
    assert item["sdr"].shape == (3, 32, 32)
    assert item["hdr"].shape == (3, 32, 32)
    assert 0 <= item["sdr"].min() and item["sdr"].max() <= 1
    assert 0 <= item["hdr"].min() and item["hdr"].max() <= 1


def test_hf_dataset_export(curated):
    ds = to_hf_dataset(curated)
    assert len(ds) == 8  # 4 records x 2 variants
    assert set(ds.column_names) >= {"id", "sdr_path", "hdr_path", "variant"}
