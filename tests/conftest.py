from __future__ import annotations

import pytest
import torch

from lumaflux.utils.io import save_hdr_png16


@pytest.fixture(scope="session")
def synthetic_sources(tmp_path_factory: pytest.TempPathFactory):
    """Create two tiny PQ image sequences for curation tests."""
    root = tmp_path_factory.mktemp("synthetic-sources")
    base = torch.linspace(0.02, 0.72, 64 * 64).reshape(1, 64, 64)
    for category_index, category in enumerate(("pgc", "ugc")):
        sequence = root / category / "scene"
        sequence.mkdir(parents=True)
        for frame_index in range(2):
            offsets = torch.tensor([0.00, 0.02, 0.04]).reshape(3, 1, 1)
            frame = (base + offsets + 0.01 * (category_index + frame_index)).clamp(0, 1)
            save_hdr_png16(frame, sequence / f"frame-{frame_index:03d}.png")
    return root
