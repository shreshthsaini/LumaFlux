"""Paired SDR-HDR dataset over a curation manifest (Sec. 5).

Implements the paper's sampling strategy: PGC and UGC records are drawn in
a 1:1 ratio, and each draw picks a random SDR variant (TMO x CRF or expert)
of the chosen HDR frame. Also provides an export to a HuggingFace
``datasets.Dataset`` for hub-hosted corpora.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..utils.io import load_hdr_png16, load_sdr_png


def load_manifest(manifest: str | Path) -> list[dict]:
    with open(manifest) as f:
        return [json.loads(line) for line in f if line.strip()]


class SdrHdrPairs(Dataset):
    """Random-variant paired dataset with synchronized crops/flips."""

    def __init__(
        self,
        manifest: str | Path,
        crop_size: int | None = 512,
        random_flip: bool = True,
        balance_categories: bool = True,
        seed: int = 0,
    ) -> None:
        self.root = Path(manifest).parent
        self.records = load_manifest(manifest)
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.rng = random.Random(seed)
        self.by_cat: dict[str, list[dict]] = {}
        for r in self.records:
            self.by_cat.setdefault(r.get("category", "pgc"), []).append(r)
        self.balance = balance_categories and len(self.by_cat) > 1

    def __len__(self) -> int:
        return len(self.records)

    def _pick_record(self, idx: int) -> dict:
        if self.balance:
            # 1:1 PGC/UGC sampling regardless of corpus imbalance.
            cat = self.rng.choice(sorted(self.by_cat))
            return self.rng.choice(self.by_cat[cat])
        return self.records[idx]

    def _crop_flip(self, sdr: torch.Tensor, hdr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.crop_size is not None:
            c = self.crop_size
            h, w = sdr.shape[-2:]
            if h < c or w < c:
                pad_h, pad_w = max(0, c - h), max(0, c - w)
                sdr = torch.nn.functional.pad(sdr, (0, pad_w, 0, pad_h), mode="replicate")
                hdr = torch.nn.functional.pad(hdr, (0, pad_w, 0, pad_h), mode="replicate")
                h, w = sdr.shape[-2:]
            top = self.rng.randint(0, h - c)
            left = self.rng.randint(0, w - c)
            sdr = sdr[..., top : top + c, left : left + c]
            hdr = hdr[..., top : top + c, left : left + c]
        if self.random_flip and self.rng.random() < 0.5:
            sdr = sdr.flip(-1)
            hdr = hdr.flip(-1)
        return sdr, hdr

    def __getitem__(self, idx: int) -> dict:
        rec = self._pick_record(idx)
        variant_key = self.rng.choice(sorted(rec["sdr_variants"]))
        sdr = load_sdr_png(self.root / rec["sdr_variants"][variant_key])
        hdr = load_hdr_png16(self.root / rec["hdr"])
        if sdr.shape[-2:] != hdr.shape[-2:]:
            sdr = torch.nn.functional.interpolate(
                sdr.unsqueeze(0), size=hdr.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0)
        sdr, hdr = self._crop_flip(sdr, hdr)
        return {"sdr": sdr, "hdr": hdr, "id": rec["id"], "variant": variant_key,
                "category": rec.get("category", "pgc")}


def to_hf_dataset(manifest: str | Path):
    """Export the manifest as a HuggingFace ``datasets.Dataset`` of file paths."""
    from datasets import Dataset as HFDataset

    root = Path(manifest).parent
    rows = []
    for rec in load_manifest(manifest):
        for variant, rel in rec["sdr_variants"].items():
            rows.append(
                {
                    "id": rec["id"],
                    "category": rec.get("category", "pgc"),
                    "variant": variant,
                    "sdr_path": str(root / rel),
                    "hdr_path": str(root / rec["hdr"]),
                }
            )
    return HFDataset.from_list(rows)
