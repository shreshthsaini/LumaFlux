"""Paired SDR/HDR datasets over curation manifests.

``SdrHdrShards`` is the primary training path. It slices memory-mapped integer
tensors before collate-time scaling. ``SdrHdrPairs`` retains the deprecated PNG
format for compatibility with older manifests.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from ..utils.io import load_hdr_png16, load_sdr_png


def load_manifest(manifest: str | Path) -> list[dict]:
    with open(manifest) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return [record for record in records if record.get("type") != "header"]


def load_manifest_header(manifest: str | Path) -> dict | None:
    """Return an optional first-line manifest header."""
    with open(manifest) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            return record if record.get("type") == "header" else None
    return None


def is_shard_manifest(manifest: str | Path) -> bool:
    """Identify a shard manifest without opening any tensor files."""
    with open(manifest) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                continue
            required = {"hdr_shard", "hdr_key", "sdr_shard", "sdr_key"}
            return required <= record.keys()
    return False


class SdrHdrShards(Dataset):
    """Memory-mapped safetensors pairs with synchronized random integer crops."""

    def __init__(
        self,
        manifest: str | Path,
        crop_size: int | None = 512,
        random_flip: bool = True,
        seed: int = 0,
        max_open_shards: int = 32,
    ) -> None:
        self.root = Path(manifest).parent
        self.records = load_manifest(manifest)
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")
        if not is_shard_manifest(manifest):
            raise ValueError(f"Not a safetensors shard manifest: {manifest}")
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.rng = random.Random(seed)
        self.max_open_shards = max_open_shards
        self._handles: OrderedDict[Path, object] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _handle(self, relative: str):
        path = self.root / relative
        handle = self._handles.pop(path, None)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
        self._handles[path] = handle
        while len(self._handles) > self.max_open_shards:
            self._handles.popitem(last=False)
        return handle

    @staticmethod
    def _pad_replicate(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        rows = torch.arange(height).clamp(max=x.shape[-2] - 1)
        columns = torch.arange(width).clamp(max=x.shape[-1] - 1)
        return x.index_select(-2, rows).index_select(-1, columns)

    def _read_pair(self, record: dict) -> tuple[torch.Tensor, torch.Tensor]:
        sdr_slice = self._handle(record["sdr_shard"]).get_slice(record["sdr_key"])
        hdr_slice = self._handle(record["hdr_shard"]).get_slice(record["hdr_key"])
        sdr_shape, hdr_shape = tuple(sdr_slice.get_shape()), tuple(hdr_slice.get_shape())
        if sdr_shape != hdr_shape:
            raise ValueError(
                f"Mismatched shard pair shapes for {record['sdr_key']}: {sdr_shape} vs {hdr_shape}"
            )
        _, height, width = sdr_shape
        if self.crop_size is None:
            top, left, crop_height, crop_width = 0, 0, height, width
        else:
            crop_height = crop_width = self.crop_size
            top = self.rng.randint(0, max(0, height - crop_height))
            left = self.rng.randint(0, max(0, width - crop_width))
        bottom, right = min(height, top + crop_height), min(width, left + crop_width)
        sdr = sdr_slice[:, top:bottom, left:right]
        hdr = hdr_slice[:, top:bottom, left:right]
        if sdr.shape[-2:] != (crop_height, crop_width):
            sdr = self._pad_replicate(sdr, crop_height, crop_width)
            hdr = self._pad_replicate(hdr, crop_height, crop_width)
        if self.random_flip and self.rng.random() < 0.5:
            reverse_columns = torch.arange(sdr.shape[-1] - 1, -1, -1)
            sdr = sdr.index_select(-1, reverse_columns)
            hdr = hdr.index_select(-1, reverse_columns)
        return sdr, hdr

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        sdr, hdr = self._read_pair(record)
        variant = (
            record["tmo"]
            if record["crf"] is None
            else f"{record['tmo']}_crf{record['crf']}"
        )
        return {
            "sdr": sdr,
            "hdr": hdr,
            "id": f"{record['video']}:{record['frame']}",
            "variant": variant,
            "video": record["video"],
            "frame": record["frame"],
            "tmo": record["tmo"],
            "crf": record["crf"],
            "split": record["split"],
            "source_dataset": record["source_dataset"],
        }


def collate_sdr_hdr(batch: list[dict]) -> dict:
    """Stack paired samples and scale integer code values to float32 [0, 1]."""
    sdr = torch.stack([sample["sdr"] for sample in batch])
    hdr = torch.stack([sample["hdr"] for sample in batch])
    if sdr.dtype == torch.uint8:
        sdr = sdr.to(torch.float32).div_(255.0)
    if hdr.dtype == torch.uint16:
        hdr = hdr.to(torch.float32).div_(65535.0)
    result = {"sdr": sdr, "hdr": hdr}
    for key in batch[0].keys() - {"sdr", "hdr"}:
        result[key] = [sample[key] for sample in batch]
    return result


class SdrHdrPairs(Dataset):
    """Deprecated PNG paired dataset with synchronized crops and flips."""

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
