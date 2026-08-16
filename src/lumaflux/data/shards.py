"""Safetensors shard writing and atomic JSONL manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

SHARD_FORMAT = "lumaflux-shards"
DEFAULT_SHARD_BYTES = 1_000_000_000


class ShardWriter:
    """Accumulate tensors into immutable safetensors files near a byte target."""

    def __init__(
        self,
        root: str | Path,
        prefix: str,
        *,
        target_bytes: int = DEFAULT_SHARD_BYTES,
        metadata: dict[str, str] | None = None,
    ) -> None:
        if target_bytes <= 0:
            raise ValueError("target_bytes must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.target_bytes = target_bytes
        self.metadata = {"format": SHARD_FORMAT, **(metadata or {})}
        self.index = 0
        self.tensors: dict[str, torch.Tensor] = {}
        self.nbytes = 0
        self.paths: list[Path] = []

    @property
    def current_name(self) -> str:
        return f"{self.prefix}-{self.index:05d}.safetensors"

    def add(self, key: str, tensor: torch.Tensor) -> str:
        tensor = tensor.detach().cpu().contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if self.tensors and self.nbytes + tensor_bytes > self.target_bytes:
            self.flush()
        if key in self.tensors:
            raise KeyError(f"Duplicate tensor key in shard: {key}")
        shard_name = self.current_name
        self.tensors[key] = tensor
        self.nbytes += tensor_bytes
        return shard_name

    def flush(self) -> None:
        if not self.tensors:
            return
        path = self.root / self.current_name
        save_file(self.tensors, path, metadata=self.metadata)
        self.paths.append(path)
        self.tensors = {}
        self.nbytes = 0
        self.index += 1

    def close(self) -> list[Path]:
        self.flush()
        return list(self.paths)


def write_jsonl_atomic(
    path: str | Path,
    records: list[dict],
    *,
    header: dict | None = None,
) -> Path:
    """Replace a JSONL manifest only after all content has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        if header is not None:
            handle.write(json.dumps({"type": "header", **header}, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)
    return path
