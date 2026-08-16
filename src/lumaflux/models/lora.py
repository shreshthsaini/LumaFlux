"""Small, dependency-free LoRA wrappers for backbone ablations."""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen linear projection plus a zero-initialized low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frozen = self.base(x)
        adapter_input = x.to(self.down.weight.dtype)
        residual = self.up(self.down(adapter_input)) * self.scale
        return frozen + residual.to(frozen.dtype)


def _is_lora_target(name: str, mode: str) -> bool:
    if ".attn." in name:
        return True
    if mode == "attention_mlp":
        return any(
            marker in name
            for marker in (".ff.", ".ff_context.", ".proj_mlp", ".proj_out")
        )
    return False


def _replace_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def inject_backbone_lora(
    transformer: nn.Module,
    *,
    rank: int,
    mode: str,
) -> list[LoRALinear]:
    """Inject LoRA into attention only or into attention plus MLP projections."""
    if mode not in {"none", "attention", "attention_mlp"}:
        raise ValueError(f"Unknown LoRA mode: {mode}")
    if mode == "none":
        return []
    replacements = []
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, nn.Linear) or not _is_lora_target(name, mode):
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = transformer.get_submodule(parent_name)
        wrapped = LoRALinear(module, rank)
        _replace_child(parent, child_name, wrapped)
        replacements.append(wrapped)
    if not replacements:
        raise RuntimeError(f"LoRA mode {mode} matched no backbone projections")
    return replacements


__all__ = ["LoRALinear", "inject_backbone_lora"]
