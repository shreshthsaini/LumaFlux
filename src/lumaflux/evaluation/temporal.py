"""Temporal consistency metrics in PU21 luminance space."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..color.spaces import luma_2020, pu21_encode
from ..color.transfer import pq_eotf_nits


def _as_sequence(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("temporal inputs must have shape (T, 3, H, W)")
    if frames.shape[0] < 2:
        raise ValueError("temporal consistency requires at least two frames")
    return frames


def pu21_flicker_energy(frames_pq: torch.Tensor) -> torch.Tensor:
    """Mean adjacent-frame absolute change in PU21 luminance."""
    frames = _as_sequence(frames_pq)
    luminance = pu21_encode(luma_2020(pq_eotf_nits(frames)))
    differences = (luminance[1:] - luminance[:-1]).abs().flatten(1)
    return differences.sum(dim=1).mean()


def temporal_consistency(pred_pq: torch.Tensor, target_pq: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return output, reference, and output-minus-reference flicker energy."""
    if pred_pq.shape != target_pq.shape:
        raise ValueError("temporal inputs must have matching shapes")
    raw = pu21_flicker_energy(pred_pq)
    reference = pu21_flicker_energy(target_pq)
    return {
        "flicker_raw": raw,
        "flicker_reference": reference,
        "flicker_excess": raw - reference,
    }


@dataclass
class TemporalAccumulator:
    """Streaming form of :func:`temporal_consistency`."""

    previous_pred: torch.Tensor | None = None
    previous_target: torch.Tensor | None = None
    pred_total: float = 0.0
    target_total: float = 0.0
    transitions: int = 0

    @staticmethod
    def _pu21_luminance(frame: torch.Tensor) -> torch.Tensor:
        return pu21_encode(luma_2020(pq_eotf_nits(frame.detach().to("cpu"))))

    def update(self, pred_pq: torch.Tensor, target_pq: torch.Tensor) -> None:
        if pred_pq.shape != target_pq.shape or pred_pq.ndim != 3 or pred_pq.shape[0] != 3:
            raise ValueError("temporal accumulator inputs must have matching (3, H, W) shapes")
        pred = self._pu21_luminance(pred_pq)
        target = self._pu21_luminance(target_pq)
        if self.previous_pred is not None:
            self.pred_total += (pred - self.previous_pred).abs().sum().item()
            self.target_total += (target - self.previous_target).abs().sum().item()
            self.transitions += 1
        self.previous_pred = pred
        self.previous_target = target

    def compute(self) -> dict[str, float | str | None]:
        if self.transitions == 0:
            return {
                "flicker_raw": None,
                "flicker_reference": None,
                "flicker_excess": None,
                "temporal": "skipped (single frame)",
            }
        raw = self.pred_total / self.transitions
        reference = self.target_total / self.transitions
        return {
            "flicker_raw": raw,
            "flicker_reference": reference,
            "flicker_excess": raw - reference,
        }
