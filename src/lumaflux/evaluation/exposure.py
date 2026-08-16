"""Expert-track exposure calibration around unchanged frame metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from ..color.spaces import luma_2020
from ..color.transfer import pq_eotf_nits, pq_oetf_nits

MIN_EXPOSURE_GAIN = 0.25
MAX_EXPOSURE_GAIN = 8.0

FrameScores = dict[str, float | None]
FrameScorer = Callable[[torch.Tensor, torch.Tensor], FrameScores]


def fit_exposure_gain(
    predictions_pq: Sequence[torch.Tensor],
    targets_pq: Sequence[torch.Tensor],
    *,
    min_gain: float = MIN_EXPOSURE_GAIN,
    max_gain: float = MAX_EXPOSURE_GAIN,
) -> float:
    """Fit one content-level gain as the median target/prediction luminance ratio."""
    if len(predictions_pq) != len(targets_pq):
        raise ValueError("predictions and targets must have the same length")
    if not predictions_pq:
        raise ValueError("at least one prediction/target pair is required")
    if not 0.0 < min_gain <= max_gain:
        raise ValueError("gain bounds must satisfy 0 < min_gain <= max_gain")

    ratios = []
    for prediction, target in zip(predictions_pq, targets_pq):
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        prediction_luminance = luma_2020(pq_eotf_nits(prediction))
        target_luminance = luma_2020(pq_eotf_nits(target))
        valid = (
            torch.isfinite(prediction_luminance)
            & torch.isfinite(target_luminance)
            & (prediction_luminance > 1e-6)
        )
        if valid.any():
            ratios.append((target_luminance[valid] / prediction_luminance[valid]).flatten())
    if not ratios:
        return 1.0
    gain = torch.cat(ratios).median().clamp(min_gain, max_gain)
    return float(gain.item())


def apply_exposure_gain(prediction_pq: torch.Tensor, gain: float) -> torch.Tensor:
    """Apply a scalar gain to absolute linear-light RGB and return PQ code values."""
    if gain <= 0.0:
        raise ValueError("gain must be positive")
    if gain == 1.0:
        return prediction_pq
    return pq_oetf_nits(pq_eotf_nits(prediction_pq) * gain)


def score_exposure_calibrated_content(
    predictions_pq: Sequence[torch.Tensor],
    targets_pq: Sequence[torch.Tensor],
    score_frame: FrameScorer,
) -> tuple[float, list[FrameScores]]:
    """Fit one gain for a content and score each calibrated frame with a wrapper."""
    gain = fit_exposure_gain(predictions_pq, targets_pq)
    scores = [
        score_frame(apply_exposure_gain(prediction, gain), target)
        for prediction, target in zip(predictions_pq, targets_pq)
    ]
    return gain, scores
