"""Non-reportable diagnostics for catching HDR output-scale failures."""

from __future__ import annotations

import torch


def pq_code_statistics(frame_pq: torch.Tensor) -> dict[str, float]:
    """Return scalar PQ-code statistics without interpreting code as luminance."""
    values = frame_pq.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty tensor")
    return {
        "mean_pq_code": values.mean().item(),
        "p99_pq_code": torch.quantile(values, 0.99).item(),
        "max_pq_code": values.max().item(),
    }


def diagnostic_ref_max_rescale(
    prediction_pq: torch.Tensor,
    reference_pq: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Match maximum PQ code for diagnosis only.

    This operation uses reference information, so its scores are invalid as
    method results. It exists only to expose whether a large score deficit is
    dominated by a global output-scale mismatch.
    """
    if prediction_pq.shape != reference_pq.shape:
        raise ValueError("prediction and reference shapes must match")
    pred_max = prediction_pq.detach().amax()
    ref_max = reference_pq.detach().amax()
    factor = ref_max / pred_max.clamp(min=1e-8)
    return (prediction_pq * factor).clamp(0.0, 1.0), factor.item()
