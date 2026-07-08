"""HDR-LPIPS: LPIPS computed on PU21-encoded frames (Sec. 6.1).

Requires the ``lpips`` package and its pretrained AlexNet weights (network
access on first use); when weights cannot be fetched the metric reports
``None`` and the benchmark runner skips the column.
"""

from __future__ import annotations

import torch

from ..color.spaces import PU21_PEAK, pu21_encode
from ..color.transfer import pq_eotf_nits

_lpips_model = None


def _get_model():
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex", verbose=False)
        _lpips_model.eval()
    return _lpips_model


def hdr_lpips(pred_pq: torch.Tensor, target_pq: torch.Tensor) -> torch.Tensor | None:
    """Both inputs PQ/BT.2020 (3,H,W) or (B,3,H,W) in [0,1]."""
    try:
        model = _get_model()
    except Exception:
        return None
    if pred_pq.dim() == 3:
        pred_pq, target_pq = pred_pq.unsqueeze(0), target_pq.unsqueeze(0)
    a = pu21_encode(pq_eotf_nits(pred_pq)) / PU21_PEAK * 2.0 - 1.0
    b = pu21_encode(pq_eotf_nits(target_pq)) / PU21_PEAK * 2.0 - 1.0
    with torch.no_grad():
        return model(a, b).mean()
