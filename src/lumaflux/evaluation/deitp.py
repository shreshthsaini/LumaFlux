"""Delta-E ITP color difference, ITU-R BT.2124 (Sec. 6.1).

Computed in ICtCp space from PQ/BT.2020 signals:

    dE_ITP = 720 * sqrt(dI^2 + dT^2 + dP^2),  T = 0.5 * Ct
"""

from __future__ import annotations

import torch

from ..color.spaces import pq2020_to_ictcp


def delta_e_itp(pred_pq: torch.Tensor, target_pq: torch.Tensor,
                reduce: str = "mean") -> torch.Tensor:
    a = pq2020_to_ictcp(pred_pq)
    b = pq2020_to_ictcp(target_pq)
    di = a[..., 0, :, :] - b[..., 0, :, :]
    dt = 0.5 * (a[..., 1, :, :] - b[..., 1, :, :])
    dp = a[..., 2, :, :] - b[..., 2, :, :]
    de = 720.0 * torch.sqrt(di**2 + dt**2 + dp**2 + 1e-20)
    if reduce == "mean":
        return de.mean()
    if reduce == "none":
        return de
    raise ValueError(reduce)
