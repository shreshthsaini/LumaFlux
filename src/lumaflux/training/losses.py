"""Training objective (Sec. 4.6, Eq. 18) + latent flow-matching term.

The latent objective trains the adapters to steer the frozen backbone along
the SDR->HDR latent bridge; the pixel-domain terms supervise physical
fidelity in linear light through the frozen VAE decoder and the RQS head:

    L = w_v * ||v_pred - v*||^2
      + lambda1 * ||Y_lin - Y*_lin||_1
      + lambda2 * ||x_lin - x*_lin||_1
      + lambda3 * L_spline-smooth

Linear-light terms use the ST 2084 EOTF to recover absolute cd/m2, then divide
by the fixed corpus mastering peak. The spline smoothness penalty stabilizes
adjacent RQS knot slopes.
"""

from __future__ import annotations

import torch

from ..color.spaces import luma_2020
from ..color.transfer import CORPUS_PEAK_NITS, pq_eotf_nits
from ..models.rqs import spline_smoothness_loss


def velocity_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(v_pred.float(), v_target.float())


def reconstruction_losses(
    hdr_pred_pq: torch.Tensor,
    hdr_target_pq: torch.Tensor,
    spline: dict[str, torch.Tensor] | None = None,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    lambda3: float = 0.1,
    peak_nits: float = CORPUS_PEAK_NITS,
) -> dict[str, torch.Tensor]:
    """Both frames are PQ/BT.2020 signals mastered at ``peak_nits``."""
    if peak_nits <= 0.0:
        raise ValueError("peak_nits must be positive")
    pred_lin = pq_eotf_nits(hdr_pred_pq) / peak_nits
    tgt_lin = pq_eotf_nits(hdr_target_pq) / peak_nits
    l_luma = (luma_2020(pred_lin) - luma_2020(tgt_lin)).abs().mean()
    l_rgb = (pred_lin - tgt_lin).abs().mean()
    out = {"luma_l1": l_luma, "rgb_l1": l_rgb}
    total = lambda1 * l_luma + lambda2 * l_rgb
    if spline is not None:
        l_smooth = spline_smoothness_loss(spline["widths"], spline["heights"], spline["derivs"])
        out["spline_smooth"] = l_smooth
        total = total + lambda3 * l_smooth
    out["recon_total"] = total
    return out
