"""Electro-optical / opto-electronic transfer functions.

All functions operate on torch tensors of arbitrary shape with values in
nominal ranges. ST 2084 always uses its absolute 10,000 cd/m2 reference
scale. It is not the mastering peak. LumaFlux corpus signals are mastered at
1,000 cd/m2, whose maximum legal corpus code is about 0.7518.

References:
    SMPTE ST 2084:2014 (PQ), ITU-R BT.709-6, ITU-R BT.1886, ITU-R BT.2100-2.
"""

from __future__ import annotations

import torch

# SMPTE ST 2084 (PQ) constants.
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0

PQ_PEAK_NITS = 10_000.0
CORPUS_PEAK_NITS = 1_000.0


def pq_oetf(linear: torch.Tensor) -> torch.Tensor:
    """ST 2084 inverse EOTF: normalized linear light [0,1] -> PQ signal [0,1].

    ``linear`` is absolute luminance divided by 10,000 nits. The tiny
    positive clamp keeps pow() gradients finite at 0 (exponent < 1).
    """
    lin = linear.clamp(min=1e-10, max=1.0)
    lm = lin.pow(PQ_M1)
    return ((PQ_C1 + PQ_C2 * lm) / (1.0 + PQ_C3 * lm)).pow(PQ_M2)


def pq_eotf(signal: torch.Tensor) -> torch.Tensor:
    """ST 2084 EOTF: PQ signal [0,1] -> normalized linear light [0,1].

    The tiny positive clamp keeps pow() gradients finite at 0 (exponent < 1);
    the value impact is < 1e-18 in linear light.
    """
    e = signal.clamp(min=1e-8, max=1.0)
    em = e.pow(1.0 / PQ_M2)
    num = (em - PQ_C1).clamp(min=0.0)
    den = PQ_C2 - PQ_C3 * em
    return (num / den).pow(1.0 / PQ_M1)


def pq_eotf_nits(signal: torch.Tensor) -> torch.Tensor:
    """PQ signal -> absolute luminance in cd/m^2."""
    return pq_eotf(signal) * PQ_PEAK_NITS


def pq_oetf_nits(nits: torch.Tensor) -> torch.Tensor:
    """Absolute luminance in cd/m^2 -> PQ signal."""
    return pq_oetf(nits / PQ_PEAK_NITS)


CORPUS_PEAK_PQ = pq_oetf_nits(torch.tensor(CORPUS_PEAK_NITS, dtype=torch.float64)).item()


def bt709_oetf(linear: torch.Tensor) -> torch.Tensor:
    """BT.709 OETF: scene-linear [0,1] -> gamma-encoded SDR signal [0,1]."""
    lin = linear.clamp(min=0.0, max=1.0)
    lo = 4.5 * lin
    hi = 1.099 * lin.clamp(min=1e-12).pow(0.45) - 0.099
    return torch.where(lin < 0.018, lo, hi)


def bt709_eotf(signal: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`bt709_oetf` (camera-side inverse, not BT.1886)."""
    e = signal.clamp(min=0.0, max=1.0)
    lo = e / 4.5
    hi = ((e + 0.099) / 1.099).clamp(min=1e-12).pow(1.0 / 0.45)
    return torch.where(e < 0.018 * 4.5, lo, hi)


def bt1886_eotf(signal: torch.Tensor, gamma: float = 2.4) -> torch.Tensor:
    """BT.1886 display EOTF: SDR signal [0,1] -> display-linear [0,1]."""
    return signal.clamp(0.0, 1.0).pow(gamma)


def bt1886_inverse_eotf(linear: torch.Tensor, gamma: float = 2.4) -> torch.Tensor:
    return linear.clamp(0.0, 1.0).pow(1.0 / gamma)


def hlg_oetf(linear: torch.Tensor) -> torch.Tensor:
    """BT.2100 HLG OETF: scene-linear [0,1] -> HLG signal [0,1]."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    lin = linear.clamp(min=0.0, max=1.0)
    lo = (3.0 * lin).clamp(min=0.0).sqrt()
    hi = a * (12.0 * lin - b).clamp(min=1e-12).log() + c
    return torch.where(lin <= 1.0 / 12.0, lo, hi)


def hlg_inverse_oetf(signal: torch.Tensor) -> torch.Tensor:
    """HLG signal [0,1] -> scene-linear [0,1]."""
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    e = signal.clamp(min=0.0, max=1.0)
    lo = e.pow(2.0) / 3.0
    hi = (((e - c) / a).exp() + b) / 12.0
    return torch.where(e <= 0.5, lo, hi)
