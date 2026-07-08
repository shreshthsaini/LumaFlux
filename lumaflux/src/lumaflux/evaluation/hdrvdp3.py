"""Optional HDR-VDP-3 hook.

HDR-VDP-3 (Mantiuk et al., 2023) ships as a MATLAB toolbox with no official
Python port. If a MATLAB (or GNU Octave) installation with the toolbox is
available, point ``HDRVDP3_PATH`` at it and this wrapper will shell out per
frame pair; otherwise the metric is reported as ``None`` and skipped by the
benchmark runner.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..color.transfer import pq_eotf_nits


def hdrvdp3_available() -> bool:
    return bool(os.environ.get("HDRVDP3_PATH")) and (
        shutil.which("matlab") is not None or shutil.which("octave") is not None
    )


def hdr_vdp3(pred_pq: torch.Tensor, target_pq: torch.Tensor,
             pixels_per_degree: float = 30.0) -> float | None:
    """Returns the HDR-VDP-3 quality score (Q_JOD-style) or None."""
    if not hdrvdp3_available():
        return None
    toolbox = os.environ["HDRVDP3_PATH"]
    runner = "matlab" if shutil.which("matlab") else "octave"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, x in (("pred", pred_pq), ("ref", target_pq)):
            nits = pq_eotf_nits(x).cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            np.save(td / f"{name}.npy", nits)
        script = f"""
        addpath(genpath('{toolbox}'));
        pred = double(readNPY('{td}/pred.npy'));
        ref  = double(readNPY('{td}/ref.npy'));
        res = hdrvdp3('quality', pred, ref, 'rgb-bt.2020', {pixels_per_degree});
        fid = fopen('{td}/q.txt','w'); fprintf(fid, '%f', res.Q); fclose(fid);
        """
        (td / "run_vdp.m").write_text(script)
        cmd = ([runner, "-batch", f"run('{td}/run_vdp.m')"] if runner == "matlab"
               else [runner, "--eval", script])
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            return float((td / "q.txt").read_text())
        except Exception:
            return None
