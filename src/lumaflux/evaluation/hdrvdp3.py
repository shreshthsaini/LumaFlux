"""HDR-VDP-3 through a configurable MATLAB or GNU Octave subprocess."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..color.transfer import pq_eotf_nits
from .backends import EvalBackendUnavailable

_METRIC = "hdr_vdp3"
_DEFAULT_TOOLBOX = Path(__file__).resolve().parents[3] / "third_party" / "hdrvdp3"


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _toolbox_path() -> Path:
    return Path(os.environ.get("HDRVDP3_PATH", _DEFAULT_TOOLBOX)).expanduser().resolve()


def _resolve_backend() -> tuple[str, str, Path]:
    toolbox = _toolbox_path()
    required = (toolbox / "hdrvdp3.m", toolbox / "npy-matlab" / "readNPY.m")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise EvalBackendUnavailable(_METRIC, f"toolbox files missing: {', '.join(missing)}")

    configured = os.environ.get("HDRVDP3_BACKEND")
    if configured:
        executable = shutil.which(configured)
        if executable is None and Path(configured).is_file():
            executable = str(Path(configured).resolve())
        if executable is None:
            raise EvalBackendUnavailable(_METRIC, f"configured backend not found: {configured}")
        name = Path(executable).name.lower()
        kind = "octave" if "octave" in name else "matlab"
        return kind, executable, toolbox

    matlab = shutil.which("matlab")
    if matlab:
        return "matlab", matlab, toolbox
    octave = shutil.which("octave") or shutil.which("octave-cli")
    if octave:
        return "octave", octave, toolbox
    raise EvalBackendUnavailable(
        _METRIC,
        "neither matlab nor octave is on PATH; set HDRVDP3_BACKEND to an executable",
    )


def hdrvdp3_available() -> bool:
    """Return whether static backend discovery succeeds."""
    try:
        _resolve_backend()
    except EvalBackendUnavailable:
        return False
    return True


def hdr_vdp3(
    pred_pq: torch.Tensor,
    target_pq: torch.Tensor,
    pixels_per_degree: float = 30.0,
    timeout: float = 600.0,
    *,
    task: str = "quality",
    display_model: str | None = None,
) -> float:
    """Return the HDR-VDP-3 Q score for two PQ/BT.2020 frames.

    Inputs must have shape ``(3, H, W)``. Backend discovery and execution
    failures raise :class:`EvalBackendUnavailable` with diagnostic output.
    ``task`` and ``display_model`` expose the HDRTV1K side-by-side protocol
    while retaining the Luma-Eval quality-task defaults.
    """
    if pred_pq.shape != target_pq.shape or pred_pq.ndim != 3 or pred_pq.shape[0] != 3:
        raise ValueError("HDR-VDP-3 inputs must have matching (3, H, W) shapes")
    if task not in {"quality", "side-by-side"}:
        raise ValueError("HDR-VDP-3 task must be 'quality' or 'side-by-side'")
    kind, executable, toolbox = _resolve_backend()

    with tempfile.TemporaryDirectory(prefix="lumaflux-hdrvdp3-") as temp_dir:
        work = Path(temp_dir)
        for name, tensor in (("pred", pred_pq), ("ref", target_pq)):
            nits = (
                pq_eotf_nits(tensor.detach().to(dtype=torch.float32, device="cpu"))
                .permute(1, 2, 0)
                .numpy()
            )
            np.save(work / f"{name}.npy", nits)

        script = work / "run_vdp.m"
        output = work / "q.txt"
        display_options = (
            f", {{'rgb_display','{_matlab_quote(display_model)}'}}"
            if display_model is not None
            else ""
        )
        script.write_text(
            "\n".join(
                [
                    f"addpath(genpath('{_matlab_quote(toolbox)}'));",
                    f"pred = double(readNPY('{_matlab_quote(work / 'pred.npy')}'));",
                    f"ref = double(readNPY('{_matlab_quote(work / 'ref.npy')}'));",
                    f"res = hdrvdp3('{task}', pred, ref, 'rgb-bt.2020', "
                    f"{pixels_per_degree:.12g}{display_options});",
                    f"fid = fopen('{_matlab_quote(output)}', 'w');",
                    "if fid == -1; error('LumaFlux:Output', 'Cannot open result file'); end;",
                    "fprintf(fid, '%.17g', res.Q);",
                    "fclose(fid);",
                ]
            )
            + "\n"
        )
        if kind == "matlab":
            command = [executable, "-batch", f"run('{_matlab_quote(script)}')"]
        else:
            command = [executable, "--quiet", "--no-gui", str(script)]

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            detail = ""
            if isinstance(exc, subprocess.CalledProcessError):
                detail = (exc.stderr or exc.stdout or "").strip()[-1000:]
            reason = f"{kind} execution failed: {exc}"
            if detail:
                reason += f"; backend output: {detail}"
            raise EvalBackendUnavailable(_METRIC, reason) from exc

        if not output.is_file():
            detail = (result.stderr or result.stdout or "").strip()[-1000:]
            raise EvalBackendUnavailable(
                _METRIC,
                f"{kind} completed without a result file; backend output: {detail or '<empty>'}",
            )
        try:
            return float(output.read_text().strip())
        except ValueError as exc:
            raise EvalBackendUnavailable(_METRIC, f"invalid result: {output.read_text()!r}") from exc


def probe_hdrvdp3() -> None:
    """Execute a small end-to-end backend probe."""
    frame = torch.full((3, 128, 128), 0.5)
    hdr_vdp3(frame, frame)
