import random

import numpy as np
import torch

from .io import (
    save_sdr_png,
    load_sdr_png,
    save_hdr_png16,
    load_hdr_png16,
    extract_video_frames,
    ffmpeg_available,
)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
