"""CPU tests for the ComfyUI nodes in comfyui/, run against the miniature model stack.

ComfyUI itself is not a test dependency: small stand-ins for folder_paths and
comfy.* are installed when the real modules are absent, and the repository root
is loaded the way ComfyUI loads a custom node folder.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = "lumaflux_node_pack"


def _stub(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


def _install_comfy_stubs() -> None:
    try:
        import comfy.model_management  # noqa: F401
        import comfy.utils  # noqa: F401
        import folder_paths  # noqa: F401
        return
    except ImportError:
        pass

    base = tempfile.mkdtemp(prefix="lumaflux-comfyui-test-")
    fp = _stub("folder_paths")
    fp.models_dir = os.path.join(base, "models")
    fp.folder_names_and_paths = {
        "diffusion_models": ([os.path.join(fp.models_dir, "diffusion_models")], {".safetensors"}),
        "vae": ([os.path.join(fp.models_dir, "vae")], {".safetensors"}),
    }
    output_dir = os.path.join(base, "output")

    def add_model_folder_path(name, path, is_default=False):
        fp.folder_names_and_paths.setdefault(name, ([], set()))[0].append(path)

    def get_filename_list(name):
        paths, exts = fp.folder_names_and_paths.get(name, ([], set()))
        files = []
        for path in paths:
            if os.path.isdir(path):
                files += [f for f in os.listdir(path) if not exts or os.path.splitext(f)[1] in exts]
        return sorted(files)

    def get_full_path(name, filename):
        for path in fp.folder_names_and_paths.get(name, ([], set()))[0]:
            candidate = os.path.join(path, filename)
            if os.path.isfile(candidate):
                return candidate
        return None

    def get_full_path_or_raise(name, filename):
        path = get_full_path(name, filename)
        if path is None:
            raise FileNotFoundError(filename)
        return path

    def get_output_directory():
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_save_image_path(prefix, out_dir, w=0, h=0):
        subfolder, filename = os.path.split(os.path.normpath(prefix))
        full = os.path.join(out_dir, subfolder)
        os.makedirs(full, exist_ok=True)
        existing = [f for f in os.listdir(full) if f.startswith(filename + "_")]
        return full, filename, len(existing) + 1, subfolder, prefix

    fp.add_model_folder_path = add_model_folder_path
    fp.get_filename_list = get_filename_list
    fp.get_full_path = get_full_path
    fp.get_full_path_or_raise = get_full_path_or_raise
    fp.get_output_directory = get_output_directory
    fp.get_save_image_path = get_save_image_path

    comfy = _stub("comfy")
    mm = _stub("comfy.model_management")
    mm.get_torch_device = lambda: torch.device("cpu")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda force=False: None
    mm.throw_exception_if_processing_interrupted = lambda: None
    utils = _stub("comfy.utils")

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total, self.current = total, 0

        def update(self, value):
            self.current += value

    utils.ProgressBar = ProgressBar
    comfy.model_management, comfy.utils = mm, utils
    sys.modules.update({"folder_paths": fp, "comfy": comfy, "comfy.model_management": mm, "comfy.utils": utils})


_install_comfy_stubs()


@pytest.fixture(scope="module")
def pack():
    """The repository root imported as ComfyUI imports a custom node folder."""
    if PACK in sys.modules:
        return sys.modules[PACK]
    spec = importlib.util.spec_from_file_location(
        PACK, os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def nodes(pack):
    return sys.modules[f"{PACK}.comfyui.nodes"]


@pytest.fixture(scope="module")
def hdr_io(pack):
    return sys.modules[f"{PACK}.comfyui.hdr_io"]


@pytest.fixture(scope="module")
def tiny_handle(nodes):
    from lumaflux.inference.pipeline import LumaFluxPipeline
    from lumaflux.models.factory import build_model

    torch.manual_seed(0)
    cfg = {
        "model": {
            "backbone": "tiny", "siglip": "tiny", "rank": 4, "num_knots": 8, "phys_channels": 8,
            "stats_dim": 8, "num_bands": 4, "num_null_tokens": 4, "modulation_hidden": 32,
        }
    }
    pipe = LumaFluxPipeline(build_model(cfg), cfg={"train": {"bridge_noise": 0.05}})
    return nodes.LumaFluxHandle(pipe, "tiny", "tiny", torch.float32)


def test_root_registers_five_nodes(pack):
    assert set(pack.NODE_CLASS_MAPPINGS) == set(pack.NODE_DISPLAY_NAME_MAPPINGS)
    assert len(pack.NODE_CLASS_MAPPINGS) == 5
    for cls in pack.NODE_CLASS_MAPPINGS.values():
        assert "required" in cls.INPUT_TYPES()
        assert hasattr(cls, cls.FUNCTION)


def test_convert_pads_crops_and_stays_in_range(nodes, tiny_handle):
    node = nodes.LumaFluxSDRToHDR()
    image = torch.rand(3, 50, 70, 3)  # deliberately not a multiple of 2 * vae_scale
    hdr, preview = node.convert(tiny_handle, image, steps=2, seed=1, shared_noise=True, rqs_ema=0.8, tone_strength=1.0)
    assert hdr.shape == image.shape and preview.shape == image.shape
    for t in (hdr, preview):
        assert torch.isfinite(t).all()
        assert t.min() >= 0.0 and t.max() <= 1.0


def test_convert_is_seed_deterministic(nodes, tiny_handle):
    node = nodes.LumaFluxSDRToHDR()
    image = torch.rand(2, 32, 32, 3)
    a, _ = node.convert(tiny_handle, image, steps=2, seed=7, shared_noise=True, rqs_ema=0.0, tone_strength=1.0)
    b, _ = node.convert(tiny_handle, image, steps=2, seed=7, shared_noise=True, rqs_ema=0.0, tone_strength=1.0)
    assert torch.equal(a, b)


def test_shared_noise_gives_identical_frames_for_identical_input(nodes, tiny_handle):
    node = nodes.LumaFluxSDRToHDR()
    image = torch.rand(1, 32, 32, 3).repeat(3, 1, 1, 1)
    shared, _ = node.convert(tiny_handle, image, steps=2, seed=3, shared_noise=True, rqs_ema=0.0, tone_strength=1.0)
    assert torch.equal(shared[0], shared[1]) and torch.equal(shared[1], shared[2])
    independent, _ = node.convert(tiny_handle, image, steps=2, seed=3, shared_noise=False, rqs_ema=0.0, tone_strength=1.0)
    assert not torch.equal(independent[0], independent[1])


def test_single_image_runs_with_ema_setting(nodes, tiny_handle):
    node = nodes.LumaFluxSDRToHDR()
    hdr, _ = node.convert(tiny_handle, torch.rand(1, 32, 48, 3), steps=1, seed=0, shared_noise=True, rqs_ema=0.8, tone_strength=0.5)
    assert hdr.shape == (1, 32, 48, 3)


def test_png16_roundtrip_keeps_code_values_and_tags_pq(nodes, hdr_io, tmp_path):
    hdr = torch.rand(2, 20, 24, 3)
    node = nodes.LumaFluxSaveHDRPNG()
    node.output_dir = str(tmp_path)
    entries = node.save(hdr, filename_prefix="lumaflux/test")["ui"]["images"]
    assert len(entries) == 2
    for i, entry in enumerate(entries):
        rgb, cicp = hdr_io.read_png16(os.path.join(str(tmp_path), entry["subfolder"], entry["filename"]))
        assert rgb.shape == (20, 24, 3)
        assert np.array_equal(rgb, (hdr[i] * 65535.0).round().numpy().astype(np.uint16))
        assert cicp == bytes([9, 16, 0, 1])


def test_preview_node_outputs_display_signal(nodes):
    node = nodes.LumaFluxHDRPreview()
    hdr = torch.rand(1, 16, 16, 3) * 0.75  # up to the 1000-nit code
    (preview,) = node.tonemap(hdr, white_nits=203.0, method="reinhard")
    assert preview.shape == hdr.shape
    assert preview.min() >= 0.0 and preview.max() <= 1.0
    (clipped,) = node.tonemap(hdr, white_nits=203.0, method="clip")
    assert clipped.shape == hdr.shape


def _ffmpeg_10bit() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-h", "encoder=libx265"], capture_output=True, text=True).stdout
    return "yuv420p10le" in out


@pytest.mark.skipif(not _ffmpeg_10bit(), reason="needs ffmpeg with a 10-bit libx265")
def test_video_node_writes_pq_tagged_hevc(nodes, tmp_path):
    node = nodes.LumaFluxSaveHDRVideo()
    node.output_dir = str(tmp_path)
    entry = node.save(torch.rand(3, 64, 64, 3) * 0.75, filename_prefix="lumaflux/clip", fps=24.0, crf=20)["ui"]["images"][0]
    path = os.path.join(str(tmp_path), entry["subfolder"], entry["filename"])
    assert os.path.getsize(path) > 0
    probe = subprocess.run(["ffprobe", "-v", "quiet", "-show_streams", path], capture_output=True, text=True).stdout
    assert "smpte2084" in probe and "bt2020" in probe and "yuv420p10le" in probe
