"""LumaFlux: Lifting 8-Bit Worlds to HDR Reality with Physically-Guided
Diffusion Transformers."""

__version__ = "0.1.0"

__all__ = ["LumaFluxModel", "LumaFluxPipeline", "build_model", "load_config"]


def __getattr__(name: str):
    """Load torch-backed public objects only when callers request them."""
    if name in {"LumaFluxModel", "build_model", "load_config"}:
        from .models import LumaFluxModel, build_model, load_config

        return {
            "LumaFluxModel": LumaFluxModel,
            "build_model": build_model,
            "load_config": load_config,
        }[name]
    if name == "LumaFluxPipeline":
        from .inference import LumaFluxPipeline

        return LumaFluxPipeline
    raise AttributeError(name)
