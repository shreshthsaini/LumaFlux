"""LumaFlux as a ComfyUI custom node pack.

Cloning this repository into ComfyUI/custom_nodes registers the nodes defined in
comfyui/. Outside ComfyUI (no folder_paths module on the path) this file exports
empty mappings so importing the repository root is harmless.
"""

import importlib.util

if importlib.util.find_spec("folder_paths") is not None:
    from .comfyui.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
