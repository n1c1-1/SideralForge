"""astro_GPU nodes for ComfyUI: the astrophotography processing chain, its variants, rendering and export.

Install: link (or copy) this folder into ComfyUI/custom_nodes/ and install the astro_gpu package in ComfyUI's Python
environment (pip install -e <this repository>). See docs/COMFYUI.md.
"""
from .nodes import comfy_entrypoint  # noqa: F401

WEB_DIRECTORY = "./web"          # the live render node's sliders and preview

try:                             # the routes exist only inside a running ComfyUI
    from . import server_routes  # noqa: F401
except (ImportError, AttributeError):   # no server module, or no server running (tests)
    pass
