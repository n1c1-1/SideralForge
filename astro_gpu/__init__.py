"""astro_gpu: GPU astrophotography processing."""

import os as _os

_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # reproducible GPU sums (core/determinism.py)

__version__ = "0.1.0"
