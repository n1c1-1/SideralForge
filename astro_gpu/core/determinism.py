"""Run-to-run reproducibility: the same inputs give the same outputs.

On the GPU, sums that gather many contributions into one place (scatter-adds: drawing stars into an image,
accumulating gradients) are done in whatever order the threads finish, so the last bits of a result change from one
run to the next. Through a threshold or an iterative fit later on, those last bits could tip a decision and change a
result visibly. Inside ``deterministic()``, PyTorch uses its order-preserving versions of these operations, cuDNN its
deterministic algorithms, and cuBLAS a fixed workspace (``CUBLAS_WORKSPACE_CONFIG``, set when astro_gpu is
imported, before any GPU work).
"""
from __future__ import annotations

import contextlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402


@contextlib.contextmanager
def deterministic(strict: bool = False):
    """Deterministic GPU algorithms inside the block; the previous settings restored after. ``strict``: an operation
    without a deterministic version raises an error instead of a warning (for code written to be fully reproducible)."""
    prev = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.benchmark = prev[2]
        torch.backends.cudnn.deterministic = prev[3]
