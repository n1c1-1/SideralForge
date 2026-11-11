"""Run-to-run reproducibility: the same inputs give the same outputs.

On the GPU, sums that gather many contributions into one place (scatter-adds: drawing stars into an image,
accumulating gradients) are done in whatever order the threads finish, so the last bits of a result change from one
run to the next. Through a threshold or an iterative fit later on, those last bits could tip a decision and change a
result visibly. Inside ``deterministic()``, PyTorch uses its order-preserving versions of these operations, cuDNN its
deterministic algorithms, and cuBLAS a fixed workspace (``CUBLAS_WORKSPACE_CONFIG``, set when astro_gpu is
imported, before any GPU work).

PyTorch is imported by the functions, not by this module: the runner imports it (core/runner.py), and so did the
workers of the Calibration and Registration pools, which import the runner's module again when they start (a
spawned process imports its parent's main module) and never use PyTorch: about 700 MB reserved each for nothing
(issue #125; measured on 16 threads, pools of 12 through the runner: 956 MB a worker, 242 MB without PyTorch).
"""
from __future__ import annotations

import contextlib
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@contextlib.contextmanager
def deterministic(strict: bool = False):
    """Deterministic GPU algorithms inside the block; the previous settings restored after. ``strict``: an operation
    without a deterministic version raises an error instead of a warning (for code written to be fully reproducible)."""
    import torch
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


def warm_up() -> None:
    """Load what the first ``deterministic()`` block loads: in this PyTorch, ``use_deterministic_algorithms`` imports
    the compiler's configuration (torch._inductor, torch._dynamo), about a second once. The client calls it in a
    background thread at start, so that its first render does not wait for it."""
    import torch
    torch.use_deterministic_algorithms(torch.are_deterministic_algorithms_enabled(),
                                       warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
