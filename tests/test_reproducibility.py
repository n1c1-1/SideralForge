"""Run-to-run reproducibility: deterministic mode."""
from __future__ import annotations

import torch

from astro_gpu.core.determinism import deterministic


def test_deterministic_mode_is_restored():
    before = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    with deterministic():
        assert torch.are_deterministic_algorithms_enabled()
    assert (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark) == before
