"""Shared test configuration."""
from __future__ import annotations

import os

import pytest

# the ComfyUI nodes pick a CUDA device when imported: without a visible GPU (CUDA_VISIBLE_DEVICES="", as during the
# projects' runs) their test files cannot even be collected
collect_ignore = (["test_comfy_masks.py", "test_comfy_project_node.py"]
                  if os.environ.get("CUDA_VISIBLE_DEVICES") == "" else [])


@pytest.fixture(autouse=True)
def _private_project_registry(tmp_path, monkeypatch):
    """Projects created by the tests must not appear in the user's list of recent projects."""
    import astro_gpu.core.project as project
    monkeypatch.setattr(project, "REGISTRY", tmp_path / "projects.json")
