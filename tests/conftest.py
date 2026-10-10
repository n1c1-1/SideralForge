"""Shared test configuration."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _private_project_registry(tmp_path, monkeypatch):
    """Projects created by the tests must not appear in the user's list of recent projects."""
    import astro_gpu.core.project as project
    monkeypatch.setattr(project, "REGISTRY", tmp_path / "projects.json")
