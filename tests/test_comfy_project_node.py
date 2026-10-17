"""The ComfyUI 'Astro: New project' node: guesses from the lights folder, never overwrites an existing project.
Needs ComfyUI's code (COMFYUI_ROOT, default ../ComfyUI next to this repository); skipped without it."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "ComfyUI"))
if (ROOT / "comfy_api").is_dir() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
pytest.importorskip("comfy_api")
node = pytest.importorskip("comfyui_astro.project_node")


@pytest.fixture
def session(tmp_path, monkeypatch):
    import astro_gpu.core.project as project
    monkeypatch.setattr(project, "REGISTRY", tmp_path / "recent.json")    # not the user's recent projects
    s = tmp_path / "NGC7822_12.10.2024"
    for sub, n in (("lights", 3), ("dark", 2), ("flat", 2)):
        (s / sub).mkdir(parents=True)
        for i in range(n):
            (s / sub / f"f{i}.fits").write_bytes(b"")
    return s


def _run(**kw):
    out = node.AstroNewProject.execute(**kw)
    return out.result if hasattr(out, "result") else out.args


def test_guesses_and_creates(session):
    astro, status = _run(lights=str(session / "lights"))
    cfg = json.loads((session / "astro_gpu" / "project.json").read_text(encoding="utf-8"))
    assert Path(astro["root"]) == session / "astro_gpu"
    assert cfg["name"] == "NGC7822_12.10.2024" and cfg["target"]["name"] == "NGC7822"
    assert [Path(f).name for f in cfg["inputs"]["darks"]] == ["dark"]
    assert [Path(f).name for f in cfg["inputs"]["flats"]] == ["flat"]
    assert "3 lights" in status and "guessed next to the lights: darks, flats" in status


def test_existing_project_is_opened_not_changed(session):
    _run(lights=str(session / "lights"), name="first")
    f = session / "astro_gpu" / "project.json"
    before = f.read_text(encoding="utf-8")
    astro, status = _run(lights=str(session / "lights"), name="second", target="M 31")
    assert f.read_text(encoding="utf-8") == before
    assert "existing project opened" in status


def test_no_guess_and_explicit_lists(session):
    _run(lights=str(session / "lights"), guess_calibration=False, flats=str(session / "flat"),
         project_folder=str(session / "proj"), ra="1.5", dec="67.2")
    cfg = json.loads((session / "proj" / "project.json").read_text(encoding="utf-8"))
    assert cfg["inputs"]["darks"] == [] and [Path(f).name for f in cfg["inputs"]["flats"]] == ["flat"]
    assert cfg["target"]["ra"] == 1.5 and cfg["target"]["dec"] == 67.2


def test_lights_required_and_must_exist(session):
    with pytest.raises(ValueError):
        _run(lights="")
    with pytest.raises(FileNotFoundError):
        _run(lights=str(session / "nowhere"))
