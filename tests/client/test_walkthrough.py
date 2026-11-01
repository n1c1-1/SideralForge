"""The scripted tour of the client (tools/client_walkthrough.py), in its short version: a project created from scratch on
the heart_a7s raw frames, its frames read, its workflows edited, every screen and dialog shown in English then in
French, closed and opened again, Undo and Redo, the check of a project of the previous version. No run, no export, no
import copy (the full tour does those).

Each screenshot is checked by the tour itself: no text cut short, no English left in French; and the tour fails on any
Qt warning, Python warning or exception. It runs in its own process (offscreen, CPU only, its own settings file and
engine state folder), in a temporary workspace.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "client_walkthrough.py"
DATA = Path("F:/astro_essais/v2/heart_a7s/project.astroproj")

pytestmark = pytest.mark.skipif(not DATA.is_file(), reason="the heart_a7s test data (F:/astro_essais) is not here")


def test_the_short_walkthrough_shows_every_screen_in_both_languages_without_a_problem(tmp_path):
    out = tmp_path / "shots"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    env.pop("ASTRO_GPU_SETTINGS", None)                      # the tour makes its own
    r = subprocess.run([sys.executable, str(TOOL), "--short", "--workspace", str(tmp_path / "ws"), "--out", str(out)],
                       cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=900)
    tail = (r.stdout[-6000:] + "\n" + r.stderr[-3000:])
    rep = json.loads((out / "report.json").read_text(encoding="utf-8")) if (out / "report.json").exists() else None
    assert rep is not None, tail
    assert rep["problems"] == [], tail
    assert rep["messages"] == [], tail
    assert rep["untranslated_asked"] == [], tail
    assert r.returncode == 0, tail
    names = sorted(p.name for p in out.glob("*.png"))
    for screen in ("home", "new_project", "import_folder", "frames_sessions", "frames_browse", "new_workflow",
                   "graph_step_added", "graph_show_all", "run_all_plan", "inspect_pending", "reopened", "undo",
                   "redo", "import_previous", "settings", "about", "log"):
        for lang in ("en", "fr"):
            if screen in ("import_folder", "graph_step_added", "reopened", "undo", "redo") and lang == "fr":
                continue                                           # made once, in English
            if screen in ("about", "log") and lang == "en":
                continue                                           # shown in the French tour
            assert any(n.endswith(f"_{screen}_{lang}.png") for n in names), (screen, lang, names)
    assert rep["shots"] >= 40
