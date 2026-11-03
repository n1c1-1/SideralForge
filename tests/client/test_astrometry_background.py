"""Inspect > Astrometry: the user picks the picture behind the solution among the workflow's computed images.

The choices (the stack, the colour-calibrated stack, the gradient removed, the final image...) are only the images the
workflow has computed and up to date; no Compare with list on that tab. The list is checked on a model of stages (no
project); the tab on a computed bench project, opened read-only, when this machine has one.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.steps import Steps  # noqa: E402
from astro_gpu.client.widgets import QSS  # noqa: E402

BENCH = Path("F:/astro_essais/bench/iris/v2x_express")


class FakeModel:
    """The stages of one workflow and their states (what ``Steps`` asks of a model)."""

    def __init__(self, stages: dict[str, str]):
        self.stages = stages                                   # stage -> "done" | "stale" | "pending"

    def all_stages(self, p):
        return list(self.stages)

    def state_info(self, p, stage):
        st = self.stages.get(stage)
        if st is None:
            return None
        return SimpleNamespace(state=st, key=f"k-{stage}", previous_key=f"old-{stage}" if st == "stale" else "")


def steps_of(stages: dict[str, str]) -> Steps:
    s = Steps.__new__(Steps)                                   # no service: only the stages are asked
    s.s = FakeModel(stages)
    return s


P = SimpleNamespace(pid=1, name="WF")
BASE = {"register": "done", "stack": "done", "astrometry": "done", "surveys": "done"}


def keys(stages):
    return [b.key for b in steps_of(stages).backgrounds(P)]


def test_the_backgrounds_are_the_computed_images_in_the_order_of_the_workflow():
    assert keys(BASE) == ["stack"]
    assert keys({**BASE, "color": "done", "background_poly": "done", "render_simple": "done"}) == \
        ["stack", "color", "result", "output"]
    full = {**BASE, "color": "done", "denoise": "done", "background": "done", "starless": "done", "optics": "done",
            "render_prep": "done"}
    assert keys(full) == ["stack", "color", "denoised", "color_denoised", "result", "output"]
    captions = [b.caption for b in steps_of(full).backgrounds(P)]
    assert captions == ["Stack", "Colour-calibrated stack", "Denoised stack", "Colour-calibrated denoised stack",
                        "Gradient removed", "Final image"]


def test_an_image_not_computed_is_not_listed_an_outdated_one_is_with_its_previous_result():
    assert keys({**BASE, "color": "pending", "background_poly": "done", "render_simple": "done"}) == ["stack", "output"]
    # an Outdated result is listed (its previous one, marked), as the rest of Inspect shows it (M31, 2 November: every
    # stage Outdated after an engine update left the list empty)
    assert keys({**BASE, "color": "stale", "background_poly": "done", "render_simple": "done"}) ==         ["stack", "color", "result", "output"]
    bgs = steps_of({**BASE, "stack": "stale"}).backgrounds(P)
    assert [b.key for b in bgs] == ["stack"] and bgs[0].previous


def test_the_solution_tab_has_a_background_choice_and_nothing_to_compare_with():
    steps = steps_of({**BASE, "color": "done"})
    vis = steps.visuals(P, "astrometrie")
    sol, survey = vis[0], vis[1]
    assert sol.key == "solution" and sol.sky and sol.backgrounds and sol.compare == ()
    assert survey.key == "survey" and survey.compare == ("stack",) and not survey.backgrounds
    assert "compare" not in sol.note.lower()
    assert steps.visuals(P, "empilement")[0].compare and not steps.visuals(P, "empilement")[0].backgrounds


# ---------------------------------------------------------------------------------------------- the tab itself
def _fingerprint(root: Path):
    return sorted((f.relative_to(root).as_posix(), f.stat().st_size, f.stat().st_mtime_ns)
                  for f in root.rglob("*") if f.is_file() and "store" not in f.relative_to(root).parts[:1])


@pytest.fixture(scope="module")
def window():
    if not (BENCH / "project.astroproj").is_file():
        pytest.skip("the iris bench project is not on this machine")
    import tempfile
    before = _fingerprint(BENCH)
    tmp = Path(tempfile.mkdtemp(prefix="astro_bg_"))
    old = os.environ.get("ASTRO_GPU_SETTINGS")
    os.environ["ASTRO_GPU_SETTINGS"] = str(tmp / "settings.json")
    prefs.load()
    qa = QApplication.instance() or QApplication([])
    qa.setStyleSheet(QSS + appmod.EXTRA_QSS)
    appmod.apply_language("en")
    w = appmod.Main()
    w.resize(1680, 1000)
    w.show()
    w.readonly = True
    w.open_project(BENCH)
    end = time.time() + 240
    while time.time() < end and not (w.s is not None and w.s.ready and not w.worker.busy()):
        qa.processEvents()
        time.sleep(0.02)
    yield w
    w.close()
    assert _fingerprint(BENCH) == before, "the bench project was touched"
    if old is None:
        os.environ.pop("ASTRO_GPU_SETTINGS", None)
    else:
        os.environ["ASTRO_GPU_SETTINGS"] = old


def settle(w, timeout=240.0):
    qa = QApplication.instance()
    end = time.time() + timeout
    quiet = 0
    while time.time() < end and quiet < 6:
        qa.processEvents()
        time.sleep(0.05)
        views = list(w.tune.views)
        busy = w.worker.busy() or w.tune.service.busy() or any(
            v._want is not None and v.prepared() is None and v.canvas.message in ("", None) for v in views)
        loading = any((v.canvas.message or "").endswith("…") for v in views)
        quiet = 0 if busy or loading else quiet + 1


def test_real_astrometry_tab_lists_the_images_and_remembers_the_choice(window):
    w = window
    pid = next(q.pid for q in w.s.paths if q.name == "PI-own")
    t = w.tune
    p = w.s.view(pid)
    w.select(index=w.s.index_of(pid), step="astrometrie")
    w.set_mode(1)
    t.bg_of.clear()
    t.tab_of.clear()
    t.refresh()
    settle(w)
    cb = next(c for c in t.findChildren(QComboBox) if c.isVisibleTo(t) and c.findData("stack") >= 0)
    assert [cb.itemData(i) for i in range(cb.count())] == [b.key for b in t.steps.backgrounds(p)]
    assert [cb.itemText(i) for i in range(cb.count())][:2] == ["Stack", "Colour-calibrated stack"]
    assert cb.currentData() == "stack"                           # the default
    v = t.view
    assert not v.cmp_box.isVisibleTo(v) and not v._targets      # nothing to compare with, nothing from the bench either
    # a choice is remembered when the tab is built again, and falls back on the stack when it is gone
    cb.setCurrentIndex(cb.findData("color"))
    cb.activated.emit(cb.currentIndex())
    settle(w)
    assert t.bg_of["astrometrie"] == "color"
    t.refresh()
    settle(w)
    cb = next(c for c in t.findChildren(QComboBox) if c.isVisibleTo(t) and c.findData("stack") >= 0)
    assert cb.currentData() == "color"
    t.bg_of["astrometrie"] = "denoised"                          # not in this workflow
    t.refresh()
    settle(w)
    cb = next(c for c in t.findChildren(QComboBox) if c.isVisibleTo(t) and c.findData("stack") >= 0)
    assert cb.currentData() == "stack"
    t.bg_of.clear()
    t.tab_of.clear()
