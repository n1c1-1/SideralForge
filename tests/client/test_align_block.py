"""The Align images block: one block across its images' lanes, its reference, removed at once; Post's button."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox  # noqa: E402

from astro_gpu import api  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import prefs  # noqa: E402
from astro_gpu.client.blocks import Graph  # noqa: E402
from astro_gpu.client.graph import AlignItem  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.core import project2  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "RTX", 16.0)], 16, 64.0)


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return app


@pytest.fixture()
def world(tmp_path, monkeypatch, qapp):
    import cv2
    from astropy.io import fits
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    prefs.set("home_sections", [])
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: None)
    png, fit, fit2 = tmp_path / "a.png", tmp_path / "b.fits", tmp_path / "c.fits"
    cv2.imwrite(str(png), np.full((8, 8, 3), 128, np.uint8))
    fits.PrimaryHDU(np.full((3, 8, 8), 0.1, np.float32)).writeto(fit)
    fits.PrimaryHDU(np.full((3, 8, 8), 0.2, np.float32)).writeto(fit2)
    w = appmod.Main()
    w.resize(1500, 800)
    w.show()
    w.worker.wait()
    w.new_project()
    w.worker.wait()
    settle(w)
    w.graph.place_images([str(png), str(fit), str(fit2)])
    settle(w)
    yield w
    w.close()
    w.deleteLater()
    QApplication.processEvents()


def settle(w, n=12):
    for _i in range(n):
        QApplication.processEvents()
        w.worker.wait(30)
    QApplication.processEvents()


def item(w) -> AlignItem:
    return next(x for x in w.graph.graph.scene().items() if isinstance(x, AlignItem))


def test_one_block_across_the_lanes_and_a_wire_each(world):
    w = world
    assert w.graph.align_images()
    settle(w)
    s = w.s
    assert w.s.sel_step == "align_images"
    assert all(s.display_keys(q)[:2] == ["image", "align_images"] for q in s.paths)
    it = item(w)
    g = w.graph.graph.graph
    lanes = [g.node_of(q, "align_images") for q in s.paths]
    assert len({id(n) for n in lanes}) == 3 and len(it.nodes) == 3 and it.selected
    assert it.rect.top() <= min(n.y for n in lanes) and it.rect.bottom() >= max(n.y for n in lanes)
    assert len({n.col for n in lanes}) == 1


def test_the_first_image_is_the_reference_by_default_and_it_can_change(world):
    w = world
    w.graph.align_images()
    settle(w)
    s = w.s
    first = s.paths[0].pid
    assert {q.source["align"] for q in s.paths} == {first}
    assert s.p.workflow(s.paths[1].pid).source["path"].endswith("b.fits")
    cb = next(c for c in w.graph.panel.findChildren(QComboBox) if c.itemData(0) == first)
    cb.setCurrentIndex(2)
    cb.activated.emit(2)
    settle(w)
    assert {q.source["align"] for q in s.paths} == {s.paths[2].pid}


def test_the_reference_is_untouched_and_the_others_change_key_and_read_its_file(world):
    w = world
    s = w.s
    before = [api.stage_key(s.p, q.pid, "image") for q in s.paths]
    w.graph.align_images()
    settle(w)
    after = [api.stage_key(s.p, q.pid, "image") for q in s.paths]
    assert after[0] == before[0] and after[1] != before[1] and after[2] != before[2]
    from astro_gpu.core.store import Keys
    keys = Keys(s.p) if callable(Keys) else None
    if keys is not None:
        v = keys.overrides(s.paths[1].pid, "image")
        assert v["align_to"].endswith("a.png")
        assert keys.overrides(s.paths[0].pid, "image")["align_to"] == ""


def test_the_cross_removes_the_whole_block_and_ctrl_z_brings_it_back(world):
    w = world
    w.graph.align_images()
    settle(w)
    s = w.s
    n = item(w).nodes[0]
    w.graph.remove_tool(n, None)
    settle(w)
    assert all("align" not in q.source for q in s.paths)
    assert not [x for x in w.graph.graph.scene().items() if isinstance(x, AlignItem)]
    assert s.undo()
    settle(w)
    assert all("align" in q.source for q in s.paths)


def test_a_workflow_deleted_leaves_a_valid_block_or_none(world):
    w = world
    w.graph.align_images()
    settle(w)
    s = w.s
    s.remove_path(s.paths.index(s.paths[0]))                      # the reference goes
    settle(w)
    assert all("align" not in (q.source or {}) for q in s.paths)
    project2.Project2.from_dict(s.p.to_dict(), None).validate()


def test_post_button_is_greyed_with_its_reason_under_two_images_and_opens_inspect(world):
    w = world
    w.set_mode(2)
    settle(w)
    view = w.post_holder.page.view
    assert view.b_align.isEnabled()
    view.alignAsked.emit()
    settle(w)
    assert w.mode == 1 and w.s.sel_step == "align_images" and all("align" in q.source for q in w.s.paths)
    w.s.remove_path(1)
    w.s.remove_path(1)
    settle(w)
    w.set_mode(2)
    settle(w)
    view.rebuild()
    assert not view.b_align.isEnabled() and "needs 2 images" in view.b_align.toolTip()


def test_the_layers_stay_the_same_workflows_so_their_settings_stay(world):
    w = world
    pids = [q.pid for q in w.s.paths]
    w.graph.align_images()
    settle(w)
    assert [q.pid for q in w.s.paths] == pids
    g = Graph(w.s)
    assert set(pids) <= set(g.to_post)


def test_the_palette_card_is_greyed_without_two_images(world):
    w = world
    assert not w.graph.cards["align_images"].reason
    w.graph.cards["align_images"].mouseDoubleClickEvent(None)
    settle(w)
    assert all("align" in q.source for q in w.s.paths)
    w.s.remove_path(1)
    w.s.remove_path(1)
    settle(w)
    w.graph.build_tools()
    assert w.graph.cards["align_images"].reason == "needs 2 images"


def test_a_real_click_selects_the_block_and_its_cross_removes_it(world):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest
    w = world
    w.show()
    w.graph.align_images()
    settle(w)
    gv = w.graph.graph
    QTest.mouseClick(gv.viewport(), Qt.LeftButton, pos=gv.mapFromScene(item(w).sceneBoundingRect().center()))
    settle(w)
    assert item(w).selected
    x = item(w).mapToScene(item(w)._x_rect().center())
    QTest.mouseClick(gv.viewport(), Qt.LeftButton, pos=gv.mapFromScene(QPointF(x)))
    settle(w)
    assert not [i for i in gv.scene().items() if isinstance(i, AlignItem)]
