"""Issue #44: the GUI on small laptop screens, and the graph's zoom and navigation (real gestures on the real window).

The size ranges and their dead band, large screens unchanged, the right panel as icons (over the graph, Esc or a click
elsewhere closes it, the pin docks it), Tab / Shift+Tab / Ctrl+Space with the View menu, the solo sections, the setting,
Reset layout, the dialogs' size; then the graph as the Post image: wheel at the pointer, keys, presets, pan, minimap,
kept through rebuilds and with the project, a selection scrolled into view, the titles only below the readable size,
the touchpad; and the same in Post's viewer.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QNativeGestureEvent, QPointingDevice, QShortcut, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QWidget,
)

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import layout, prefs  # noqa: E402
from astro_gpu.client.graph import MIN_SCALE, PALETTE_W, BlockItem, GhostItem, WireItem  # noqa: E402
from astro_gpu.client.layout import size_class  # noqa: E402
from astro_gpu.client.services.worker import Gpu, Machine  # noqa: E402
from astro_gpu.client.viewer import StaticSource, Viewer  # noqa: E402
from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.workflow import TEMPLATES  # noqa: E402

MACHINE = Machine([Gpu("cuda:0", "RTX", 16.0)], 16, 64.0)


@pytest.fixture(scope="session")
def qapp():
    from astro_gpu.client.widgets import QSS
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(QSS + appmod.EXTRA_QSS)
    return app


@pytest.fixture()
def env(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("ASTRO_GPU_SETTINGS", str(tmp_path / "settings.json"))
    monkeypatch.setenv("ASTRO_GPU_HOME", str(tmp_path / "home"))
    prefs.load()
    prefs.set("home_sections", [])
    prefs.set("cache_dir", (tmp_path / "scratch").as_posix())
    appmod.apply_language("en")
    monkeypatch.setattr(appmod, "machine_info", lambda: MACHINE)
    for kind in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, kind, lambda *a, **k: None)
    windows = []
    yield {"tmp": tmp_path, "windows": windows}
    for w in windows:
        w.close()
        w.deleteLater()
    QApplication.processEvents()
    appmod.apply_language("en")


def _project(tmp: Path, name: str = "Long") -> Path:
    """A project whose chain is long: one session, the Full template."""
    root = tmp / "ws" / name
    p = project2.create(root, name, Target(name="M 31"))
    p.sessions.append(project2.Session(1, "night", lights=["F:/raw/none"]))
    p.add_workflow("Full", TEMPLATES["Full"])
    project2.save(p)
    return root


def _settle(w, n=6):
    for _ in range(n):
        QApplication.processEvents()
        w.worker.wait(30)
    QApplication.processEvents()


def _open(env, size=(1366, 728), root=None):
    w = appmod.Main()
    w.resize(*size)
    w.show()
    env["windows"].append(w)
    w.worker.wait()
    w.open_project(root or _project(env["tmp"]))
    w.worker.wait()
    _settle(w, 10)
    return w


def _vp(w):
    return w.graph.graph.viewport()


def _activate(w):
    """Offscreen the window is not active by itself: its shortcuts need it to be."""
    w.activateWindow()
    QApplication.setActiveWindow(w) if hasattr(QApplication, "setActiveWindow") else None
    _settle(w, 2)


def _key(w, k, mod=Qt.NoModifier, widget=None):
    target = widget or _vp(w)
    _activate(w)
    target.setFocus()
    QTest.keyClick(target, k, mod)
    _settle(w)


def _wheel(widget, pos: QPoint, dy=120, dx=0, mods=Qt.NoModifier, phase=Qt.NoScrollPhase, pixel=(0, 0)):
    e = QWheelEvent(QPointF(pos), QPointF(widget.mapToGlobal(pos)), QPoint(*pixel), QPoint(dx, dy), Qt.NoButton, mods,
                    phase, False)
    QApplication.sendEvent(widget, e)
    QApplication.processEvents()


def _press(widget, pos: QPoint, button=Qt.LeftButton):
    QApplication.sendEvent(widget, QMouseEvent(QEvent.MouseButtonPress, QPointF(pos), QPointF(widget.mapToGlobal(pos)),
                                               button, button, Qt.NoModifier))


def _move(widget, pos: QPoint, buttons=Qt.LeftButton):
    QApplication.sendEvent(widget, QMouseEvent(QEvent.MouseMove, QPointF(pos), QPointF(widget.mapToGlobal(pos)),
                                               Qt.NoButton, buttons, Qt.NoModifier))


def _release(widget, pos: QPoint, button=Qt.LeftButton):
    QApplication.sendEvent(widget, QMouseEvent(QEvent.MouseButtonRelease, QPointF(pos), QPointF(widget.mapToGlobal(pos)),
                                               button, Qt.NoButton, Qt.NoModifier))


def _drag(widget, a: QPoint, b: QPoint, button=Qt.LeftButton):
    _press(widget, a, button)
    _move(widget, QPoint((a.x() + b.x()) // 2, (a.y() + b.y()) // 2), button)
    _move(widget, b, button)
    _release(widget, b, button)
    QApplication.processEvents()


def _icon(page, i=0):
    """An icon of the strip, as it is now (the strip is built again with the panel)."""
    return page.dock.icons.itemAt(i).widget()


def _blocks(w):
    return [it for it in w.graph.graph.scene().items() if isinstance(it, BlockItem)]


def _block(w, key):
    return next(it for it in _blocks(w) if it.node.key == key)


def _point_of(w, it) -> QPoint:
    return w.graph.graph.mapFromScene(it.sceneBoundingRect().center())


def _empty_point(w) -> QPoint:
    """A point of the view with no block, wire or place under it (bands are empty area)."""
    gv = w.graph.graph
    vp = gv.viewport().rect()
    for y in range(20, vp.height() - 20, 12):
        for x in range(10, vp.width() - 10, 12):
            if not isinstance(gv.itemAt(QPoint(x, y)), (BlockItem, WireItem, GhostItem)) \
                    and not gv.b_zoom.geometry().contains(x, y) and not gv.map.geometry().contains(x, y):
                return QPoint(x, y)
    raise AssertionError("no empty point")


# ---------------------------------------------------------------------------------------------- the size ranges
def test_size_ranges_and_their_dead_band():
    assert size_class(1920, 1080) == "large" and size_class(1600, 880) == "large"
    assert size_class(1536, 864) == "medium" and size_class(1280, 800) == "medium"
    assert size_class(1366, 728) == "small" and size_class(1280, 720) == "small" and size_class(1000, 900) == "small"
    # dragging the edge: the range is kept inside the dead band, on both sides of a threshold
    assert size_class(1590, 900, "large") == "large" and size_class(1570, 900, "large") == "medium"
    assert size_class(1610, 900, "medium") == "medium" and size_class(1625, 905, "medium") == "large"
    assert size_class(1290, 800, "small") == "small" and size_class(1310, 800, "small") == "medium"
    assert size_class(1270, 800, "medium") == "medium" and size_class(1250, 800, "medium") == "small"


def test_a_large_screen_keeps_todays_layout(env):
    w = _open(env, (1920, 1080))
    page = w.graph
    assert w.size_range == "large" and not w.compact and w.styleSheet() == ""
    assert page.palette_bar.width() == PALETTE_W and page.panel.width() == 390 and page.panel.isVisible()
    assert page.dock.state == "docked" and not page.dock.strip.isVisible() and page.left_state == "docked"
    assert w.proj_bar.height() == 46 and w.proj_bar.isVisible() and w.statusBar().isVisible()
    assert w.minimumWidth() == 1024 and w.minimumHeight() == 640


def test_a_laptop_window_gets_the_compact_look_and_the_folded_panels(env):
    w = _open(env, (1366, 728))
    page = w.graph
    assert w.size_range == "small" and w.compact and w.styleSheet() == layout.COMPACT_QSS
    assert page.dock.state == "icons" and page.dock.strip.isVisible() and page.dock.strip.width() == layout.STRIP_W
    assert not page.panel.isVisible() and page.left_state == "folded" and page.palette_bar.width() < PALETTE_W
    assert w.proj_bar.height() == 36
    assert page.graph.width() > 1200                                 # the room goes to the graph
    # a 1920 x 1080 screen at 125 %: compact, the panels docked
    w2 = _open(env, (1536, 864), root=_project(env["tmp"], "Mid"))
    assert w2.size_range == "medium" and w2.compact and w2.graph.dock.state == "docked"
    assert w2.proj_bar.height() == 36


def test_the_icons_open_the_panel_over_the_graph_and_esc_or_a_click_elsewhere_close_it(env):
    w = _open(env, (1366, 728))
    page = w.graph
    assert page.dock.icons.count() >= 2 and all(page.dock.icons.itemAt(i).widget().toolTip()
                                                for i in range(page.dock.icons.count()))   # the panel, then each section
    QTest.mouseClick(_icon(page), Qt.LeftButton)
    _settle(w)
    assert page.dock.is_open() and page.panel.isVisible()
    r = page.panel.geometry()
    assert r.right() < page.dock.strip.x() + 2 and page.graph.geometry().contains(r.center())   # over the main area
    assert page.panel.testAttribute(Qt.WA_StyledBackground)        # not see-through
    # a click inside keeps it, a click on the graph closes it
    QTest.mouseClick(page.panel, Qt.LeftButton, Qt.NoModifier, QPoint(50, 50))
    _settle(w)
    assert page.dock.is_open()
    QTest.mouseClick(_vp(w), Qt.LeftButton, Qt.NoModifier, _empty_point(w))
    _settle(w)
    assert not page.dock.is_open() and not page.panel.isVisible()
    # Esc closes it (before the graph's own Esc, which would deselect)
    QTest.mouseClick(_icon(page), Qt.LeftButton)
    _settle(w)
    assert page.dock.is_open()
    _key(w, Qt.Key_Escape)
    assert not page.dock.is_open()
    # the section icon opens that section; the pin docks the panel, and that is remembered for this size range
    QTest.mouseClick(page.dock.pin, Qt.LeftButton)
    _settle(w)
    assert page.dock.state == "docked" and page.panel.isVisible() and layout.choices("small")["right"] == "docked"
    # the panel's » folds it again
    QTest.mouseClick(page.panel.corner, Qt.LeftButton)
    _settle(w)
    assert page.dock.state == "icons" and layout.choices("small")["right"] == "icons"


def test_tab_shift_tab_and_ctrl_space_with_their_menu_marks(env):
    w = _open(env, (1920, 1080))
    page = w.graph
    assert not w.a_hide.isChecked()
    _key(w, Qt.Key_Tab)
    assert page.dock.state == "hidden" and page.left_state == "hidden" and w.a_hide.isChecked()
    assert page.dock.arrow.isVisible() and page.left_arrow.isVisible()       # a small arrow brings them back
    _key(w, Qt.Key_Tab)
    assert page.dock.state == "docked" and page.left_state == "docked" and not w.a_hide.isChecked()
    # Shift+Tab: only the right panel
    _key(w, Qt.Key_Tab, Qt.ShiftModifier)
    assert page.dock.state == "hidden" and page.left_state == "docked" and w.a_hide_r.isChecked()
    QTest.mouseClick(page.dock.arrow, Qt.LeftButton)
    _settle(w)
    assert page.dock.state == "docked" and not w.a_hide_r.isChecked()
    # Ctrl+Space: the graph alone; remembered for this size range
    _key(w, Qt.Key_Space, Qt.ControlModifier)
    assert not w.proj_bar.isVisible() and not w.statusBar().isVisible() and page.dock.state == "hidden" \
        and page.left_state == "hidden" and w.a_max.isChecked()
    assert prefs.get("layout")["large"]["max"] is True
    w.a_max.trigger()
    _settle(w)
    assert w.proj_bar.isVisible() and page.left_state == "docked"
    # the Reset layout command forgets every choice
    _key(w, Qt.Key_Tab)
    w.reset_layout()
    _settle(w)
    assert prefs.get("layout") == {} and page.left_state == "docked" and page.dock.state == "docked"


def test_tab_in_a_text_field_moves_the_focus_as_in_a_form(env):
    w = _open(env, (1920, 1080))
    ed = next(x for x in w.graph.panel.findChildren(QLineEdit) if x.isVisible())
    _activate(w)
    ed.setFocus()
    QTest.keyClick(ed, Qt.Key_Tab)
    _settle(w)
    assert w.graph.dock.state == "docked" and not w.a_hide.isChecked()   # the panels did not move


def test_ctrl_1_is_100_percent_in_the_graph_and_the_mode_switch_elsewhere(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    w.set_mode(0)
    _settle(w)
    gv.zoom_to(0.7)
    _key(w, Qt.Key_1, Qt.ControlModifier)
    assert w.mode == 0 and abs(gv.zoom() - 1.0) < 1e-6                 # not the window's "Graph": the readable size
    w.set_mode(2)
    _settle(w)
    _activate(w)
    w.mode_btns[0].setFocus()
    QTest.keyClick(w.mode_btns[0], Qt.Key_1, Qt.ControlModifier)
    _settle(w)
    assert w.mode == 0                                                  # on a page without a view: the mode switch


def test_sections_fold_and_alt_click_opens_only_one(env):
    w = _open(env, (1920, 1080))
    page = w.graph
    w.select(step="fond")
    _settle(w, 10)
    panel = page.panel
    ts = panel.sections
    assert len(ts) >= 2 and not any(t.folded for t in ts)
    QTest.mouseClick(ts[0], Qt.LeftButton, Qt.NoModifier, ts[0].rect().center())
    _settle(w)
    assert ts[0].folded and not ts[1].folded
    QTest.mouseClick(ts[1], Qt.LeftButton, Qt.AltModifier, ts[1].rect().center())
    _settle(w)
    assert ts[0].folded and not ts[1].folded and all(t.folded for t in ts[2:])
    QTest.mouseClick(ts[0], Qt.LeftButton, Qt.NoModifier, ts[0].rect().center())
    _settle(w)
    assert not ts[0].folded
    # the rows of a folded section go with it; a rebuild keeps it
    assert all(not r.isVisible() for r in ts[1].rows if ts[1].folded) or not ts[1].folded
    panel.refresh(panel.node)
    _settle(w)
    assert not panel.sections[0].folded


def test_a_small_window_shows_one_section_at_a_time_and_a_large_one_never(env):
    w = _open(env, (1366, 728))
    page = w.graph
    w.select(step="fond")
    _settle(w, 10)
    QTest.mouseClick(page.dock.icons.itemAt(0).widget(), Qt.LeftButton)
    _settle(w)
    ts = page.panel.sections
    assert len(ts) >= 2 and not ts[0].folded and all(t.folded for t in ts[1:])          # the first one open
    QTest.mouseClick(ts[1], Qt.LeftButton, Qt.NoModifier, ts[1].rect().center())         # a click: only that one
    _settle(w)
    assert ts[0].folded and not ts[1].folded
    w.resize(1920, 1080)
    _settle(w)
    assert not page.panel.solo


def test_the_setting_never_compact_is_todays_layout_on_any_screen(env):
    w = _open(env, (1366, 728))
    assert w.compact
    prefs.set("layout_density", "comfortable")
    w.apply_layout()
    _settle(w)
    assert not w.compact and w.graph.dock.state == "docked" and w.graph.left_state == "docked" \
        and w.proj_bar.height() == 46
    prefs.set("layout_density", "compact")
    w.apply_layout()
    _settle(w)
    assert w.compact


def test_the_dialogs_never_exceed_the_room(env):
    from astro_gpu.client.gallery import TemplateGallery
    from astro_gpu.client.settingsdlg import SettingsDialog
    w = _open(env, (1280, 720))
    made = [SettingsDialog(w.worker, w.machine, can_clear=True, parent=w), TemplateGallery(w.s, parent=w),
            w.about_dialog()]
    for d in made:
        d.show()
        _settle(w, 3)
        assert d.width() <= w.width() * 0.91 and d.height() <= w.height() * 0.91, d.__class__.__name__
        assert d.isSizeGripEnabled()
        d.reject()
        _settle(w, 2)


def _cut_controls(root: QWidget) -> list[str]:
    """The visible controls outside the window (anything inside a scroll area is reachable: it scrolls)."""
    bad = []
    for c in root.findChildren(QWidget):
        if not c.isVisible() or not isinstance(c, (QAbstractButton, QLabel, QComboBox, QLineEdit)):
            continue
        if isinstance(c, QLabel) and (not c.text() or c.wordWrap()):
            continue
        p = c.parentWidget()
        scrolled = False
        while p is not None and p is not root:
            if isinstance(p, QAbstractScrollArea) or isinstance(p.parentWidget(), QAbstractScrollArea):
                scrolled = True
                break
            p = p.parentWidget()
        if scrolled:
            continue
        r = c.rect().translated(c.mapTo(root, QPoint(0, 0)))
        if r.right() > root.width() + 1 or r.bottom() > root.height() + 1:
            bad.append(c.objectName() or getattr(c, "text", lambda: "")())
    return bad


@pytest.mark.parametrize("size", [(1536, 864), (1366, 728), (1280, 720)])
def test_no_control_is_cut_and_no_dialog_is_larger_than_the_window(env, size):
    from astro_gpu.client.gallery import TemplateGallery
    from astro_gpu.client.settingsdlg import SettingsDialog
    w = _open(env, size)
    bad = []
    for mode in (0, 1, 2):
        w.set_mode(mode)
        _settle(w, 8)
        bad += _cut_controls(w)
    w.show_home()
    _settle(w, 4)
    bad += _cut_controls(w)
    w.open_project(_project(env["tmp"], "Back"))
    w.worker.wait()
    _settle(w, 10)
    assert not bad, bad
    for d in (SettingsDialog(w.worker, w.machine, can_clear=True, parent=w), TemplateGallery(w.s, parent=w)):
        d.show()
        _settle(w, 3)
        assert d.height() <= w.height() and d.width() <= w.width()
        assert not _cut_controls(d), d.__class__.__name__
        d.reject()
        _settle(w, 2)


def test_the_smallest_window_is_about_1024_by_640(env):
    w = _open(env, (1024, 640))
    assert w.width() == 1024 and w.height() == 640 and w.minimumWidth() == 1024 and w.minimumHeight() == 640
    assert w.size_range == "small"
    # at that size Post and Inspect's pages are still reachable
    for mode in (2, 1, 0):
        w.set_mode(mode)
        _settle(w, 6)
    assert w.mode == 0 and not _cut_controls(w)


def test_post_folds_its_layers_panel_on_a_small_window(env):
    w = _open(env, (1366, 728))
    for v in w.s.paths:                                # the workflows as if computed: Post shows its panel
        v.rendered = True
    w.set_mode(2)
    _settle(w, 8)
    w.post.update_page()
    _settle(w, 8)
    view = w.post.view
    assert view.dock.state == "icons" and view.dock.strip.isVisible() and not view.right.isVisible()
    QTest.mouseClick(view.dock.icons.itemAt(0).widget(), Qt.LeftButton)
    _settle(w)
    assert view.dock.is_open() and view.right.isVisible()
    _activate(w)
    view.viewer.setFocus()
    QTest.keyClick(view.viewer, Qt.Key_Escape)
    _settle(w)
    assert not view.dock.is_open()
    # the pin docks it, remembered for this size range (its own: Post's fold never moves the graph's panel)
    QTest.mouseClick(view.dock.pin, Qt.LeftButton)
    _settle(w)
    assert view.dock.state == "docked" and layout.choices("small")["post"] == "docked" and "right" not in layout.choices("small")
    # Tab hides it too; on a large window it starts docked
    _key(w, Qt.Key_Tab, widget=view.viewer)
    assert view.dock.state == "hidden"
    _key(w, Qt.Key_Tab, widget=view.viewer)
    assert view.dock.state == "docked"
    w.resize(1920, 1080)
    _settle(w)
    assert w.size_range == "large" and view.dock.state == "docked"


# ---------------------------------------------------------------------------------------------- the graph's navigation
def test_the_wheel_zooms_at_the_pointer_and_the_zoom_is_the_users(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    vp = _vp(w)
    gv.zoom_to(2.0)                                     # room to scroll around the point on both axes
    _settle(w)
    it = _block(w, "calibration")
    pos = _point_of(w, it)
    before = gv.zoom()
    keep = gv.mapToScene(pos)
    _wheel(vp, pos, 120)
    _settle(w)
    assert gv.zoom() > before and gv._zoomed
    after = gv.mapToScene(pos)
    assert abs(after.x() - keep.x()) < 6 and abs(after.y() - keep.y()) < 6        # the point under the pointer stays
    assert gv.b_zoom.text() == f"{round(gv.zoom() * 100)} %"


def test_the_keys_and_the_presets_match_the_image_viewer(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    _key(w, Qt.Key_1)
    assert abs(gv.zoom() - 1.0) < 1e-6
    _key(w, Qt.Key_Plus)
    assert gv.zoom() > 1.2
    _key(w, Qt.Key_Minus)
    assert abs(gv.zoom() - 1.0) < 1e-6
    _key(w, Qt.Key_F)                                  # Fit is today's Show all
    assert gv.b_all.isChecked() and not gv._zoomed and gv.horizontalScrollBar().maximum() == 0
    gv.zoom_to(1.0)
    assert not gv.b_all.isChecked()
    _key(w, Qt.Key_0, Qt.ControlModifier)
    assert gv.b_all.isChecked()
    # the presets' menu, as the viewer's; D still cuts a block
    acts = gv.b_zoom.menu().actions()
    assert acts[0].text() == "Show all" and [a.text() for a in acts[1:]] == [
        "12.5 %", "25 %", "50 %", "100 %", "200 %", "400 %", "800 %"]
    acts[6].trigger()
    assert abs(gv.zoom() - 4.0) < 1e-6
    w.graph.graph.setFocus()
    assert "D" in [a.key().toString() for a in w.graph.findChildren(QShortcut)]    # D keeps cutting a block


def test_the_view_pans_with_the_middle_button_space_or_a_left_drag_on_the_empty_area(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    vp = _vp(w)
    gv.zoom_to(3.0)                                    # room to move on both axes
    _settle(w)
    hb, vb = gv.horizontalScrollBar(), gv.verticalScrollBar()
    assert hb.maximum() > 0 and vb.maximum() > 0
    # middle button
    a, b = QPoint(400, 300), QPoint(300, 260)
    h0, v0 = hb.value(), vb.value()
    _drag(vp, a, b, Qt.MiddleButton)
    assert abs(hb.value() - h0 - 100) <= 2 and abs(vb.value() - v0 - 40) <= 2
    # Space + left drag, over a block: it pans
    it = _block(w, "alignement")
    p = _point_of(w, it)
    _activate(w)
    vp.setFocus()
    QTest.keyPress(vp, Qt.Key_Space)
    h0 = hb.value()
    _drag(vp, p, p + QPoint(-50, 0))
    QTest.keyRelease(vp, Qt.Key_Space)
    assert abs(hb.value() - h0 - 50) <= 2 and w.s.sel_step != "alignement"    # moved, not selected
    # a left drag on the empty area pans; a left click on a block still selects it
    p0 = _empty_point(w)
    h0 = hb.value()
    _drag(vp, p0, p0 + QPoint(-30, 0))
    assert abs(hb.value() - h0 - 30) <= 2
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, _point_of(w, _block(w, "calibration")))
    _settle(w)
    assert w.s.sel_step == "calibration"


def test_a_click_on_the_empty_area_still_deselects_but_a_drag_does_not(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    vp = _vp(w)
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, _point_of(w, _block(w, "calibration")))
    _settle(w)
    assert w.graph.current_node().key == "calibration"
    p = _empty_point(w)
    gv.zoom_to(2.0)                                     # room to pan
    _settle(w)
    p = _empty_point(w)
    _drag(vp, p, p + QPoint(-40, -2))
    _settle(w)
    assert w.graph.current_node() is not None and w.graph.current_node().key == "calibration"
    QTest.mouseClick(vp, Qt.LeftButton, Qt.NoModifier, _empty_point(w))
    _settle(w)
    assert w.graph.current_node() is None


def test_the_minimap_shows_when_part_of_the_graph_is_off_screen_and_moves_the_view(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    assert not gv.map.isVisible() or gv.horizontalScrollBar().maximum() > 0
    gv.zoom_to(2.0)
    _settle(w)
    m = gv.map
    assert m.isVisible() and m.parent() is gv.viewport()
    assert m.x() + m.width() >= gv.viewport().width() - 20
    h0 = gv.horizontalScrollBar().value()
    QTest.mouseClick(m, Qt.LeftButton, Qt.NoModifier, QPoint(m.width() - 6, m.height() // 2))
    _settle(w)
    assert gv.horizontalScrollBar().value() > h0                    # moved towards the end of the graph
    h1 = gv.horizontalScrollBar().value()
    _drag(m, QPoint(m.width() - 6, m.height() // 2), QPoint(8, m.height() // 2))
    _settle(w)
    assert gv.horizontalScrollBar().value() < h1
    gv.fit_all()
    _settle(w)
    assert not m.isVisible()                                         # the whole graph in view: no map


def test_zoom_and_position_survive_a_rebuild_and_are_kept_with_the_project(env):
    root = _project(env["tmp"], "Kept")
    w = _open(env, (1366, 728), root)
    gv = w.graph.graph
    gv.zoom_to(1.6)
    gv.center_on(_block(w, "fond").sceneBoundingRect().center())
    _settle(w)
    k0 = gv.zoom()
    h0 = gv.horizontalScrollBar().value()
    w.graph.refresh()                                     # a graph rebuild (a block added, a cut, a run)
    _settle(w)
    assert abs(gv.zoom() - k0) < 1e-6 and abs(gv.horizontalScrollBar().value() - h0) <= 1
    w.close_project()
    _settle(w)
    saved = prefs.get("graph_view")
    assert len(saved) == 1 and next(iter(saved.values()))["k"] == pytest.approx(k0, abs=1e-3)
    assert project2.load(root).to_dict().get("graph_view") is None      # nothing written into the project by a look
    w.open_project(root)
    w.worker.wait()
    _settle(w, 10)
    assert abs(w.graph.graph.zoom() - k0) < 1e-3
    # Show all gives the automatic view back and forgets the user's
    w.graph.graph.b_all.setChecked(True)
    _settle(w)
    assert not w.graph.graph._zoomed and not prefs.get("graph_view")


def test_a_block_selected_off_screen_is_scrolled_into_view_the_zoom_kept(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    gv.zoom_to(2.0)
    gv.horizontalScrollBar().setValue(0)
    _settle(w)
    far = max(_blocks(w), key=lambda it: it.sceneBoundingRect().x())
    assert not gv.viewport().rect().contains(gv.mapFromScene(far.sceneBoundingRect()).boundingRect())
    w.graph.select(step=far.node.ref)                     # from the panel, a run, a search: the same selection
    _settle(w, 10)
    far = next(it for it in _blocks(w) if it.node.ref == far.node.ref)
    assert gv.viewport().rect().contains(gv.mapFromScene(far.sceneBoundingRect()).boundingRect())
    assert abs(gv.zoom() - 2.0) < 1e-6


def test_below_the_readable_size_only_the_titles_stay(env):
    w = _open(env, (1920, 1080))
    gv = w.graph.graph
    it = _block(w, "alignement")

    def text_pixels(k):
        gv.zoom_to(k, QPointF(gv.mapFromScene(it.sceneBoundingRect().center())))
        _settle(w, 3)
        im = gv.viewport().grab().toImage()
        r = gv.mapFromScene(it.mapToScene(11, 38, 90, 8)).boundingRect()      # the subtitle's band, under the title
        return sum(1 for y in range(r.top(), r.bottom()) for x in range(r.left(), r.right())
                   if im.pixelColor(x, y).lightness() > 90)
    assert text_pixels(1.0) > 8
    assert text_pixels(MIN_SCALE - 0.1) == 0


def test_a_touchpad_pans_with_two_fingers_and_pinches_in_the_graph(env):
    w = _open(env, (1366, 728))
    gv = w.graph.graph
    vp = _vp(w)
    gv.zoom_to(2.0)
    _settle(w)
    hb, k0 = gv.horizontalScrollBar(), gv.zoom()
    h0 = hb.value()
    pos = QPoint(500, 300)
    _wheel(vp, pos, dy=0, dx=-30, phase=Qt.ScrollUpdate, pixel=(-30, 0))        # two fingers: fine deltas, phase
    _settle(w)
    assert abs(hb.value() - h0 - 30) <= 2 and abs(gv.zoom() - k0) < 1e-6
    _wheel(vp, pos, dy=7, dx=0)                                                 # a Windows touchpad: a fine delta alone
    _settle(w)
    assert abs(gv.zoom() - k0) < 1e-6
    _wheel(vp, pos, dy=60, mods=Qt.ControlModifier)                             # a pinch on Windows: Ctrl + wheel
    _settle(w)
    assert gv.zoom() > k0
    k1 = gv.zoom()
    QApplication.sendEvent(vp, QNativeGestureEvent(Qt.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2,
                                                   QPointF(pos), QPointF(pos), QPointF(vp.mapToGlobal(pos)), 0.25,
                                                   QPointF()))
    _settle(w)
    assert gv.zoom() == pytest.approx(k1 * 1.25, abs=1e-3)


# ---------------------------------------------------------------------------------------------- Post's viewer
def _viewer(qapp):
    v = Viewer()
    v.resize(800, 600)
    v.set_source(StaticSource(np.full((900, 1400, 3), 90, np.uint8)))
    v.show()
    qapp.processEvents()
    return v


def test_the_post_viewer_has_the_graphs_gestures(qapp):
    v = _viewer(qapp)
    try:
        k0 = v.scale
        _wheel(v, QPoint(200, 100), 120)                  # the wheel zooms at the pointer
        assert v.scale == pytest.approx(k0 * 1.2)
        k1 = v.scale
        cx = v.cx
        _wheel(v, QPoint(200, 100), dy=0, dx=-40, phase=Qt.ScrollUpdate, pixel=(-40, 0))   # a touchpad pans
        assert v.scale == k1 and v.cx == pytest.approx(cx + 40 / v.scale)
        _wheel(v, QPoint(200, 100), 60, mods=Qt.ControlModifier)                           # a pinch (Windows)
        assert v.scale > k1
        k2 = v.scale
        QApplication.sendEvent(v, QNativeGestureEvent(Qt.ZoomNativeGesture, QPointingDevice.primaryPointingDevice(), 2,
                                                      QPointF(100, 100), QPointF(100, 100),
                                                      QPointF(v.mapToGlobal(QPoint(100, 100))), 0.5, QPointF()))
        qapp.processEvents()
        assert v.scale == pytest.approx(k2 * 1.5, abs=1e-3)              # a pinch (macOS, Wayland)
        # Ctrl+1 is 100 %; Space + drag pans over a tool; the minimap appears and moves the view
        QTest.keyClick(v, Qt.Key_1, Qt.ControlModifier)
        qapp.processEvents()
        assert v.scale == 1.0 and not v.whole_visible()
        v.mode = "brush"
        cx0 = v.cx
        v.setFocus()
        QTest.keyPress(v, Qt.Key_Space)
        _drag(v, QPoint(400, 300), QPoint(370, 300))
        QTest.keyRelease(v, Qt.Key_Space)
        v.mode = "pan"
        assert v.cx == pytest.approx(cx0 + 30)
        r = v.minimap_rect()
        x0 = v.cx
        QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, r.center().toPoint() + QPoint(30, 0))
        qapp.processEvents()
        assert v.cx != x0
        # fit: no minimap; the wheel with a fitted image zooms as before
        v.fit()
        assert v.whole_visible() and not v._in_minimap(QPointF(790, 590))
    finally:
        v.close()
        v.deleteLater()
