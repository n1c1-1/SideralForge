"""Issue #44: the GUI on small laptop screens, and the graph's zoom and navigation (real gestures on the real window).

The size ranges and their dead band, large screens unchanged, the right panel as icons (over the graph, Esc or a click
elsewhere closes it, the pin docks it), Tab / Shift+Tab / Ctrl+Space with the View menu, the solo sections, the setting,
Reset layout, the dialogs' size; then the graph as the Post image: wheel at the pointer, keys, presets, pan, minimap,
kept through rebuilds and with the project, a selection scrolled into view, the titles only below the readable size,
the touchpad; and the same in Post's viewer.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if Path("C:/Windows/Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QEnterEvent, QMouseEvent, QNativeGestureEvent, QPointingDevice, QShortcut, QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractScrollArea,
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QWidget,
)

from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client import layout, post3, prefs  # noqa: E402
from astro_gpu.client import graph as graphmod  # noqa: E402
from astro_gpu.client.graph import (  # noqa: E402
    MIN_SCALE,
    PALETTE_FOLD_MS,
    PALETTE_FOLDED_W,
    PALETTE_UNFOLD_MS,
    PALETTE_W,
    BlockItem,
    GhostItem,
    WireItem,
)
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
    assert not page.panel.isVisible() and page.left_state == "auto" and page.palette_bar.width() < PALETTE_W
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


def _mouse_on_palette(page, inside: bool):
    bar = page.palette_bar
    QApplication.sendEvent(bar, QEnterEvent(QPointF(5, 5), QPointF(5, 5), QPointF(bar.mapToGlobal(QPoint(5, 5))))
                           if inside else QEvent(QEvent.Leave))


def test_the_palette_of_a_small_screen_unfolds_under_the_mouse_over_the_graph(env):
    w = _open(env, (1366, 728))
    page = w.graph
    card = page.cards["denoise"]
    graph_at, y = page.graph.mapTo(page, QPoint(0, 0)), card.y()
    assert page.palette_slot.width() == page.palette_bar.width() == PALETTE_FOLDED_W and not card.text.isVisible()
    assert card.width() <= PALETTE_FOLDED_W - 12
    # the mouse only crossing it opens nothing
    _mouse_on_palette(page, True)
    _mouse_on_palette(page, False)
    QTest.qWait(PALETTE_UNFOLD_MS + 60)
    assert page.palette_bar.width() == PALETTE_FOLDED_W
    # staying: unfolded, over the graph, nothing moved
    _mouse_on_palette(page, True)
    QTest.qWait(PALETTE_UNFOLD_MS + 60)
    _settle(w, 2)
    assert page.palette_bar.width() == PALETTE_W and card.text.isVisible() and card.y() == y
    assert page.palette_slot.width() == PALETTE_FOLDED_W and page.graph.mapTo(page, QPoint(0, 0)) == graph_at
    assert page.palette_bar.geometry().right() > graph_at.x() and card.width() > PALETTE_FOLDED_W
    # a slip out and back keeps it; out for good folds it
    _mouse_on_palette(page, False)
    _mouse_on_palette(page, True)
    QTest.qWait(PALETTE_FOLD_MS + 60)
    assert page.palette_bar.width() == PALETTE_W
    _mouse_on_palette(page, False)
    QTest.qWait(PALETTE_FOLD_MS + 60)
    _settle(w, 2)
    assert page.palette_bar.width() == PALETTE_FOLDED_W and not card.text.isVisible() and card.y() == y


def test_a_drag_from_the_unfolded_palette_folds_it_and_keeps_its_card(env, monkeypatch):
    w = _open(env, (1366, 728))
    page = w.graph
    card, seen = page.cards["denoise"], []

    class Drag:
        def __init__(self, source):
            pass

        def setMimeData(self, m):                                # noqa: N802
            pass

        def setPixmap(self, p):                                  # noqa: N802
            pass

        def setHotSpot(self, p):                                 # noqa: N802
            pass

        def exec(self, action):
            seen.append(page.palette_bar.width())
    monkeypatch.setattr(graphmod, "QDrag", Drag)
    page.unfold_palette(True)
    assert page.palette_bar.width() == PALETTE_W
    _press(card, QPoint(5, 5))
    _move(card, QPoint(40, 5))
    assert seen == [PALETTE_FOLDED_W] and page.cards["denoise"] is card


def test_the_palette_menu_has_three_ways_each_remembered(env, monkeypatch):
    w = _open(env, (1366, 728))
    page = w.graph
    want = []
    class Menu(QMenu):                                           # built, not run (exec would wait for a click)
        def exec(self, *a):
            return next(x for x in self.actions() if x.data() == want[-1])
    monkeypatch.setattr(graphmod, "QMenu", Menu)

    def choose(state):
        want.append(state)
        QTest.mouseClick(page.palette_button, Qt.LeftButton)
        _settle(w, 2)
    choose("docked")
    assert page.left_state == "docked" and page.palette_slot.width() == page.palette_bar.width() == PALETTE_W
    assert layout.choices("small")["left"] == "docked"
    _mouse_on_palette(page, False)
    QTest.qWait(PALETTE_FOLD_MS + 60)
    assert page.palette_bar.width() == PALETTE_W                 # always open: the mouse leaving changes nothing
    choose("folded")
    assert page.left_state == "folded" and page.palette_slot.width() == page.palette_bar.width() == PALETTE_FOLDED_W
    _mouse_on_palette(page, True)
    QTest.qWait(PALETTE_UNFOLD_MS + 60)
    assert page.palette_bar.width() == PALETTE_FOLDED_W          # always folded: the mouse opens nothing
    _mouse_on_palette(page, False)
    choose("auto")
    assert page.left_state == "auto" and layout.choices("small")["left"] == "auto"
    _mouse_on_palette(page, True)
    QTest.qWait(PALETTE_UNFOLD_MS + 60)
    assert page.palette_bar.width() == PALETTE_W


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
    ts = panel.folds.titles
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
    assert not panel.folds.titles[0].folded


def test_a_small_window_never_folds_a_section_by_itself(env):
    """#119: the small window opened one section at a time and closed the others; only the user folds now, the column
    scrolls."""
    w = _open(env, (1366, 728))
    page = w.graph
    w.select(step="fond")
    _settle(w, 10)
    QTest.mouseClick(page.dock.icons.itemAt(0).widget(), Qt.LeftButton)
    _settle(w)
    ts = page.panel.folds.titles
    assert len(ts) >= 2 and not any(t.folded for t in ts)
    QTest.mouseClick(ts[1], Qt.LeftButton, Qt.NoModifier, ts[1].rect().center())         # a click folds that one only
    _settle(w)
    assert not ts[0].folded and ts[1].folded


def _post_view(w):
    for v in w.s.paths:                                # the workflows as if computed: Post shows its panel
        v.rendered = True
    w.set_mode(2)
    _settle(w, 8)
    w.post.update_page()
    _settle(w, 8)
    return w.post.view


def test_posts_property_groups_fold_as_the_block_panels_sections(env):
    w = _open(env, (1920, 1080))
    view = _post_view(w)
    ts = view.folds.titles
    assert len(ts) >= 2 and not any(t.folded for t in ts) and ts[0].rows
    row = next(r for r in ts[0].rows if r.isVisible())
    QTest.mouseClick(ts[0], Qt.LeftButton, Qt.NoModifier, ts[0].rect().center())
    _settle(w)
    assert ts[0].folded and not row.isVisible() and not ts[1].folded
    # Alt-click: only that one; a rebuild of the column keeps the folds
    QTest.mouseClick(ts[1], Qt.LeftButton, Qt.AltModifier, ts[1].rect().center())
    _settle(w)
    assert ts[0].folded and not ts[1].folded
    view.build_props()
    _settle(w)
    assert view.folds.titles[0].folded and not view.folds.titles[1].folded


def test_a_row_its_panel_keeps_hidden_stays_hidden_when_its_section_opens(env):
    w = _open(env, (1920, 1080))
    view = _post_view(w)
    t = view.folds.titles[0]
    kept = next(r for r in t.rows if r.isVisible())
    kept.hide()
    t.set_folded(True)
    t.set_folded(False)
    _settle(w)
    assert kept.isHidden()


def test_posts_groups_all_stay_open_on_a_small_window(env):
    """#119: opening Starless closed Stars alone and the other way round; the user folds, the column scrolls."""
    w = _open(env, (1366, 728))
    view = _post_view(w)
    ts = view.folds.titles
    assert len(ts) >= 2 and not any(t.folded for t in ts)
    # a title that stands out from the settings: its own look, that nothing sets again after it (the laptop's check:
    # Post and the block panel put the grey one back)
    assert ts[0].objectName() == "sectiontitle" and "#e6edf7" in ts[0].styleSheet() and "border-top" in ts[0].styleSheet()
    w.set_mode(0)
    w.select(step="fond")
    _settle(w, 10)
    assert all("#e6edf7" in t.styleSheet() for t in w.graph.panel.folds.titles)
    icons = [view.dock.icons.itemAt(i).widget().toolTip() for i in range(view.dock.icons.count())]
    assert [t.text() for t in ts][0] in icons                  # one icon per group


def test_a_dialog_left_open_elsewhere_does_not_keep_the_panel_open(env):
    """What the click or the key went to decides, never what else is open in the process (a modal dialog of another
    window, or left by a previous test in the same worker, kept the panel open)."""
    w = _open(env, (1366, 728))
    page = w.graph
    other = QDialog()
    other.setModal(True)
    other.show()
    _settle(w, 2)
    try:
        assert QApplication.activeModalWidget() is other
        QTest.mouseClick(_icon(page), Qt.LeftButton)
        _settle(w)
        assert page.dock.is_open()
        QTest.mouseClick(_vp(w), Qt.LeftButton, Qt.NoModifier, _empty_point(w))
        _settle(w)
        assert not page.dock.is_open()
    finally:
        other.reject()
        other.deleteLater()


def test_a_list_opened_from_the_panel_keeps_its_click_and_its_escape(env):
    w = _open(env, (1366, 728))
    page = w.graph
    QTest.mouseClick(_icon(page), Qt.LeftButton)
    _settle(w)
    popup = QMenu(page.panel)                                  # a window of its own, as a combo's list
    popup.addAction("x")
    popup.show()
    _settle(w, 2)
    try:
        QTest.keyClick(popup, Qt.Key_Escape)
        _settle(w, 2)
        assert page.dock.is_open()
        popup.show()
        _settle(w, 2)
        QTest.mouseClick(popup, Qt.LeftButton, Qt.NoModifier, QPoint(2, 2))
        _settle(w, 2)
        assert page.dock.is_open()
    finally:
        popup.close()
        popup.deleteLater()


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


def _wait(w, until, ms=1500):
    """Real time passing (the automatic side's delays), the events handled meanwhile."""
    end = time.monotonic() + ms / 1000
    while not until() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    return until()


def test_post_has_its_layers_on_the_left_and_its_settings_on_the_right(env):
    """As the usual image editors; on a small window the layers are automatic (the strip opens under the mouse), the
    settings docked."""
    w = _open(env, (1366, 728))
    view = _post_view(w)
    assert view.ldock.state == "auto" and view.ldock.strip.isVisible() and not view.left.isVisible()
    assert view.dock.state == "docked" and view.right.isVisible()
    assert view.ldock.strip.x() < view.viewer.mapTo(view, QPoint(0, 0)).x() < view.right.mapTo(view, QPoint(0, 0)).x()
    # the mouse resting on the strip opens the layers over the image; leaving them closes them
    QTest.mouseMove(view.ldock.strip, QPoint(20, 200))
    assert _wait(w, view.ldock.is_open) and view.left.isVisible()
    assert view.left.geometry().left() >= view.ldock.strip.geometry().right()
    QTest.mouseMove(view.viewer, view.viewer.rect().center())
    assert _wait(w, lambda: not view.ldock.is_open()) and not view.left.isVisible()
    # only crossing it opens nothing
    QTest.mouseMove(view.ldock.strip, QPoint(20, 200))
    QTest.mouseMove(view.viewer, view.viewer.rect().center())
    _wait(w, lambda: False, 400)
    assert not view.ldock.is_open()


def test_post_sides_are_shown_as_chosen_and_remembered(env):
    w = _open(env, (1366, 728))
    view = _post_view(w)
    view.ldock._choose("docked")                               # as the menu at its corner does
    _settle(w)
    assert view.ldock.state == "docked" and view.left.isVisible() and layout.choices("small")["post_left"] == "docked"
    assert "left" not in layout.choices("small")                # Post's own: the graph's palette did not move
    view.dock._choose("icons")
    _settle(w)
    assert view.dock.state == "icons" and view.dock.strip.isVisible() and not view.right.isVisible()
    QTest.mouseClick(view.dock.icons.itemAt(0).widget(), Qt.LeftButton)
    _settle(w)
    assert view.dock.is_open() and view.right.isVisible()
    assert view.right.geometry().right() <= view.dock.strip.geometry().left()
    _activate(w)
    view.viewer.setFocus()
    QTest.keyClick(view.viewer, Qt.Key_Escape)
    _settle(w)
    assert not view.dock.is_open()
    QTest.mouseClick(view.dock.pin, Qt.LeftButton)
    _settle(w)
    assert view.dock.state == "docked" and layout.choices("small")["post_right"] == "docked"
    # Tab hides both, Shift+Tab the settings only; on a large window both start docked
    _key(w, Qt.Key_Tab, widget=view.viewer)
    assert view.dock.state == "hidden" and view.ldock.state == "hidden" and view.ldock.arrow.isVisible()
    _key(w, Qt.Key_Tab, widget=view.viewer)
    assert view.dock.state == "docked" and view.ldock.state == "docked"
    _key(w, Qt.Key_Backtab, Qt.ShiftModifier, widget=view.viewer)
    assert view.dock.state == "hidden" and view.ldock.state == "docked"
    _key(w, Qt.Key_Backtab, Qt.ShiftModifier, widget=view.viewer)
    w.reset_layout()
    w.resize(1920, 1080)
    _settle(w)
    assert w.size_range == "large" and view.dock.state == "docked" and view.ldock.state == "docked"
    # widths dragged, then Reset layout: back at once (the night test pass: kept until the next docking)
    view.split.setSizes([360, 1000, 464])
    view._panel_moved()
    w.reset_layout()
    _settle(w)
    assert abs(view.left.width() - post3.LAYERS_W) <= 2 and abs(view.right.width() - post3.PANEL_W) <= 2


def test_posts_keys_answer_right_after_its_tab_is_clicked(env):
    """The night test pass: after a click on the Post tab, R (and C, V…) did nothing: the keyboard stayed on the tab."""
    w = _open(env, (1920, 1080))
    for v in w.s.paths:
        v.rendered = True
    _activate(w)
    QTest.mouseClick(w.mode_btns[2], Qt.LeftButton)
    _settle(w, 8)
    _activate(w)
    view = w.post.view
    assert view.isAncestorOf(QApplication.focusWidget()) or QApplication.focusWidget() is view.viewer
    QTest.keyClick(QApplication.focusWidget(), Qt.Key_R)
    _settle(w)
    assert view.b_crop.isChecked()


def test_escape_while_cropping_cancels_the_crop_and_stays_in_post(env):
    """#143: Escape while the crop was drawn left Post for the graph, the box drawn so far kept as the crop and the tool
    still on. As in the reference image editors, Escape is the tool's first (Cancel), and leaves Post only when no tool
    is on."""
    w = _open(env, (1920, 1080))
    view = _post_view(w)
    assert view.b_crop_cancel.toolTip().endswith("(Esc)")
    view.b_crop.setChecked(True)
    _settle(w)
    view._set_crop([10, 20, 300, 200])                                   # a box drawn (what a drag of its edge does)
    _key(w, Qt.Key_Escape, widget=view.viewer)
    assert w.mode == 2 and not view.b_crop.isChecked() and view.crop is None and view.viewer.crop_box is None
    assert not view.b_crop_cancel.isVisible()
    _key(w, Qt.Key_Escape, widget=view.viewer)                          # no tool on: Escape leaves Post, as before
    assert w.mode == 0


def _bar_fits(view):
    """The bar's widgets one after the other, none over the next, the last inside the centre; its hint the room."""
    _settle(view.window(), 3)
    bar = view.top_bar
    shown = [bar.itemAt(i).widget() for i in range(bar.count())]
    shown = [x for x in shown if x is not None and not x.isHidden()]
    for a, b in zip(shown, shown[1:]):
        assert a.geometry().right() < b.geometry().left(), (a.text() if hasattr(a, "text") else a, view.centre.width())
    assert shown[-1].geometry().right() < view.centre.width()
    assert bar.sizeHint().width() <= view.centre.width()
    k = next(k for k in range(len(view._bar_steps()) + 1) if view._bar_need(k) <= view.centre.width()
             or k == len(view._bar_steps()))
    assert view._bar_need(k) == bar.sizeHint().width()          # the fit's own sum is the layout's, to the pixel


@pytest.mark.parametrize("size, lang", [((1366, 700), "en"), ((1366, 700), "fr"), ((1366, 728), "en"),
                                        ((1280, 720), "fr"), ((1024, 640), "en")])
def test_posts_tools_go_into_more_rather_than_push_the_sides_out(env, size, lang):
    """The night test pass: on a laptop with the layers docked, the bar of tools held the image at 1011 px and pushed
    the layers out of the window, its buttons cut; dragging an edge did the same. #142: at 1366 x 700 New version
    covered the arrow of the version list (the fit was tried before the zoom's text grew): no overlap at any width
    from the centre's minimum, the crop's Apply and Cancel shown or not."""
    appmod.apply_language(lang)
    w = _open(env, size)
    view = _post_view(w)
    view.ldock._choose("docked")
    view.dock._choose("docked")
    _settle(w)
    sizes = view.split.sizes()
    assert sum(sizes) <= view.split.width() + 2 * len(sizes) and all(x > 0 for x in sizes)
    assert view.left.geometry().left() >= 0 and view.right.geometry().right() <= view.split.width()
    assert view.b_more.isVisible() and not view.b_names.isVisible()               # the least used first
    _bar_fits(view)
    view._fill_more()
    hidden = [a.text() for a in view.more_menu.actions()]
    assert view.b_names.text() in hidden and view.b_export.isVisible()
    a = next(x for x in view.more_menu.actions() if x.text() == view.b_names.text())
    assert a.isEnabled() == view.b_names.isEnabled()
    for zoom in (0.27, 1.0):                                    # "Zoom 27 %" is wider than "100 %"
        for crop in (True, False):
            view.viewer.zoom_to(zoom)
            view.b_crop.setChecked(crop)
            _bar_fits(view)
    view.b_crop.setChecked(True)                                 # the centre at its narrowest
    view.split.setSizes([post3.LAYERS_MAX, 0, post3.PANEL_MAX])
    _settle(w)
    assert view.centre.width() == post3.CENTRE_MIN, view.split.sizes()
    _bar_fits(view)
    view._fill_more()
    names = [a.text() for a in view.more_menu.actions()]
    assert view.b_crop_ok.isVisible() and view.b_crop_cancel.isVisible() and view.zoom_lab.isVisible()
    assert all(n for n in names) and (view.b_undo.isVisible() or post3._("Undo") in names)
    view.b_crop.setChecked(False)
    w.resize(1920, 1080)
    _settle(w)
    view._fit_split()
    _settle(w)
    _bar_fits(view)
    if lang == "en":                                             # (the French names are longer: one tool or two in "…")
        assert not view.b_more.isVisible() and view.b_names.isVisible()
    assert view.b_crop.isVisible() and view.b_undo.isVisible() and view.b_version_new.text()


def test_a_long_version_name_shows_its_start_and_never_pushes_the_bar(env):
    """#146: "Strong stars, soft background" read "oft background", after renaming, a click elsewhere and a resize. As
    the reference image editors: its start, cut at its end with "…", the whole name on hover; the list's width bounded."""
    w = _open(env, (1366, 728))
    view = _post_view(w)
    view.ldock._choose("docked")
    view.dock._choose("docked")
    long_name = "Strong stars, soft background, typed while cropping"
    w.s.rename_post_version(view._version, long_name)
    view._fill_versions()
    _settle(w)
    cb, ed = view.cb_version, view.cb_version.lineEdit()
    assert cb.currentText() == long_name and ed.cursorPosition() == 0
    assert ed.shown().startswith("Strong") and ed.shown().endswith("…") and long_name in cb.toolTip()
    assert cb.width() <= post3.VERSION_W[0][1]
    _bar_fits(view)
    # typed in, then Enter and a click elsewhere: its start again
    _activate(w)
    ed.setFocus()
    _settle(w, 2)
    ed.end(False)
    QTest.keyClicks(ed, " again")
    assert ed.hasFocus() and ed.cursorPosition() == len(long_name) + 6 and ed.shown() == ed.text()
    QTest.keyClick(ed, Qt.Key_Return)
    _settle(w, 2)
    assert cb.currentText() == long_name + " again" and ed.cursorPosition() == 0
    view.viewer.setFocus()
    _settle(w, 2)
    assert not ed.hasFocus() and ed.cursorPosition() == 0 and ed.shown().startswith("Strong")
    w.resize(1280, 720)
    _settle(w)
    assert ed.cursorPosition() == 0
    _bar_fits(view)


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


def test_after_a_crop_the_minimap_is_of_the_cropped_picture(qapp):
    """#144: after a crop the minimap of the whole image stayed over the corner of the cropped picture fitted to the
    view. Fitted: none; zoomed in: the cropped picture, the frame of the view on its place in it, a click going there."""
    a = np.zeros((900, 1400, 3), np.uint8)
    a[100:500, 300:900] = 200                                    # the crop: bright, the rest black
    v = Viewer()
    v.resize(800, 600)
    v.set_source(StaticSource(a))
    v.show()
    qapp.processEvents()
    try:
        v.crop_box = (300, 100, 900, 500)
        v.fit()
        assert v.whole_visible() and not v._in_minimap(QPointF(790, 590))
        v.zoom_to(2.0, 600, 300)                                  # the view's centre: the crop's centre
        assert not v.whole_visible()
        r = v.minimap_rect()
        assert r.width() / r.height() == pytest.approx(600 / 400, rel=0.02)
        thumb = v._thumbnail()
        assert thumb.pixelColor(1, 1).red() == 200 and thumb.pixelColor(thumb.width() - 2, thumb.height() - 2).red() == 200
        img = v.grab().toImage()
        assert img.pixelColor(r.center().toPoint() + QPoint(int(r.width() * 0.4), 0)).red() > 150   # its pictures, not black
        QTest.mouseClick(v, Qt.LeftButton, Qt.NoModifier, (r.topLeft() + QPointF(r.width() / 4, r.height() / 4)).toPoint())
        qapp.processEvents()
        assert v.cx == pytest.approx(450, abs=6) and v.cy == pytest.approx(200, abs=6)
        v.crop_edit = True                                       # the Crop tool on: the whole image again
        assert v.extent() == (0, 0, 1400, 900)
    finally:
        v.close()
        v.deleteLater()


def test_a_project_of_two_stretched_pictures_opens_again(env, monkeypatch):
    """The night test pass: a project of two JPEGs (two workflows without blocks) did not open again: the graph took
    their shared Output out twice, and the window stayed on Home with the project half open."""
    import cv2
    import numpy as np
    root = env["tmp"] / "ws" / "Two"
    p = project2.create(root, "Two", Target(name="M 31"))
    for n in ("a", "b"):
        jpg = env["tmp"] / f"{n}.jpg"
        cv2.imwrite(str(jpg), np.full((8, 8, 3), 128, np.uint8))
        p.add_workflow(n, {}).source = {"path": jpg.as_posix(), "stretched": True}
        p.workflows[-1].blocks = []
    project2.save(p)
    w = _open(env, (1600, 900), root)
    assert w.graph_page() is not None and [q.name for q in w.s.paths] == ["a", "b"]
    # a project that fails to be shown for another reason: closed and said, never left half open
    w.close_project()
    monkeypatch.setattr(w, "build_project", lambda: 1 / 0)
    assert w.open_project(root) is False and w.project is None and w.s is None
    assert "cannot be shown" in w.banner.text.text()
