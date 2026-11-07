"""Shared helpers of the sources_and_files exploration (explore_sources_*.py): waiting for the worker, the panel's
widgets and texts, overlaps and cut texts, the texts left in English, a de-duplicated note."""
from __future__ import annotations

import faulthandler
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import TMP, Explorer, stray_qt_errors  # noqa: E402,F401
from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QWidget,
)

from astro_gpu.client import i18n  # noqa: E402
from astro_gpu.client.i18n import _  # noqa: E402,F401

PY_ERR = re.compile(r"Traceback|Error:|Exception|KeyError|ValueError|TypeError|AttributeError|NoneType|\bNone\b")
INTERNAL = re.compile(r"\b(legacy|v1|v2|render_simple|star_removal|astrometrie|debruitage|etoiles|"
                      r"\w+~\d|session@\d|calibration@\d|pid|sid)\b")


EXC: list[str] = []                       # the exceptions the window raised in its slots (sys.excepthook)


def start(timeout: int = 1500):
    """A hard limit on the script (the tracebacks of every thread printed, then exit), the Qt warnings shown, and the
    exceptions raised by the window's slots kept (``EXC``)."""
    import traceback
    faulthandler.dump_traceback_later(timeout, exit=True)
    stray_qt_errors()

    def hook(t, v, tb):
        EXC.append("".join(traceback.format_exception(t, v, tb)))
        sys.__excepthook__(t, v, tb)
    sys.excepthook = hook


def wait(ex: Explorer, secs: float = 0.6):
    """Events processed, the worker's answers delivered, for ``secs`` at least, then until the model is ready."""
    end = time.time() + secs
    while time.time() < end:
        ex.settle(2)
    if ex.w is not None:
        ex.w.worker.wait(20)
    ex.settle(4)
    s = getattr(ex.w, "s", None) if ex.w is not None else None
    t1 = time.time() + 20
    while s is not None and not s.ready and time.time() < t1:
        ex.settle(2)
        ex.w.worker.wait(2)
    ex.settle(4)


def panel(ex):
    return ex.page.panel


def visible(root: QWidget, cls) -> list:
    return [w for w in root.findChildren(cls) if w.isVisibleTo(root.window()) and w.isVisible()]


def texts(root: QWidget) -> list[str]:
    """Every text a visible widget under ``root`` shows (labels, buttons, edits, combos), and their tooltips."""
    out = []
    for w in [root] + root.findChildren(QWidget):
        if not w.isVisible():
            continue
        if isinstance(w, (QLabel, QAbstractButton)) and w.text():
            out.append(w.text())
        elif isinstance(w, QLineEdit) and (w.text() or w.placeholderText()):
            out.append(w.text() or w.placeholderText())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        if w.toolTip():
            out.append(w.toolTip())
    return out


def ptexts(ex) -> list[str]:
    return texts(panel(ex))


def button(root: QWidget, part: str, cls=QAbstractButton):
    """The first visible button whose text holds ``part`` (case not minded)."""
    for b in root.findChildren(cls):
        if b.isVisible() and part.lower() in b.text().lower():
            return b
    return None


def buttons(root: QWidget, part: str, cls=QAbstractButton) -> list:
    return [b for b in root.findChildren(cls) if b.isVisible() and part.lower() in b.text().lower()]


def menu_button(ex, b, pick: str | None) -> list[dict]:
    """A button with a menu clicked: its menu's items (and ``pick`` triggered)."""
    n = len(ex.menus)
    ex.menu_pick = pick
    ex.click(b)
    ex.settle(4)
    return ex.menus[-1] if len(ex.menus) > n else []


def overlaps(root: QWidget, min_px: int = 3) -> list[tuple[str, str]]:
    """Visible text widgets under ``root`` drawn over each other (neither inside the other)."""
    ws = [w for w in root.findChildren(QWidget) if w.isVisible()
          and isinstance(w, (QLabel, QAbstractButton, QLineEdit, QComboBox, QAbstractSpinBox))
          and not w.visibleRegion().isEmpty() and w.width() > 2 and w.height() > 2]
    # what is really drawn of each: its visible region (a row scrolled out of its list is clipped by the list)
    rects = [(w, w.visibleRegion().boundingRect().translated(w.mapTo(root, QPoint(0, 0)))) for w in ws]
    out = []
    for i, (a, ra) in enumerate(rects):
        for b, rb in rects[i + 1:]:
            if a.isAncestorOf(b) or b.isAncestorOf(a):
                continue
            inter = ra.intersected(rb)
            if inter.width() > min_px and inter.height() > min_px:
                out.append((_desc(a), _desc(b)))
    return out


def scene_overlaps(ex) -> list[tuple[str, str]]:
    """Blocks and dashed places of the graph drawn over each other (the drop places of a drag aside)."""
    from astro_gpu.client.graph import BlockItem, GhostItem
    items = [it for it in ex.view.scene().items() if isinstance(it, (BlockItem, GhostItem))
             and not (isinstance(it, GhostItem) and it in getattr(ex.view, "ghosts", []))]
    out = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            r = a.sceneBoundingRect().intersected(b.sceneBoundingRect())
            if r.width() > 4 and r.height() > 4:
                name = lambda x: getattr(x, "title", None) or getattr(x, "text", "") or x.__class__.__name__  # noqa
                out.append((str(name(a))[:40], str(name(b))[:40]))
    return out


def _desc(w) -> str:
    t = getattr(w, "text", lambda: "")()
    if isinstance(w, QComboBox):
        t = w.currentText()
    return f"{w.__class__.__name__}('{str(t)[:40]}')"


def cut_texts(root: QWidget) -> list[str]:
    """Texts cut without a tooltip: a label without word wrap narrower than its text, a button narrower than its
    text."""
    out = []
    for w in root.findChildren(QWidget):
        if not w.isVisible() or w.visibleRegion().isEmpty():
            continue
        if isinstance(w, QLabel) and w.text() and not w.wordWrap() and w.textFormat() != Qt.RichText \
                and "<" not in w.text() and w.pixmap().isNull():
            need = w.fontMetrics().horizontalAdvance(w.text().split("\n")[0])
            m = w.contentsMargins()
            if need > w.width() - m.left() - m.right() + 2 and not w.toolTip() \
                    and w.__class__.__name__ != "ElidedLabel":
                out.append(f"QLabel '{w.text()[:60]}' ({need} px of text in {w.width()} px)")
        elif isinstance(w, QPushButton) and w.text():
            need = w.fontMetrics().horizontalAdvance(w.text()) + 8
            if need > w.width() and not w.toolTip():
                out.append(f"QPushButton '{w.text()[:60]}' ({need} px of text in {w.width()} px)")
        elif isinstance(w, QComboBox) and w.currentText():
            need = w.fontMetrics().horizontalAdvance(w.currentText()) + 28
            if need > w.width() and not w.toolTip():
                out.append(f"QComboBox '{w.currentText()[:60]}' ({need} px in {w.width()} px)")
    return out


def raw_words(txts: list[str]) -> list[str]:
    """Texts holding a Python error or an internal word."""
    out = []
    for t in txts:
        if PY_ERR.search(t) or INTERNAL.search(t):
            out.append(t)
    return out


def missing_fr() -> set[str]:
    return set(i18n.missing())


def graph_texts(ex) -> list[tuple[str, str, str]]:
    """(key, title, summary) of every drawn block."""
    return [(it.node.key, it.title, it.summary) for it in ex.blocks()]


def block_by_title(ex, part: str):
    for it in ex.blocks():
        if part.lower() in (it.title or "").lower():
            return it
    return None


class Notes:
    """One note per distinct anomaly (its title), and the list of what was checked."""

    def __init__(self, ex: Explorer):
        self.ex = ex
        self.seen = {n["title"] for n in ex.notes()}
        self.checked: list[str] = []

    def note(self, severity, title, detail="", shot=True, steps=None):
        if title in self.seen:
            return
        self.seen.add(title)
        self.ex.note(severity, title, detail, shot=shot, steps=steps or [])

    def exceptions(self, steps: list | None = None, done: list = []):  # noqa: B006 - the ones noted already
        """The exceptions raised since the last call: each an anomaly (bloquant), with the steps that led to it."""
        for e in EXC[len(done):]:
            done.append(e)
            last = e.strip().splitlines()[-1]
            self.note("bloquant", f"Exception Python : {last[:90]}", e[-2500:], steps=steps)

    def ok(self, what: str):
        self.checked.append(what)
        print("OK", what, flush=True)

    def check(self, cond: bool, severity, title, detail="", steps=None, shot=True):
        if cond:
            return True
        self.note(severity, title, detail, shot=shot, steps=steps)
        return False


def guard(ex, notes: Notes, title: str, fn, *a, **k):
    """A step that may raise: its exception is an anomaly (once per title)."""
    import traceback

    from harness import RunsBlocked
    try:
        return fn(*a, **k)
    except RunsBlocked:
        return None
    except Exception as e:  # noqa: BLE001
        notes.note("bloquant", f"Exception : {title}", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2500:]}")
        return None


def look(ex, notes: Notes, where: str, root: QWidget | None = None, steps=None):
    """The checks of every screen: overlaps, cut texts, raw words, in the panel (or ``root``)."""
    root = root or panel(ex)
    ov = overlaps(root)
    if ov:
        notes.note("mineur", f"Textes superposés : {where}", "Widgets dessinés l'un sur l'autre : " +
                   "; ".join(f"{a} / {b}" for a, b in ov[:6]), steps=steps)
    cut = cut_texts(root)
    if cut:
        notes.note("mineur", f"Texte coupé sans infobulle : {where}", "; ".join(cut[:6]), steps=steps)
    try:
        sov = scene_overlaps(ex) if ex.w is not None and ex.w.project is not None and ex.w.graph is not None else []
    except RuntimeError:
        sov = []
    for a, b in sov[:4]:
        notes.note("mineur", f"Graphe : « {a.strip()} » et « {b.strip()} » se chevauchent",
                   f"Vu dans : {where}. Tous les chevauchements : " + "; ".join(f"« {x} » / « {y} »" for x, y in sov[:6]),
                   steps=steps)
    raw = raw_words(texts(root))
    if raw:
        notes.note("gênant", f"Mot interne ou erreur brute affiché : {where}", " | ".join(raw[:6]), steps=steps)


KEY_MISSES: list[str] = []


def undo(ex, redo: bool = False) -> str:
    """Ctrl+Z (Ctrl+Y) as the user presses it after a click in the graph: the key sent to the graph's viewport; when
    the key changed nothing while there was something to undo, Edit > Undo (Redo) of the menu. Returns how it went:
    "key", "menu" or "nothing"."""
    from PySide6.QtTest import QTest
    s = ex.s
    stack = s._redo if redo else s._undo
    n = len(stack)
    if not n:
        return "nothing"
    ex.w.activateWindow()                  # offscreen, a closed context menu leaves no active window (harness)
    ex.settle(2)
    vp = ex.view.viewport()
    vp.setFocus()
    QTest.keyClick(vp, Qt.Key_Y if redo else Qt.Key_Z, Qt.ControlModifier)
    wait(ex, 0.8)
    if len(stack) < n:
        return "key"
    KEY_MISSES.append("redo" if redo else "undo")
    (ex.w.a_redo if redo else ex.w.a_undo).trigger()
    wait(ex, 0.8)
    return "menu" if len(stack) < n else "nothing"


class Dispatch:
    """Every modal dialog answered by its class: ``handlers[class name](d)``; it stays armed until ``stop()``. A
    dialog of a class without handler is recorded and cancelled."""

    def __init__(self, ex, **handlers):
        self.ex = ex
        self.handlers = handlers
        self.seen: list[tuple[str, str, list]] = []          # (class, title, the dialogs visible at that moment)
        self.on = True
        ex.dialog_handler = self

    def __call__(self, d):
        others = [x.__class__.__name__ for x in QApplication.topLevelWidgets()
                  if x.isVisible() and x is not d and x.__class__.__name__ not in ("Main", "QMenu")
                  and x.inherits("QDialog")]
        self.seen.append((d.__class__.__name__, d.windowTitle(), others))
        h = self.handlers.get(d.__class__.__name__)
        if self.on:
            self.ex.dialog_handler = self
        try:
            if h is None:
                d.reject()
            else:
                h(d)
        except Exception as e:  # noqa: BLE001
            print("handler failed:", d.__class__.__name__, e, flush=True)
            try:
                d.reject()
            except RuntimeError:
                pass

    def stop(self):
        self.on = False
        if self.ex.dialog_handler is self:
            self.ex.dialog_handler = None


def app_quit(ex):
    try:
        ex.close()
    except Exception as e:  # noqa: BLE001
        print("close failed:", e)
    QApplication.processEvents()
