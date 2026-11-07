"""Round 4, area clavier_fenetre, axis 2: window sizes, live resize, Windows scaling (QT_SCALE_FACTOR in a new process).
Prints the suspects; screenshots in exploration4/clavier_fenetre/win_<tag>_*.png.

    python explore5_clavier_window.py W H [scale] [lang]
    (scale: set QT_SCALE_FACTOR before the harness starts)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

W, H = int(sys.argv[1]), int(sys.argv[2])
SCALE = sys.argv[3] if len(sys.argv) > 3 else ""
LANG = sys.argv[4] if len(sys.argv) > 4 else "fr"
if SCALE:
    os.environ["QT_SCALE_FACTOR"] = SCALE
sys.path.insert(0, str(Path(__file__).parent))
from explore_post_common import P, start  # noqa: E402
from explore2_blocks_common import G, Explorer  # noqa: E402
from PySide6.QtCore import QPoint, QRect, Qt  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QAbstractScrollArea, QApplication, QComboBox, QLabel, QLineEdit,  # noqa: E402
                               QScrollArea, QWidget)

start(1200)
TAG = f"{W}x{H}" + (f"_s{SCALE}" if SCALE else "") + ("" if LANG == "fr" else f"_{LANG}")
ex = Explorer("clavier_fenetre", lang=LANG, size=(W, H))
g = G(ex)
pp = P(ex)
CHAIN = ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color", "stars"]


def in_scroll(x: QWidget) -> bool:
    p = x.parentWidget()
    while p is not None:
        if isinstance(p, QAbstractScrollArea):
            return True
        p = p.parentWidget()
    return False


def text_of(x):
    try:
        return x.text()
    except Exception:  # noqa: BLE001
        return ""


def check(root: QWidget, where: str):
    """Widgets outside the window (not in a scroll area), cut texts without tooltip, overlapping texts."""
    out = []
    R = root.rect()
    vis = [x for x in root.findChildren(QWidget) if x.isVisible() and x.width() > 0 and x.window() is root.window()]
    for x in vis:
        r = QRect(x.mapTo(root, QPoint(0, 0)), x.size())
        if not in_scroll(x) and not R.contains(r) and r.intersects(R) is not None:
            if r.right() > R.right() + 1 or r.bottom() > R.bottom() + 1 or r.left() < -1 or r.top() < -1:
                out.append(f"OUTSIDE {type(x).__name__} {text_of(x)[:40]!r} {r.getRect()} in {R.getRect()}")
        t = text_of(x)
        if isinstance(x, (QLabel, QAbstractButton)) and t and "\n" not in t and not t.startswith("<"):
            fm = x.fontMetrics()
            need = fm.horizontalAdvance(t.replace("&&", "&"))
            have = x.contentsRect().width() - (16 if isinstance(x, QAbstractButton) else 0)
            wrap = isinstance(x, QLabel) and x.wordWrap()
            if not wrap and need > have + 2 and x.pixmap().isNull() if isinstance(x, QLabel) else need > have + 2:
                if isinstance(x, QLabel) and wrap:
                    pass
                else:
                    # visible part of the widget: is it clipped by a scroll viewport or parent?
                    out.append(f"CUT {type(x).__name__} {t[:60]!r} need {need} have {have} tip={bool(x.toolTip())}")
        if isinstance(x, QLabel) and x.wordWrap() and t:
            if x.heightForWidth(x.width()) > x.height() + 4:
                out.append(f"CUTH {type(x).__name__} {t[:60]!r} h {x.height()} needs {x.heightForWidth(x.width())}")
        if isinstance(x, QComboBox) and x.currentText():
            need = x.fontMetrics().horizontalAdvance(x.currentText()) + 28
            if need > x.width() + 2:
                out.append(f"CUT combo {x.currentText()[:50]!r} need {need} have {x.width()}")
        if isinstance(x, QScrollArea):
            wd = x.widget()
            if wd is not None and (wd.height() > x.viewport().height() + 2) and \
                    x.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff:
                out.append(f"NOSCROLL {x.objectName()} content {wd.height()} viewport {x.viewport().height()}")
    # texts overlapping texts (siblings or not), on the window
    boxes = []
    for x in vis:
        if isinstance(x, (QLabel, QAbstractButton, QLineEdit, QComboBox)) and (text_of(x) or isinstance(x, QComboBox)):
            boxes.append((QRect(x.mapTo(root, QPoint(0, 0)), x.size()), x))
    for i, (a, xa) in enumerate(boxes):
        for b, xb in boxes[i + 1:]:
            if xa.isAncestorOf(xb) or xb.isAncestorOf(xa):
                continue
            inter = a.intersected(b)
            if inter.width() > 4 and inter.height() > 4:
                out.append(f"OVERLAP {type(xa).__name__} {text_of(xa)[:30]!r} {a.getRect()} x {type(xb).__name__} "
                           f"{text_of(xb)[:30]!r} {b.getRect()}")
    print(f"--- {where}: window {root.width()}x{root.height()} (min hint {root.minimumSizeHint().width()}x"
          f"{root.minimumSizeHint().height()})", flush=True)
    for line in dict.fromkeys(out):
        print("   ", line, flush=True)
    return out


def dialog_probe(name):
    res = {}

    def h(d):
        ex.settle(4)
        res["size"] = (d.width(), d.height())
        res["min"] = (d.minimumSizeHint().width(), d.minimumSizeHint().height())
        res["title"] = d.windowTitle()
        res["issues"] = check(d, f"dialog {name} {d.windowTitle()}")
        d.grab().save(str(ex.out / f"win_{TAG}_dlg_{name}.png"))
        d.reject()
    return res, h


root = ex.project("session_empty_wf")
ex.open()
ex.shot(f"win_{TAG}_home")
check(ex.w, "Home")
ex.open(root)
g.act()
g.place(*CHAIN)
print("actual window", ex.w.width(), ex.w.height(), "requested", W, H)
for key in ("sources", "empilement", "etoiles", "composition"):
    it = g.blk(key)
    if it is None:
        continue
    ex.click_block(it)
    ex.settle(6)
    check(ex.w, f"graph + panel {key}")
    ex.shot(f"win_{TAG}_graph_{key}")
# Inspect
ex.click_block(g.blk("fond"))
pp.key(Qt.Key_Return)
ex.settle(10)
check(ex.w, "Inspect Gradient")
ex.shot(f"win_{TAG}_inspect")
print("after Inspect window", ex.w.width(), ex.w.height())
# Post page
ex.w.set_mode(2)
ex.settle(12)
check(ex.w, "Post")
ex.shot(f"win_{TAG}_post")
print("after Post window", ex.w.width(), ex.w.height())
ex.w.set_mode(0)
ex.settle(6)
# Log open
ex.w.b_log.setChecked(True)
ex.settle(6)
check(ex.w, "graph + Log")
ex.shot(f"win_{TAG}_log")
ex.w.b_log.setChecked(False)
ex.settle(4)
# dialogs
for name, fn in (("settings", lambda: ex.w.open_settings()), ("about", lambda: ex.w.about()),
                 ("rename", lambda: g.menu(g.blk("couleur"), "Renommer" if LANG == "fr" else "Rename"))):
    res, h = dialog_probe(name)
    ex.dialog_handler = h
    fn()
    ex.settle(10)
    print(f"DIALOG {name}: {res.get('title')!r} size {res.get('size')} min {res.get('min')} vs window {W}x{H}")
# the run window (a blocked run: the start fails; whatever shows)
pp.key(Qt.Key_F5, ex.w)
ex.settle(10)
check(ex.w, "after F5")
ex.shot(f"win_{TAG}_after_f5")
# live resize: big, small, big
for size in ((2560, 1440), (1280, 720), (1024, 640), (W, H)):
    ex.w.resize(*size)
    ex.settle(8)
    print(f"RESIZE to {size}: got {ex.w.width()}x{ex.w.height()}")
    if size == (1024, 640):
        ex.shot(f"win_{TAG}_resized_1024")
check(ex.w, "after live resize back")
ex.shot(f"win_{TAG}_resized_back")
ex.close()
