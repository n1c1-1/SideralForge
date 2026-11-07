"""Words, round 2: what the user reads when something goes wrong: an image source that is not an image (a damaged
FITS, a missing file), the Open dialog's filters, the Inspect of such a block, Post. Prints the texts.

    python explore2_words_errors.py fr
"""
from __future__ import annotations

import faulthandler
import json
import sys
import time

faulthandler.dump_traceback_later(400, exit=True)
from harness import TMP, Explorer, stray_qt_errors  # noqa: E402
from PySide6.QtWidgets import QFileDialog, QLabel, QWidget  # noqa: E402

from astro_gpu.client import graph as graphmod  # noqa: E402
from astro_gpu.client import app as appmod  # noqa: E402
from astro_gpu.client.widgets import ElidedLabel  # noqa: E402
from astro_gpu.core import project2  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
stray_qt_errors()
ex = Explorer("words_layout", lang=LANG, size=(1366, 768))
SAID = []
_st = graphmod.GraphView.show_toast
graphmod.GraphView.show_toast = lambda self, t: (SAID.append(str(t)), _st(self, t))[1]
_fl = appmod.Main.flash
appmod.Main.flash = lambda self, t: (SAID.append("[barre] " + str(t)), _fl(self, t))[1]
OUT = {}
ASKS = []
_orig_open = QFileDialog.getOpenFileName


def rec_open(*a, **k):
    ASKS.append([str(x) for x in a[1:4]] + [str(v) for v in k.values()])
    return ("", "")


QFileDialog.getOpenFileName = staticmethod(rec_open)
QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (ASKS.append([str(x) for x in a[1:4]]), ([], ""))[1])


def texts(root: QWidget):
    out = []
    for x in [root] + root.findChildren(QWidget):
        if not x.isVisible():
            continue
        if isinstance(x, ElidedLabel):
            out.append(x.full_text())
        elif isinstance(x, QLabel) and x.text():
            out.append(x.text())
        if x.toolTip():
            out.append("[tip] " + x.toolTip())
    return out


def wait(sec):
    t = time.time()
    while time.time() - t < sec:
        ex.settle(3)


for case in ("damaged", "missing"):
    root = ex.project("image", f"Err {case} {LANG}")
    p = project2.load(root)
    w = p.workflows[0]
    bad = TMP / "data" / f"bad_{case}.fits"
    if case == "damaged":
        bad.write_bytes(b"SIMPLE  =                    T" + b"\0" * 3000)
    w.source = {"path": str(bad), "stretched": False}
    project2.save(p)
    ex.open(root)
    ex.settle(8)
    it = ex.block("image")
    ex.click_block(it)
    OUT[f"{case}/panel"] = texts(ex.page.panel)
    it = ex.block("image")                                 # the click redraws the graph
    OUT[f"{case}/graph_tip"] = it.toolTip() if it is not None else ""
    ex.double_click_block(ex.block("image"))
    wait(6)
    OUT[f"{case}/inspect"] = texts(ex.w.tune)
    ex.shot(f"errors_{LANG}_{case}_inspect")
    ex.w.set_mode(2)
    wait(4)
    OUT[f"{case}/post"] = texts(ex.w)
    ex.shot(f"errors_{LANG}_{case}_post")
    ex.w.set_mode(0)
    ex.settle(4)
    OUT[f"{case}/said"] = list(SAID)
    SAID.clear()
# the Open dialog of the menu bar
mb = ex.w.menuBar()
for a in mb.actions():
    if a.menu() is None:
        continue
    for b in a.menu().actions():
        t = b.text().replace("&", "")
        if t.startswith(("Ouvrir", "Open")):
            b.trigger()
            ex.settle(4)
OUT["open_dialogs"] = ASKS
OUT["popups"] = [list(p) for p in ex.popups]
(ex.out / f"errors_{LANG}.json").write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in OUT.items():
    print("==", k)
    for x in (v if isinstance(v, list) else [v]):
        print("   ", str(x)[:300].replace("\n", " / "))
ex.close()
