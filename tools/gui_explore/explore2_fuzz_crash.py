"""Round 2 fuzz: the process crash seen in explore2_fuzz_focus.py, replayed alone, step by step (prints before each).

    python explore2_fuzz_crash.py <template> <block title> [n]
"""
import os
import sys
from pathlib import Path

TPL, TITLE = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
for k in range(N):
    print("start", k, flush=True)
    fz.start({"template": TPL, "lang": "fr"}, f"cr{k}")
    print("undo stack at start:", len(ex.s._undo), [x.get("label") if isinstance(x, dict) else type(x).__name__
                                                     for x in ex.s._undo][-5:], flush=True)
    fz.activate()
    it = next(x for x in fz.items() if fz.title(x) == TITLE)
    ex.click_block(it)
    pn = ex.page.panel
    eds = [w for w in pn.findChildren(QAbstractSpinBox) + pn.findChildren(QLineEdit)
           if w.isVisible() and w.isEnabled() and not isinstance(w.parent(), QAbstractSpinBox)]
    e = eds[0]
    print("field", type(e).__name__, e.toolTip()[:60], flush=True)
    ex.click(e)
    e.setFocus()
    QTest.keyClick(e, Qt.Key_Up)
    ex.settle()
    print("typed; undo stack:", len(ex.s._undo), flush=True)
    undo = next((a2 for a in ex.w.menuBar().actions() if a.menu() is not None for a2 in a.menu().actions()
                 if a2.text().replace("&", "").split("\t")[0] in ("Annuler", "Undo")), None)
    print("Edit > Annuler", undo is not None and undo.isEnabled(), flush=True)
    undo.trigger()
    print("triggered", flush=True)
    fz.wait_ready()
    ex.settle(6)
    print("settled; errs", [x.strip().splitlines()[-1] for x in fz.errs], flush=True)
    fz.errs.clear()
    ex.shot(f"crash_{TPL}_{k}")
print("END OK", flush=True)
os._exit(0)
