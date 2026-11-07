"""Round 2 fuzz: for every block whose panel has an edit field and a Delete button: a value being typed (the field
keeps the focus), then Delete in the panel (or Ctrl+Z, or the block's menu) — does the field write into a block gone?

    python explore2_fuzz_focus.py
"""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit, QPushButton  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
DEL = ("Supprimer", "Delete")
results = []
import itertools
CNT = itertools.count()
for k in range(2):
    for st in ({"template": "classic_dn", "lang": "fr"}, {"template": "hoo", "lang": "fr"},
               {"kind": "session+image", "lang": "fr"}, {"kind": "two_flows", "lang": "fr"},
               {"template": "image_colours", "lang": "fr"}):
        fz.start(st, f"fo{k}_{next(CNT)}")
        keys = [(x.node.key, fz.title(x)) for x in fz.items()]
        for key, title in keys:
            for how in ("panel", "undo_btn"):
                print("START", k, st, title, how, flush=True)
                fz.start(st, f"fo{k}_{next(CNT)}")
                print("STARTED", flush=True)
                fz.errs.clear()
                fz.activate()
                it = next((x for x in fz.items() if x.node.key == key and fz.title(x) == title), None)
                if it is None:
                    continue
                ex.click_block(it)
                print("CLICKED", flush=True)
                pn = ex.page.panel
                eds = [w for w in pn.findChildren(QAbstractSpinBox) + pn.findChildren(QLineEdit)
                       if w.isVisible() and w.isEnabled() and not isinstance(w.parent(), QAbstractSpinBox)]
                dels = [b for b in pn.findChildren(QPushButton) if b.isVisible() and b.isEnabled() and b.text() in DEL]
                if not eds or (how == "panel" and not dels):
                    continue
                e = eds[0]
                print("CASE", k, st, title, type(e).__name__, how, flush=True)
                ex.click(e)
                e.setFocus()
                if isinstance(e, QAbstractSpinBox):
                    QTest.keyClick(e, Qt.Key_Up)
                else:
                    QTest.keyClicks(e, "x")
                ex.settle()
                if how == "panel":
                    ex.click(dels[0])
                else:                                    # Edit menu's Undo, the field still holding the focus
                    acts = [a for a in ex.w.menuBar().actions()]
                    undo = next((a2 for a in acts if a.menu() is not None for a2 in a.menu().actions()
                                 if a2.text().replace("&", "").split("\t")[0] in ("Annuler", "Undo")), None)
                    if undo is None or not undo.isEnabled():
                        continue
                    undo.trigger()
                fz.wait_ready()
                ex.settle(6)
                errs = [x.strip().splitlines()[-1] for x in fz.errs]
                fz.errs.clear()
                what = f"{st.get('template') or st.get('kind')} · {title} · {type(e).__name__} · {how}"
                results.append((k, what, errs))
                if errs:
                    print("ERR", k, what, errs, flush=True)
print("done", len(results), "cases,", sum(1 for r in results if r[2]), "with errors", flush=True)
for r in results:
    print(r, flush=True)
os._exit(0)
