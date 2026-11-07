"""Round 2 fuzz: the Fusion deleted from its panel while one of its weights has the keyboard focus (seed 1027 of
explore2_fuzz.py, shrunk by hand), replayed twice, then noted.

    python explore2_fuzz_note_fusion.py [note]
"""
import os
import sys
from pathlib import Path

NOTE = len(sys.argv) > 1 and sys.argv[1] == "note"
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractSpinBox, QPushButton  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
out = []
for k in range(2):
    fz.start({"template": "hoo", "lang": "fr"}, f"nf{k}")
    fz.errs.clear()
    fz.activate()
    it = next(x for x in fz.items() if x.node.key == "fusion")
    ex.click_block(it)
    pn = ex.page.panel
    spins = [w for w in pn.findChildren(QAbstractSpinBox) if w.isVisible()]
    dels = [b for b in pn.findChildren(QPushButton) if b.isVisible() and b.text() in ("Supprimer", "Delete")]
    if not spins or not dels:
        out.append(("no spin/del", [b.text() for b in pn.findChildren(QPushButton) if b.isVisible()]))
        continue
    sp = spins[0]
    ex.click(sp)                                    # the cursor in the weight, as when the user types a value
    sp.setFocus()
    QTest.keyClick(sp, Qt.Key_Up)
    ex.settle()
    print("focus before:", type(ex.app.focusWidget()).__name__, flush=True)
    ex.click(dels[0])                               # then Supprimer at the top of the panel
    fz.wait_ready()
    ex.settle(6)
    errs = list(fz.errs)
    fz.errs.clear()
    out.append((len(errs), [e.strip().splitlines()[-1] for e in errs]))
    if k == 1 and errs and NOTE:
        shot = ex.shot("fusion_delete_with_focus")
print(out, flush=True)
if NOTE and all(o[0] for o in out):
    ex.note("gênant", "Supprimer la Fusion depuis son panneau, un poids en cours de saisie : erreur Python",
            "Projet HOO : clic sur le bloc Fusion, le curseur dans un de ses poids (on vient d'en changer la valeur), "
            "puis « Supprimer » en haut du panneau : la Fusion part, puis le poids, en perdant le focus, essaie "
            f"d'enregistrer sa valeur dans le workflow qui n'existe plus : erreur Python (« {out[-1][1][0]} »). "
            "Vu par le fuzz (graine 1027, le poids gardait le focus d'un geste d'avant), réduit à la main. "
            "Reproduit 2/2.",
            shot=True, steps=["nouveau projet, modèle « HOO »", "clic sur le bloc « Fusion »",
                              "clic dans le premier poids du panneau, flèche haut",
                              "clic sur « Supprimer » en haut du panneau"])
os._exit(0)
