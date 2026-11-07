"""Round 2 fuzz: Run to here (Shift+F5) on a project without any frame, replayed twice, then noted."""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
seen = []
shot = None
for k in range(2):
    fz.start({"template": "hoo", "lang": "fr"}, f"nr{k}")
    ex.key(Qt.Key_F5, ex.w, Qt.NoModifier)
    ex.settle(8)
    f5 = ([d.__class__.__name__ for d in ex.dialogs()], ex.w.status.text())
    ex.close_dialogs()
    it = next(x for x in fz.items() if x.node.key == "composition")
    ex.click_block(it)
    ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
    ex.settle(8)
    ds = [d for d in ex.dialogs() if d.__class__.__name__ == "RunDialog"]
    texts = [x.text() for x in ds[0].findChildren(QLabel) if x.isVisible() and x.text()] if ds else []
    disk = next((t for t in texts if "Go" in t or "GB" in t), "")
    if ds and k == 1:
        shot = ex.shot("run_to_here_no_frames", ds[0])
    seen.append((f5, bool(ds), disk))
    ex.close_dialogs()
print(seen, flush=True)
if all(s[1] and not s[0][0] for s in seen):
    ex.note("mineur", "Jusqu'ici (Maj+F5) ouvre la fenêtre d'exécution sur un projet sans brute",
            "Projet fait avec le modèle HOO, aucune brute ajoutée. F5 ne lance rien et la barre d'état dit "
            f"« {seen[-1][0][1]} ». Maj+F5 (Jusqu'ici) sur le bloc Recombiner ouvre au contraire la fenêtre "
            "d'exécution : Exécuter y est grisé avec la raison (« Rien à calculer : ajouter brutes ou image »), mais "
            f"elle annonce « {seen[-1][2]} » pour un calcul qui n'a rien à lire. Les deux gestes ne se comportent "
            "pas pareil pour la même cause. À noter aussi dans cette fenêtre : la liste montre « Sortie « Ha » » mais "
            "pas la sortie d'OIII. Reproduit 2/2.",
            shot=shot or True, steps=["nouveau projet, modèle « HOO » (sans brutes)", "F5 : rien ne s'ouvre",
                                      "clic sur le bloc « Recombiner », puis Maj+F5"])
os._exit(0)
