"""Round 2 fuzz: the first Ctrl+Z after a project is made from a template, replayed twice, then noted (question)."""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
out = []
for k in range(2):
    for tpl in ("classic_dn", "sho", "image"):
        fz.start({"template": tpl, "lang": "fr"}, f"tu{k}_{tpl}")
        n_undo = len(ex.s._undo)
        b0 = len(fz.items())
        fz.undo()
        fz.wait_ready()
        a = ([w.name for w in ex.s.p.workflows], [s.name for s in ex.s.p.sessions], [fz.title(x) for x in fz.items()],
             ex.w.status.text())
        if k == 1 and tpl == "classic_dn":
            ex.shot("template_undone")
        fz.redo()
        fz.wait_ready()
        out.append((tpl, n_undo, b0, a, len(fz.items())))
for o in out:
    print(o, flush=True)
if all(o[1] >= 1 and not o[3][0] for o in out):
    ex.note("question", "Le premier Ctrl+Z après la création défait tout le modèle",
            "Nouveau projet fait avec un modèle (vu avec « Classique + débruitage », « SHO » et « Image ») : le "
            "premier Ctrl+Z retire tout ce que le modèle a posé, workflows et sessions compris, et laisse une page "
            f"vide (« {out[-3][3][3]} »). Ctrl+Y le rend. La fenêtre des modèles ne revient pas (elle ne vient qu'à "
            "la création). À juger : la création depuis un modèle doit-elle être une étape qu'on annule ? Reproduit "
            "2/2.",
            shot=str(ex.out / "template_undone.png"),
            steps=["nouveau projet, modèle « Classique + débruitage »", "Ctrl+Z"])
os._exit(0)
