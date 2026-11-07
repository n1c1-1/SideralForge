"""Round 2 fuzz: "Delete workflow" on a workflow that feeds a Fusion (seed 2010 of explore2_fuzz.py, shrunk by hand),
replayed twice on several templates; with "note", noted.

    python explore2_fuzz_note_delwf.py [note]
"""
import os
import sys
import traceback
from pathlib import Path

NOTE = len(sys.argv) > 1 and sys.argv[1] == "note"
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
out = []
shot = None
for k in range(2):
    for tpl in ("lrgb", "sho", "hoo", "rgb"):
        fz.start({"template": tpl, "lang": "fr"}, f"dw{k}_{tpl}")
        fz.errs.clear()
        it = next((x for x in fz.items() if fz.title(x) == "Sortie"), None)
        if it is None:
            out.append((k, tpl, "pas de Sortie"))
            continue
        wf = it.node.paths[0].name if it.node.paths else "?"
        n0 = len(ex.s.p.workflows)
        ex.right_click_block(it, pick="Supprimer workflow")
        fz.wait_ready()
        try:
            ex.s.p.validate()
            valid = "valide"
        except Exception as e:  # noqa: BLE001
            valid = f"invalide: {e}"
        errs = [x.strip().splitlines()[-1] for x in fz.errs]
        fz.errs.clear()
        toast = ex.view.toast.text() if ex.view.toast.isVisible() else ""
        if k == 1 and tpl == "lrgb":
            shot = ex.shot("delete_workflow_feeding_fusion")
        fz.undo()
        fz.wait_ready()
        u_errs = [x.strip().splitlines()[-1] for x in fz.errs]
        fz.errs.clear()
        out.append((k, tpl, wf, n0, len(ex.s.p.workflows), valid, toast, errs, "undo:", u_errs))
for o in out:
    print(o, flush=True)
bad = [o for o in out if len(o) > 5 and o[5] != "valide"]
if NOTE and bad and all(any(o[1] == b[1] for b in bad if b[0] == 1) for o in bad if o[0] == 0):
    tpls = sorted({o[1] for o in bad})
    ok = sorted({o[1] for o in out if len(o) > 5 and o[5] == "valide"})
    ex.note("bloquant", "« Supprimer workflow » sur un canal d'une Fusion casse le projet",
            "Projet fait avec le modèle LRVB : clic droit sur le premier bloc « Sortie » (le workflow L, qui alimente "
            "la Fusion LRVB) → « Supprimer workflow ». Le workflow part, mais la Fusion le lit toujours : la "
            "définition du projet devient invalide (« la fusion lit des workflows que le projet n'a pas »), plus "
            "aucun plan ni contrôle ne se calcule ; Ctrl+Z rend le workflow, mais Ctrl+Y (le refaire) lève une erreur "
            "Python (vu par le fuzz). Le message dit « Workflow « L » supprimé · Ctrl+Z ». C'est la même "
            "cause que la session retirée du premier tour (corrigée pour la session : la Fusion garde un canal "
            "vide), mais par le menu du workflow. "
            f"Cassé avec les modèles : {', '.join(tpls)} ; sans problème : {', '.join(ok) or 'aucun'}. Vu par le fuzz "
            "(graine 2010). Reproduit 2/2.",
            shot=shot or True, steps=["nouveau projet, modèle « LRVB »",
                                      "clic droit sur le premier bloc « Sortie » → « Supprimer workflow »"])
os._exit(0)
