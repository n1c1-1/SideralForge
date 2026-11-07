"""Round 2 fuzz: a field of the panel being typed in while its block goes away (Edit > Undo from the menu, or Delete
in the panel). Replayed twice on fresh projects; with "note", the grouped anomaly is written (it replaces the line of
the Fusion case noted first, same cause).

    python explore2_fuzz_note_focus.py [note]
"""
import json
import os
import sys
from pathlib import Path

NOTE = len(sys.argv) > 1 and sys.argv[1] == "note"
sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QLineEdit  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex


def menu_undo():
    a = next((a2 for a in ex.w.menuBar().actions() if a.menu() is not None for a2 in a.menu().actions()
              if a2.text().replace("&", "").split("\t")[0] in ("Annuler", "Undo")), None)
    a.trigger()


out = []
shot = None
for k in range(2):
    fz.start({"kind": "two_flows", "lang": "fr"}, f"nfo{k}")
    fz.activate()
    ex.double_click_card("session")                  # Session 3 added (one Undo entry)
    fz.wait_ready()
    it = next(x for x in fz.items() if fz.title(x) == "Session 3")
    ex.click_block(it)
    e = next(w for w in ex.page.panel.findChildren(QLineEdit) if w.isVisible() and w.text() == "Session 3")
    fz.errs.clear()
    ex.click(e)
    e.setFocus()
    e.selectAll()
    QTest.keyClicks(e, "Nuit du 12")                 # typed, not yet Enter
    ex.settle()
    menu_undo()                                      # Édition > Annuler, with the mouse
    fz.wait_ready()
    ex.settle(6)
    errs = [x.strip().splitlines()[-1] for x in fz.errs]
    fz.errs.clear()
    if k == 1:
        shot = ex.shot("focus_undo_session_name")
    # and right after making a project from a template: what the first Ctrl+Z does
    fz.start({"template": "classic_dn", "lang": "fr"}, f"nfo_t{k}")
    w0 = [w.name for w in ex.s.p.workflows]
    fz.undo()
    fz.wait_ready()
    w1 = [w.name for w in ex.s.p.workflows], [s.name for s in ex.s.p.sessions]
    out.append((errs, w0, w1))
print(out, flush=True)
if NOTE and all(o[0] for o in out):
    log = ex.log
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    old = next((x for x in lines if x["title"].startswith("Supprimer la Fusion depuis son panneau")), None)
    lines = [x for x in lines if x is not old]
    log.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    ex.note("gênant", "Un champ du panneau en cours de saisie écrit dans un bloc qui vient de partir : erreur Python",
            "Quand le bloc affiché part alors qu'un de ses champs a encore le curseur (valeur tapée, pas encore "
            "validée), le champ, en perdant le curseur, enregistre sa valeur dans une session ou un workflow qui "
            "n'existe plus : erreur Python, la valeur est perdue.\n"
            "Cas vus, chacun reproduit 2/2 : (1) deux flux, une Session ajoutée par double-clic, son nom retapé dans "
            f"le panneau, puis Édition › Annuler à la souris : « {out[-1][0][0]} » ; (2) projet HOO, clic sur la "
            "Fusion, un poids changé (flèche haut), puis « Supprimer » en haut du panneau : « KeyError: 'no workflow "
            "3' » (vu aussi deux fois par le fuzz, graines 1027 et 1064) ; (3) même chose avec le nom d'une session "
            "(Ha, OIII) ou d'un workflow (bloc « Sortie ») d'un projet fait avec un modèle, puis Édition › Annuler. "
            "Les réglages des autres blocs (Calibration, Alignement, Empilement, Astrométrie, Gradient, Couleur) "
            "n'ont pas l'erreur.",
            shot=shot or True, steps=["projet de deux flux (Session 1, Session 2)",
                                      "double-clic sur la carte « Session » de la palette (Session 3)",
                                      "clic sur le bloc « Session 3 »; dans le panneau, taper « Nuit du 12 » dans son "
                                      "nom sans Entrée", "menu Édition › Annuler"])
os._exit(0)
