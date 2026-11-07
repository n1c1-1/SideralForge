"""Round 3 (control) fuzz: Shift+F5 without frames (texts of the run window), and the new points replayed twice here,
then noted with their screenshot."""
import os
import sys
import traceback
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_blocks_lib import TOASTS, overlaps  # noqa: E402
import explore3_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QPushButton  # noqa: E402

fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex
sys.excepthook = sys.__excepthook__


def blk(pred):
    return next((x for x in fz.items() if pred(x)), None)


ov, fc = [], []
shots = {}
try:
    for k in range(2):
        fz.start({"template": "hoo", "lang": "fr"}, f"n3_{k}")
        ex.click_block(blk(lambda x: x.node.key == "composition"))
        ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
        ex.settle(8)
        ds = [d for d in ex.dialogs() if d.__class__.__name__ == "RunDialog"]
        if ds:
            texts = [x.text() for x in ds[0].findChildren(QLabel) if x.isVisible() and x.text()]
            run = [(b.text(), b.isEnabled(), b.toolTip()[:100]) for b in ds[0].findChildren(QPushButton) if b.isVisible()]
            print(k, "A3 run window texts:", [t[:100] for t in texts if "Go" in t or "GB" in t or "brute" in t
                                              or "calcul" in t or "Sortie" in t], "| buttons", run, flush=True)
            if k == 1:
                ex.shot("r3_run_to_here_dialog", ds[0])
        else:
            print(k, "A3 no run window; status", ex.w.status.text(), TOASTS[-1:], flush=True)
        ex.close_dialogs()
        # overlap: a channel block's image to Post
        fz.start({"template": "hoo", "lang": "fr"}, f"n_ov_{k}")
        fz.activate()
        ex.right_click_block(blk(lambda x: x.node.key == "astrometrie"), pick="Vers Post")
        fz.wait_ready()
        ov.append((TOASTS[-1:], overlaps(ex)))
        if k == 1:
            shots["ov"] = ex.shot("r3_overlap_hoo")
        # the Fusion card on a colour project
        fz.start({"template": "classic_dn", "lang": "fr"}, f"n_fc_{k}")
        reason = ex.card("fusion").reason
        ex.double_click_card("fusion")
        fz.wait_ready()
        fu = blk(lambda x: x.node.key == "fusion")
        fc.append((reason, TOASTS[-1:], fu.summary if fu else None, fu.toolTip() if fu else ""))
        if k == 1:
            shots["fc"] = ex.shot("r3_fusion_on_colour")
    print("OV", ov, "\nFC", fc, flush=True)
    if all(o[1] for o in ov):
        ex.note("gênant", "Envoyer au Post l'image d'un bloc d'un canal de Fusion : les blocs des canaux se superposent",
                "Projet fait avec le modèle HOO. Clic droit sur l'Astrométrie du canal Ha → « Vers Post » : le calque "
                f"est bien fait ({ov[-1][0][0] if ov[-1][0] else ''}), mais le graphe redessine les deux canaux l'un "
                f"sur l'autre : {', '.join(ov[-1][1][:4])} se chevauchent (on ne voit plus qu'une colonne de blocs). "
                "Même chose avec SHO (Gradient du canal SII → « Vers Post ») et avec LRVB en tirant le port de "
                "Débruitage sur Post (vu par le fuzz, graines 2010, 2033, 2051, 4036, 4069, séquence réduite à ce "
                "seul geste). Sur un projet couleur (« Classique + débruitage »), le même geste ne superpose rien.\n"
                "(reproduit 2 fois)",
                shot=shots.get("ov", True), steps=["nouveau projet, modèle « HOO »",
                                                    "clic droit sur l'Astrométrie de Ha → « Vers Post »"])
    if all(not f[0] and f[2] for f in fc):
        ex.note("gênant", "Projet couleur : la carte Fusion n'est pas grisée et pose une Fusion HOO aux canaux absents",
                "Projet fait avec le modèle « Classique + débruitage » (une session couleur, aucun filtre ; pareil avec "
                "« Complet »). La carte Fusion de la palette n'est pas grisée. Double-clic : « "
                f"{fc[-1][1][0] if fc[-1][1] else ''} » ; le bloc dit « {fc[-1][2]} » alors que le projet n'a ni "
                "session Ha ni OIII ; son infobulle dit « Ne peut pas s'exécuter : Ha, OIII : sans brutes ». Un "
                "Étoiles/fond a été ajouté au workflow couleur. La règle : la carte reste grisée tant que chaque canal "
                "n'a pas ce qu'il faut, avec quoi faire.\n(reproduit 2 fois)",
                shot=shots.get("fc", True), steps=["nouveau projet, modèle « Classique + débruitage »",
                                                    "double-clic sur la carte « Fusion » de la palette"])
except Exception:
    traceback.print_exc()
os._exit(0)
