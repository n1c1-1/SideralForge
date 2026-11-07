"""Round 2 fuzz: two Image blocks placed without a file are drawn as one (replayed twice in explore2_fuzz_probe.py);
here once more with what Delete and a chosen file do on that block, then noted."""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore2_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

fz = G.Fuzz2("fr")
ex = fz.ex
res = []
for k in range(2):
    fz.start({"kind": "blank", "lang": "fr"}, f"i2n_{k}")
    fz.activate()
    for _ in range(2):
        ex.double_click_card("image")
        fz.wait_ready()
    it = next(x for x in fz.items() if x.node.key == "image")
    ex.click_block(it)
    ex.key(Qt.Key_Delete)
    fz.wait_ready()
    after_del = [w.name for w in ex.s.p.workflows]
    toast = ex.view.toast.text() if ex.view.toast.isVisible() else ""
    res.append((after_del, toast))
print(res, flush=True)
fz.start({"kind": "blank", "lang": "fr"}, "i2n_shot")
for _ in range(2):
    ex.double_click_card("image")
    fz.wait_ready()
shot = ex.shot("two_images_one_block")
ex.note("gênant", "Deux Images posées sans fichier : un seul bloc Image pour les deux flux",
        "Page vierge (ou projet d'une session) : double-clic deux fois sur la carte Image. Le projet a bien deux "
        "workflows d'image (« Image » et « Image 2 », chacun avec sa « Sortie »), mais le graphe ne dessine qu'un bloc "
        "Image, relié aux deux sorties ; son panneau dit « pour « Image 2 » ». On ne peut donc ni choisir le fichier "
        f"de « Image » à part, ni le supprimer seul : Suppr sur ce bloc laisse {res[-1][0] or 'aucun workflow'} "
        f"(message : « {res[-1][1]} »). Une fois un fichier choisi, ce sont deux sources différentes. Reproduit 2/2 "
        "(et sur un projet de JPEG en trop, d'où la première vue par le recheck).",
        shot=shot, steps=["page vierge", "double-clic sur la carte « Image » de la palette", "encore une fois"])
os._exit(0)
