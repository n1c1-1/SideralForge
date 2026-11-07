"""Round 3 (control) block_actions: the new points, each replayed twice here, then noted with its screenshot."""
from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(600, exit=True)

from explore2_blocks_common import SHORT, TOASTS, EXC, G, Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402
from astro_gpu.client.graph import WireItem  # noqa: E402

ex = Explorer("block_actions", lang="fr")
g = G(ex)


def key_on(it, k):
    ex.click_block(it)
    g.act()
    n = len(TOASTS)
    ex.key(k)
    ex.settle(4)
    return TOASTS[n:]


seen = {"astro_d": [], "fusion_d": [], "join": [], "port": []}
shots = {}
for rep in range(2):
    g.mono_fusion()
    t = key_on(g.blk("astrometrie", pid=g.pid_of("Ha")), Qt.Key_D)
    seen["astro_d"].append(t)
    if rep == 1:
        shots["astro_d"] = ex.shot("r3_d_astrometrie_canal")
    t = key_on(g.blk("fusion"), Qt.Key_D)
    seen["fusion_d"].append(t)
    if rep == 1:
        shots["fusion_d"] = ex.shot("r3_d_fusion")
    g.new("session", ["calibration", "registration", "stacking"])
    n = len(TOASTS)
    ex.double_click_card("session")
    ex.settle(8)
    shown = [x.text() for x in ex.view.findChildren(QLabel) if x.isVisible() and x.text()]
    seen["join"].append((TOASTS[n:], shown[:2]))
    if rep == 1:
        shots["join"] = ex.shot("r3_join_message")
    g.new("session", SHORT)
    w = [x for x in ex.view.scene().items() if isinstance(x, WireItem) and x.isVisible()][0]
    pt = ex.view.mapFromScene(w.path().pointAtPercent(0.5))
    ex.hover(ex.view.viewport(), pt)
    ex.click(ex.view.viewport(), pt)
    g.act()
    ex.key(Qt.Key_Delete)
    ex.settle(4)
    n = len(TOASTS)
    ex.pull_port_to_post(g.blk("couleur"))
    ex.settle(4)
    seen["port"].append((TOASTS[n:], g.blk("final").summary))
    if rep == 1:
        shots["port"] = ex.shot("r3_port_couleur_post")
print(seen, flush=True)

if all(t == ["nécessaire à « HOO »"] for t in seen["astro_d"]):
    ex.note("mineur", "D sur l'Astrométrie d'un canal de Fusion : le message n'a pas de sujet",
            "Projet mono Ha + OIII avec Fusion HOO. Cliquer l'Astrométrie du workflow Ha, touche D : le refus "
            "s'affiche seulement « nécessaire à « HOO » », sans dire quel bloc est gardé ni pourquoi en entier ; Suppr "
            "sur le même bloc dit bien « Astrométrie gardée : nécessaire à « HOO » ».\n(reproduit 2 fois)",
            shot=shots.get("astro_d", True),
            steps=["Projet mono, Astrométrie dans Ha et OIII, Fusion", "Cliquer l'Astrométrie de Ha, D"])
if all(t and "source" in t[0] for t in seen["fusion_d"]):
    ex.note("mineur", "D sur la Fusion : « c'est une source », alors que la Fusion lit les canaux",
            f"Projet mono Ha + OIII avec Fusion HOO. Cliquer la Fusion, touche D : « {seen['fusion_d'][-1][0]} ». "
            "La Fusion n'est pas une source (elle lit Ha et OIII) ; la raison donnée est fausse. Pour le Recombiner, "
            "le message explique bien (« masquer son calque dans Post, ou Suppr le supprime »).\n(reproduit 2 fois)",
            shot=shots.get("fusion_d", True), steps=["Projet mono, Astrométrie dans Ha et OIII, Fusion",
                                                     "Cliquer la Fusion, D"])
if all(any("rejoint" in x for x in j[0]) and not any("rejoint" in x for x in j[1]) for j in seen["join"]):
    ex.note("mineur", "Deuxième Session posée : « Session 2 rejoint Workflow » aussitôt remplacé par le coût",
            "Projet à une session (Calibration, Alignement, Empilement). Double-clic sur la carte Session : la Session 2 "
            "rejoint bien le workflow, et le message « « Session 2 » rejoint « Workflow » · Ctrl+Z » est émis, mais il "
            f"est remplacé tout de suite par « {seen['join'][-1][0][-1]} » : on ne voit que le coût, pas ce qui a été "
            "fait. (Sur un projet à deux flux, où le coût ne change pas, le message reste.)\n(reproduit 2 fois)",
            shot=shots.get("join", True), steps=["Projet à une session + Calibration, Alignement, Empilement",
                                                  "Double-clic sur la carte Session de la palette"])
if all(not p[0] and p[1].startswith("0") for p in seen["port"]):
    ex.note("mineur", "Fil vers Post retiré : tirer le port du dernier bloc sur Post ne fait rien, sans un mot",
            "Chaîne jusqu'à Couleur. Cliquer le fil Couleur → Post, Suppr (« « Workflow » hors Post »). Tirer ensuite "
            "le port de Couleur sur le bloc Post : rien ne se passe et rien ne s'affiche (Post reste à « 0 calque »). "
            "Le même geste depuis Gradient ajoute bien un calque, et le sous-menu « Workflow » de Couleur → « Vers "
            "Post » rend le fil. Le geste naturel est muet.\n(reproduit 2 fois)",
            shot=shots.get("port", True), steps=["Chaîne jusqu'à Couleur", "Cliquer le fil vers Post, Suppr",
                                                  "Tirer le port de Couleur sur Post"])
for e in EXC:
    print("EXC", e[-500:])
ex.close()
