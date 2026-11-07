"""post_wires round 2, fifth batch: notes of what explore2_post_visible.py reproduced twice (two fresh projects)."""
from pathlib import Path

from harness import Explorer

ex = Explorer("post_wires")
OUT = ex.out
ST = ["Nouveau projet avec une session de 15 brutes", "Double-clic dans la palette : Calibration … Couleur, "
      "Étoiles/fond", "(état simulé : workflows calculés)"]
ex.note("question", "Les images des blocs du milieu arrivent masquées dans Post",
        "Image sans étoiles tirée sur Post, puis « Vers Post » sur Gradient, Empilement et Couleur : dans la page Post, "
        "« Sortie Gradient », « Empilement brut » et « Sortie Couleur » arrivent avec leur case décochée (masqués), "
        "alors que « Workflow · Image sans étoiles » et « Workflow · Recombiner » arrivent visibles. Rien ne le dit au "
        "moment de l'envoi. Voulu ?", shot=str(OUT / "visible_run2.png"),
        steps=ST + ["Tirer le ● du haut d'Étoiles/fond sur Post", "Clic droit sur Gradient > Vers Post",
                    "Tirer le port ● d'Empilement sur Post", "Onglet Post"])
ex.note("gênant", "Calque renommé dans Post : le graphe garde l'ancien nom",
        "Dans la page Post, le calque « Sortie Gradient » renommé « Fond seul » (double-clic sur son nom) : le panneau "
        "du bloc Post liste « Fond seul », mais le workflow, son bloc Sortie (« Sortie Gradient · vers Post ») et son "
        "fil (« « Sortie Gradient » : son image vers Post ») gardent l'ancien nom. Dans l'autre sens (renommer le "
        "workflow dans le graphe), le calque suit.", shot=str(OUT / "visible_rename_run2.png"),
        steps=ST + ["Clic droit sur Gradient > Vers Post", "Onglet Post", "Double-clic sur le nom « Sortie Gradient » : "
                    "Fond seul", "Onglet Graphe, clic sur Post et sur le fil de la branche"])
