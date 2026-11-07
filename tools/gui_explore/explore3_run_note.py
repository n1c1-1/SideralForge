"""run_status round 3: the new problem found (reproduced twice: F5 and Shift+F5), written with ex.note."""
from harness import Explorer

ex = Explorer("run_status", lang="fr")
ex.note("mineur", "Pause vérif sur un bloc désactivé : gardée et annoncée, mais l'exécution passe sans s'arrêter",
        "Session → Calibration, Alignement, Empilement, Gradient, Débruitage ; Pause vérif sur Gradient, puis D sur "
        "Gradient. La pause est gardée (correction du tour 2) et l'info-bulle du Gradient dit toujours « Point de "
        "vérification : l'exécution s'arrête ici pour votre vérification. ». Mais F5 prévoit Calibration, Alignement, "
        "Empilement, Débruitage sans aucune pause, et Maj+F5 sur Débruitage aussi (aucune ligne « Pause vérif ») : "
        "l'exécution passe le bloc désactivé sans s'arrêter, et rien ne le dit. Réactivé (D), la pause marche de "
        "nouveau (« Pause vérif après Gradient »). Le menu du bloc désactivé propose toujours « Pause vérif ».",
        shot="F:/astro_essais/A_REGARDER/exploration3/run_status/kf12_here.png",
        steps=["Session → Calibration, Alignement, Empilement, Gradient, Débruitage",
               "Clic droit sur Gradient > Pause vérif", "Clic sur Gradient, D",
               "Survoler Gradient ; F5, ou clic sur Débruitage puis Maj+F5"])
ex.close()
