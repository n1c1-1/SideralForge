"""block_actions round 2: eighth batch (reproduced twice)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
ex.note("mineur", "Remettre au Post l'image d'un workflow par « Vers Post » : aucun message",
        "Chaîne jusqu'à Couleur. Cliquer le fil Couleur → Post, Suppr : « « Workflow » hors Post · menu : Vers Post », "
        "Post dit « 0 calque ». Clic droit Couleur → « Vers Post » : le fil revient et Post dit « 1 calque », mais rien "
        "ne s'affiche (les « Vers Post » des autres blocs disent « Calque « … » vers Post »). Même silence pour « Vers "
        "Post » de la Fusion.\n(reproduit 2 fois)",
        shot=str(ex.out / "dig_main_wire_1.png"),
        steps=["Chaîne jusqu'à Couleur", "Cliquer le fil vers Post, Suppr", "Clic droit Couleur → Vers Post"])
