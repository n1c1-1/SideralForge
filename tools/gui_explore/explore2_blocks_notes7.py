"""block_actions round 2: seventh batch (each reproduced twice)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
R = "\n(reproduit 2 fois)"
NOTES = [
    ("mineur", "Calques faits depuis le menu d'Étoiles/fond : noms anglais, deux « Recombine », « + Recombiner » sans message",
     "Interface en français, chaîne complète jusqu'à Étoiles/fond. Clic droit → « Étoiles vers Post », « Sans étoiles "
     "vers Post », « + Recombiner ». Les calques enregistrés dans la composition s'appellent « Workflow · Stars alone », "
     "« Workflow · Starless image », « Workflow · Recombine » et de nouveau « Workflow · Recombine » (le Recombiner "
     "donné d'office et le nouveau ont le même nom). Ces noms sont ceux que Post affichera une fois calculé (Post est "
     "vide tant que rien n'est calculé, ici). « + Recombiner » n'affiche aucun message, alors que « Étoiles vers Post » "
     "dit « Étoiles seules · « Workflow » vers Post ».",
     ["Chaîne complète", "Clic droit Étoiles/fond → Étoiles vers Post, Sans étoiles vers Post, + Recombiner"],
     None),
]
for sev, title, detail, steps, shot in NOTES:
    ex.note(sev, title, detail + R, shot=False, steps=steps)
print(len(NOTES), "noted")
