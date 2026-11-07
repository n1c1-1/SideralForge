"""block_actions round 2: ninth batch (reproduced twice)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from harness import Explorer  # noqa: E402

ex = Explorer("block_actions")
ex.note("question", "Poser une Fusion ajoute, sans le dire, Débruitage et Étoiles/fond dans chaque canal",
        "Projet mono : Ha et OIII ont chacun Calibration → Alignement → Empilement → Astrométrie. Double-clic sur la carte "
        "Fusion : en plus de la Fusion et d'un Recombiner, Débruitage et Étoiles/fond apparaissent dans les workflows Ha "
        "et OIII (4 blocs que l'utilisateur n'a pas posés). Aucun message ne le dit (seul le panneau de la Fusion explique "
        "que les étoiles sont retirées dans chaque canal). Un seul Ctrl+Z retire le tout. Est-ce voulu, et faut-il le dire ?"
        "\n(reproduit 2 fois)",
        shot=str(ex.out / "dig_fusion_place_1.png"),
        steps=["Projet mono, Astrométrie dans Ha et OIII", "Double-clic sur la carte Fusion"])
