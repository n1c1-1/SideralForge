"""Report screenshots, second pass: the wire to Post at 1920x1080, the Fusion card's reason, a refused run."""
import os
import sys

os.environ.setdefault("GUI_EXPLORE_OUT", "F:/astro_essais/A_REGARDER/bilan_images")
sys.path.insert(0, os.path.dirname(__file__))
from harness import Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402

ex = Explorer("shots", lang="fr", size=(1920, 1080))
ex.open(ex.project("legacy", "Complet"))
ex.right_click_block(ex.block("fond"), pick="Vers Post")
ex.page.graph.fit() if hasattr(ex.page.graph, "fit") else None
ex.shot("1_vers_post_sans_sortie")

ex.open(ex.project("mono", "Mono"))
c = ex.card("fusion")
ex.page.show_hint(c, c.toolTip() or getattr(c, "reason", "") or "")
ex.shot("3_fusion_grisee")

root = ex.project("session", "Dossier_absent")
p = project2.load(root)
p.sessions[0].lights = ["F:/astro_essais/_nulle_part/lights"]
p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
project2.save(p)
ex.open(root)
ex.key(Qt.Key_F5, ex.w)
ex.shot("4_refus_avant_calcul")
ex.close()
print("ok")
