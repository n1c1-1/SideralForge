"""Screenshots of the final state for the user's report (control round): a few fixed points, French, 1366x768."""
import os
import sys

os.environ.setdefault("GUI_EXPLORE_OUT", "F:/astro_essais/A_REGARDER/bilan_images")
sys.path.insert(0, os.path.dirname(__file__))
from harness import Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402

ex = Explorer("shots", lang="fr", size=(1366, 768))

# 1. A block's image sent to Post: no Output block, the wire carries it
ex.open(ex.project("legacy", "Complet"))
ex.right_click_block(ex.block("fond"), pick="Vers Post")
ex.shot("1_vers_post_sans_sortie")

# 2. Several nights: a third Session joins the two
ex.new_project("Nuits", template="nights")
ex.double_click_card("session")
ex.shot("2_nuits_session_jointe")

# 3. Fusion card greyed until every channel has Astrometry
ex.open(ex.project("mono", "Mono"))
c = ex.card("fusion")
ex.hover(c, c.rect().center())
ex.shot("3_fusion_grisee")

# 4. Run refused before the run: the frames' folder is gone
root = ex.project("session", "Dossier_absent")
p = project2.load(root)
p.sessions[0].lights = ["F:/astro_essais/_nulle_part/lights"]
w = p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
project2.save(p)
ex.open(root)
ex.key(Qt.Key_F5, ex.w)
ex.shot("4_refus_avant_calcul")

# 5-6. French agreement, then Undo says what it undid
ex.open(ex.project("legacy", "Accords"))
ex.click_block(ex.block("calibration"))
ex.key(Qt.Key_Delete)
ex.shot("5_calibration_supprimee")
ex.undo()
ex.shot("6_annule_dit_quoi")

# 7. The template window fits a 768 screen, names whole
d = ex.page.new_workflow()
ex.settle(6)
ex.shot("7_modeles_1366", d)
d.reject()
ex.close()
print("ok")
