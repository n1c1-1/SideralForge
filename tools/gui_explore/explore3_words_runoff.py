"""Words, round 3: Run to here on a switched-off block (Gradient 2): what the run window says."""
from __future__ import annotations

import sys

from harness import Explorer
from PySide6.QtWidgets import QLabel

from astro_gpu.client import i18n

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1366
T = i18n._
ex = Explorer("words_layout", lang=LANG, size=(W, 768 if W == 1366 else 1080))
ex.open(ex.project("session"))
ex.settle(6)
ex.click_block(ex.block("session") or ex.blocks()[0])
for k in ("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "gradient"):
    ex.double_click_card(k)
    ex.settle(3)
g2 = sorted([x for x in ex.blocks() if x.node.key.startswith("fond")], key=lambda x: x.pos().x())[-1]
ex.right_click_block(g2, T("Switch off"))
ex.settle(4)
g2 = sorted([x for x in ex.blocks() if x.node.key.startswith("fond")], key=lambda x: x.pos().x())[-1]
print("TITLES", [x.title for x in sorted(ex.blocks(), key=lambda x: x.pos().x())])
ex.click_block(g2)
d = ex.w.run.run_to_here()
ex.settle(8)
if d is not None:
    print("RUNWIN", [x.text() for x in d.findChildren(QLabel) if x.isVisible() and x.text()])
    print("SHOT", ex.shot(f"runoff_{LANG}_{W}", d))
    d.hide()
ex.close()
