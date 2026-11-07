"""Round 4, clavier_fenetre: what the blocks are called and what their panels hold (to drive them). Prints only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from explore_post_common import P, start  # noqa: E402
from explore2_blocks_common import G, Explorer  # noqa: E402
from PySide6.QtWidgets import QWidget, QAbstractButton, QLabel  # noqa: E402

start(600)
ex = Explorer("clavier_fenetre")
g = G(ex)
pp = P(ex)
g.new("session_empty_wf", ["calibration", "registration", "stacking", "astrometry", "gradient", "denoise", "color",
                           "stars"])
for it in ex.blocks():
    n = it.node
    print("BLOCK", repr(n.key), repr(it.title), "sid", n.sid)
for it in list(ex.blocks()):
    key = it.node.key
    it = g.blk(key)
    if it is None:
        continue
    ex.click_block(it)
    pn = ex.page.panel
    kinds = {}
    for x in pn.findChildren(QWidget):
        if x.isVisibleTo(pn) and not isinstance(x, QLabel):
            kinds.setdefault(type(x).__name__, 0)
            kinds[type(x).__name__] += 1
    print("PANEL", key, kinds)
    print("   buttons", [b.text() for b in pn.findChildren(QAbstractButton) if b.isVisibleTo(pn)][:15])
ex.close()
