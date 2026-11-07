"""block_actions round 2: the block panels' action buttons at 1366x768 and 1920x1080, FR and EN (truncation, overlap)."""
from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(600, exit=True)

from explore2_blocks_common import CHAIN, G, Explorer, TOASTS  # noqa: E402,F401
from PySide6.QtWidgets import QAbstractButton, QLabel  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
W, H = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1366x768").split("x"))
ex = Explorer("block_actions", lang=LANG, size=(W, H))
g = G(ex)


def cut_texts():
    out = []
    pn = ex.page.panel
    for b in pn.findChildren(QAbstractButton):
        if b.isVisible() and b.text():
            need = b.fontMetrics().horizontalAdvance(b.text().replace("&&", "&")) + 12
            if need > b.width() and not b.toolTip():
                out.append(f"button «{b.text()}» {b.width()}px < {need}px")
    for lb in pn.findChildren(QLabel):
        if lb.isVisible() and lb.text() and not lb.wordWrap() and "<" not in lb.text():
            need = lb.fontMetrics().horizontalAdvance(lb.text())
            if need > lb.width() + 2 and not lb.toolTip():
                out.append(f"label «{lb.text()[:40]}» {lb.width()}px < {need}px")
    vis = pn.visibleRegion().boundingRect()
    if ex.w.width() > W + 4:
        out.append(f"window width {ex.w.width()} > screen {W}")
    return out


g.new("session", ["calibration", "registration", "stacking", "astrometry", "gradient"])
g.drag_branch("color", "astrometrie", ex.s.paths[0].pid)
for key in ("sources", "calibration", "empilement", "astrometrie", "fond", "couleur", "final"):
    it = g.blk(key)
    if it is None:
        continue
    ex.click_block(it)
    c = cut_texts()
    print(f"{LANG} {W}x{H} {key}: {c}", flush=True)
    ex.shot(f"size_{LANG}_{W}x{H}_{key}")
g.new("session", CHAIN)
for key in ("etoiles", "composition"):
    it = g.blk(key)
    if it is not None:
        ex.click_block(it)
        print(f"{LANG} {W}x{H} {key}: {cut_texts()}", flush=True)
        ex.shot(f"size_{LANG}_{W}x{H}_{key}")
ex.close()
