"""Round 3 (control) block_actions: wire to Post put back, Open in Post, Session panel, Fusion layers, join message."""
from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(600, exit=True)

from explore2_blocks_common import CHAIN, EXC, SHORT, TOASTS, G, Explorer, drawn, panel_info, state  # noqa: E402,F401
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QPushButton  # noqa: E402

ex = Explorer("block_actions", lang="fr")
g = G(ex)

for rep in range(2):
    print(f"===== rep {rep}", flush=True)
    from astro_gpu.client.graph import WireItem
    # 1. the workflow's wire to Post removed: how does it come back?
    g.new("session", SHORT)
    ws = [x for x in ex.view.scene().items() if isinstance(x, WireItem) and x.isVisible()]
    w = ws[0]
    pt = ex.view.mapFromScene(w.path().pointAtPercent(0.5))
    ex.hover(ex.view.viewport(), pt)
    ex.click(ex.view.viewport(), pt)
    g.act()
    ex.key(Qt.Key_Delete)
    ex.settle(4)
    print("wire removed:", TOASTS[-1:], "| Post:", g.blk("final").summary)
    for key in ("couleur", "fond", "final"):
        mm = g.menu(g.blk(key))
        print(f"  menu {key}:", g.texts(mm), [(x.get("text"), [y.get("text") for y in x["sub"]]) for x in mm if x.get("sub")])
    n = len(TOASTS)
    ex.pull_port_to_post(g.blk("couleur"))
    ex.settle(4)
    print("  port of Couleur pulled to Post:", TOASTS[n:], "| Post:", g.blk("final").summary, "| removed",
          ex.s.post_removed())
    ex.shot(f"r3_wire_back_{rep}")
    # 2. Open in Post on Recombiner
    g.new("session", CHAIN)
    comp = next((b for b in ex.blocks() if b.node.key == "composition"), None)
    ex.click_block(comp)
    btns = [(b.text(), b.isEnabled(), b.toolTip()[:80]) for b in ex.page.panel.findChildren(QPushButton) if b.isVisible()]
    print("Recombiner panel buttons:", btns)
    b = next((b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and "Post" in b.text()), None)
    nu, post0 = len(ex.s._undo), json.dumps(getattr(ex.s.p, "post", None), ensure_ascii=False)
    if b is not None:
        bt = b.text()
        ex.click(b)
        ex.settle(10)
        print("  after", bt, ": mode", ex.w.mode, "| undo +", len(ex.s._undo) - nu, "| composition changed",
              json.dumps(getattr(ex.s.p, "post", None), ensure_ascii=False) != post0)
        ex.w.set_mode(0)
        ex.settle(6)
    # 3. a Session's panel buttons
    g.new("two_sessions", ["calibration", "registration", "stacking"])
    ex.click_block(g.blk("session", sid=1))
    print("Session panel buttons:", [x["text"] for x in panel_info(ex)["buttons"]])
    # 4. Fusion to Post: the Post page names
    g.mono_fusion()
    n = len(TOASTS)
    g.menu(g.blk("fusion"), pick="Vers Post")
    print("Fusion Vers Post:", TOASTS[n:], "| layers", [x.get("name") for x in state(ex)["post"]])
    ex.w.set_mode(2)
    ex.settle(12)
    print("  Post page labels:", [x.text()[:60] for x in ex.w.findChildren(QLabel) if x.isVisible() and "HOO" in x.text()][:6])
    ex.w.set_mode(0)
    ex.settle(4)
    # 5. join on a one-session project: every message shown
    g.new("session", ["calibration", "registration", "stacking"])
    n = len(TOASTS)
    ex.double_click_card("session")
    ex.settle(8)
    print("join dbl Session card: all msgs", TOASTS[n:], "| shown now:",
          [x.text() for x in ex.view.findChildren(QLabel) if x.isVisible() and x.text()][:4])

for e in EXC:
    print("EXC", e[-600:])
ex.close()
