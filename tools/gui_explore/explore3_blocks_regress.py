"""Round 3 (control) block_actions: around the round-2 fixes, looking for regressions (prints what is seen, twice)."""
from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(600, exit=True)

from explore2_blocks_common import CHAIN, EXC, SHORT, TOASTS, G, Explorer, drawn, panel_info, state  # noqa: E402,F401
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLineEdit  # noqa: E402

ex = Explorer("block_actions", lang=sys.argv[1] if len(sys.argv) > 1 else "fr")
g = G(ex)


def exc(e0):
    return [x.strip().splitlines()[-1] for x in EXC[e0:]]


def key_on(it, k):
    ex.click_block(it)
    g.act()
    n = len(TOASTS)
    ex.key(k)
    ex.settle(4)
    return TOASTS[n:]


for rep in range(2):
    print(f"===== rep {rep}", flush=True)
    # 1. D on Session / Fusion / Recombiner: says why?
    g.mono_fusion()
    e0 = len(EXC)
    print("D on Session Ha:", key_on(g.blk("session", sid=1), Qt.Key_D))
    print("D on Fusion:", key_on(g.blk("fusion"), Qt.Key_D))
    comp = next((b for b in ex.blocks() if b.node.key == "composition"), None)
    print("D on Recombiner:", key_on(comp, Qt.Key_D) if comp else "no Recombiner")
    ha = g.pid_of("Ha")
    print("D on Ha's Astrométrie:", key_on(g.blk("astrometrie", pid=ha), Qt.Key_D))
    print("Suppr on Ha's Astrométrie:", key_on(g.blk("astrometrie", pid=ha), Qt.Key_Delete))
    print("Suppr on Ha's Empilement:", key_on(g.blk("empilement", pid=ha), Qt.Key_Delete), g.graph_of("Ha"))
    print("Suppr on Ha's Débruitage:", key_on(g.blk("debruitage", pid=ha), Qt.Key_Delete), g.graph_of("Ha"))
    g.undo()
    print("  undo ->", g.graph_of("Ha"), "| undo stack", len(ex.s._undo))
    # drag Ha's session to the palette
    n = len(TOASTS)
    ex.drag_block_to_palette(g.blk("session", sid=1))
    ex.settle(4)
    print("Session Ha dragged to palette:", TOASTS[n:], [x["name"] for x in state(ex)["sessions"]])
    # a second Fusion
    n = len(TOASTS)
    g.place("fusion")
    print("second Fusion dbl-click:", TOASTS[n:], [q.name for q in ex.s.paths], "fusion blocks",
          len([b for b in ex.blocks() if b.node.key == "fusion"]))
    print("EXC", exc(e0))
    ex.shot(f"r3_regress_mono_{rep}")

    # 2. join: a Session dropped on a two-flow project, the filter flow
    g.new("two_flows", clear=False)
    n = len(TOASTS)
    ex.click_block(g.blk("session", sid=2))
    g.place("session")
    st = state(ex)
    print("two flows, Session 2 selected, dbl Session card:", TOASTS[n:], "| sessions",
          [(x["id"], x["name"], x["filter"]) for x in st["sessions"]], "| workflows",
          [(w["name"], w["sessions"]) for w in st["workflows"]])
    g.undo()
    print("  undo ->", [(x["id"], x["name"]) for x in state(ex)["sessions"]])
    # 3. names: a session renamed to the other's name, then empty, in its panel
    from PySide6.QtTest import QTest
    for txt in ("Session 2", ""):
        g.new("two_sessions", ["calibration", "registration", "stacking"])
        ex.click_block(g.blk("session", sid=1))
        ed = next((e for e in ex.page.panel.findChildren(QLineEdit) if e.isVisible() and e.text() == "Session 1"), None)
        n = len(TOASTS)
        ex.click(ed)
        ed.selectAll()
        QTest.keyClick(ed, Qt.Key_Delete)
        if txt:
            QTest.keyClicks(ed, txt)
        QTest.keyClick(ed, Qt.Key_Return)
        ex.settle(6)
        print(f"rename Session 1 -> {txt!r}:", TOASTS[n:], [x["name"] for x in state(ex)["sessions"]],
              "| field shows", repr(ed.text()) if ed is not None else None)
    # 4. Duplicate then Ctrl+Z; delete workflow on the wire menu
    g.new("session", SHORT)
    n = len(TOASTS)
    g.menu(g.blk("couleur"), pick="Dupliquer")
    print("Dupliquer:", TOASTS[n:], [q.name for q in ex.s.paths], "| drawn", g.titles())
    g.undo()
    print("  undo ->", [q.name for q in ex.s.paths])
    # 5. empty click deselects; Ctrl+Z selects what it brings back
    g.new("session", SHORT)
    ex.click_block(g.blk("fond"))
    g.act()
    ex.key(Qt.Key_Delete)
    g.undo()
    print("after Suppr Gradient + Ctrl+Z: selected", ex.s.sel_step)
    from PySide6.QtCore import QPoint
    ex.click(ex.view.viewport(), QPoint(30, 30))
    ex.settle(3)
    n = len(TOASTS)
    s0 = g.snap()
    g.act()
    ex.key(Qt.Key_Delete)
    ex.key(Qt.Key_D)
    print("click empty: selected", ex.s.sel_step, "| Suppr/D change", g.snap() != s0, TOASTS[n:])
    # with nothing selected, the palette targets the single workflow
    n = len(TOASTS)
    g.place("denoise")
    print("nothing selected, dbl Débruitage:", TOASTS[n:], g.graph_of())
    print("EXC", exc(e0))

for e in EXC:
    print("EXC", e[-600:])
ex.close()
