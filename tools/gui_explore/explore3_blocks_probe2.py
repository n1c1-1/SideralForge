"""Round 3 (control) block_actions: the workflow's image put back to Post (submenu, port pulled), twice."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from explore2_blocks_common import SHORT, TOASTS, EXC, G, Explorer  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from astro_gpu.client.graph import WireItem  # noqa: E402
ex = Explorer("block_actions", lang="fr")
g = G(ex)
for rep in range(2):
    for how in ("submenu", "port"):
        g.new("session", SHORT)
        w = [x for x in ex.view.scene().items() if isinstance(x, WireItem) and x.isVisible()][0]
        pt = ex.view.mapFromScene(w.path().pointAtPercent(0.5))
        ex.hover(ex.view.viewport(), pt); ex.click(ex.view.viewport(), pt); g.act(); ex.key(Qt.Key_Delete); ex.settle(4)
        n, nu = len(TOASTS), len(ex.s._undo)
        if how == "submenu":
            g.menu(g.blk("couleur"), pick="Vers Post")
        else:
            ex.pull_port_to_post(g.blk("couleur")); ex.settle(4)
        print(rep, how, "msgs", TOASTS[n:], "| Post", g.blk("final").summary, "| undo +", len(ex.s._undo) - nu,
              "| wires", len([x for x in ex.view.scene().items() if isinstance(x, WireItem) and x.isVisible()]), flush=True)
        if how == "port":
            # the same gesture on Gradient (not the last block): for comparison
            n = len(TOASTS)
            ex.pull_port_to_post(g.blk("fond")); ex.settle(4)
            print("   port of Gradient pulled to Post:", TOASTS[n:], "| Post", g.blk("final").summary)
            ex.shot(f"r3_port_pull_{rep}")
for e in EXC:
    print("EXC", e[-500:])
ex.close()
