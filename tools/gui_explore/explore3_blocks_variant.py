"""block_actions round 2: "Try as variant", Replace, Cancel, Keep, Discard, with Ctrl+Z / Ctrl+Y.

Runs are blocked, so nothing is ever computed: the project is told its results would be lost by a change (the model's
"throws_away" answers yes), which is what makes the window offer Replace / Try as variant. Nothing else is changed."""
from __future__ import annotations

import faulthandler
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
faulthandler.dump_traceback_later(600, exit=True)

from explore2_blocks_common import EXC, SHORT, TOASTS, G, Explorer, drawn, panel_info, state  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractSpinBox, QPushButton  # noqa: E402

LANG = sys.argv[1] if len(sys.argv) > 1 else "fr"
ex = Explorer("block_actions", lang=LANG)
g = G(ex)
TXT = {"fr": ("Essayer en variante", "Remplacer", "Annuler", "Garder", "Abandonner"),
       "en": ("Try as variant", "Replace", "Cancel", "Keep", "Discard")}[LANG]


def spin_value():
    it = g.blk("calibration")
    if it is None:
        return None
    ex.click_block(it)
    sp = next((x for x in ex.page.panel.findChildren(QAbstractSpinBox) if x.isVisible()), None)
    return sp.text() if sp is not None else None


def snap():
    return json.dumps([(w.name, w.blocks, getattr(w, "variant_of", None)) for w in ex.s.p.workflows], sort_keys=True,
                      default=str)


def buttons():
    return [b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and b.isEnabled()]


def press(prefix):
    b = next((b for b in buttons() if b.text().startswith(prefix)), None)
    if b is None:
        return False
    ex.click(b)
    ex.settle(8)
    return True


def change_setting():
    """Calibration's first spin box, typed as a user would."""
    ex.click_block(g.blk("calibration"))
    sp = next((x for x in ex.page.panel.findChildren(QAbstractSpinBox) if x.isVisible()), None)
    if sp is None:
        return False
    ex.click(sp)
    sp.selectAll()
    QTest.keyClicks(sp, "0,9" if LANG == "fr" else "0.9")
    QTest.keyClick(sp, Qt.Key_Return)
    ex.settle(8)
    return True


for rep in range(2):
    for choice in (TXT[0], TXT[1], TXT[2]):
        g.new("session", SHORT)
        ex.s.throws_away = lambda *a, **k: True
        s0 = snap()
        n, e0 = len(TOASTS), len(EXC)
        change_setting()
        shown = [b.text() for b in buttons()]
        if rep == 0:
            ex.shot(f"variant_choice_{LANG}")
        ok = press(choice)
        s1 = snap()
        names = [q.name for q in ex.s.paths]
        t = TOASTS[n:]
        nu = len(ex.s._undo)
        v1 = spin_value()
        g.undo()
        s2 = snap()
        v2 = spin_value()
        g.redo()
        s3 = snap()
        v3 = spin_value()
        print(f"   spin (Calibration of the workflow) after {choice}: {v1}, after Ctrl+Z {v2}, after Ctrl+Y {v3}")
        print(f"[{rep}] {choice}: offered {[x for x in shown if any(x.startswith(p) for p in TXT[:3])]} pressed={ok} "
              f"workflows={names} msgs={t} undo_entries={nu} changed={s1 != s0} undo_ok={s2 == s0} redo_ok={s3 == s1} "
              f"exc={[x.strip().splitlines()[-1] for x in EXC[e0:]]}", flush=True)
        if choice == TXT[0] and ok:
            ex.shot(f"variant_made_{LANG}_{rep}")
            # the variant's panel: Keep / Discard
            for kd in (TXT[3], TXT[4]):
                g.new("session", SHORT)
                ex.s.throws_away = lambda *a, **k: True
                change_setting()
                press(TXT[0])
                var = next((q for q in ex.s.paths if q.variant_of is not None), None)
                if var is None:
                    print("   no variant")
                    break
                it = next((b for b in ex.blocks() if var.pid in [q.pid for q in b.node.paths] and len(b.node.paths) == 1),
                          None)
                if it is not None:
                    ex.click_block(it)
                s0 = snap()
                n = len(TOASTS)
                pb = [b.text() for b in buttons()]
                ok = press(kd)
                s1 = snap()
                g.undo()
                s2 = snap()
                g.redo()
                s3 = snap()
                print(f"   {kd}: graph of « {ex.s.paths[0].name} » after: {g.graph_of(ex.s.paths[0].name)}")
                if rep == 0:
                    ex.shot(f"variant_{kd}_{LANG}")
                print(f"   {kd}: panel had {pb[:6]} pressed={ok} workflows={[q.name for q in ex.s.paths]} msgs={TOASTS[n:]} "
                      f"changed={s1 != s0} undo_ok={s2 == s0} redo_ok={s3 == s1}", flush=True)
ex.close()

