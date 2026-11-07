"""Round 3 (control) fuzz, part 2: the points the first replay could not reach, twice."""
import os
import sys
from pathlib import Path

sys.argv = [sys.argv[0], "run"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_blocks_lib import TOASTS  # noqa: E402
import explore3_fuzz as G  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel  # noqa: E402

fz = G.Fuzz2("fr")
fz.panels = False
ex = fz.ex


def blk(pred):
    return next((x for x in fz.items() if pred(x)), None)


def wfs():
    return [w.name for w in ex.s.p.workflows]


for k in range(2):
    print("===== rep", k, flush=True)
    # A1: the greyed Étoiles/fond card's reason, before any Stacking
    for kind in ("blank", "two_flows", "session"):
        fz.start({"kind": kind, "lang": "fr"}, f"b1_{kind}_{k}")
        c = ex.card("stars")
        print("A1", kind, "stars card reason:", repr(getattr(c, "reason", None)), "| title", getattr(c, "title", None))
    # A3: Shift+F5 without frames, what the run window says
    fz.start({"template": "hoo", "lang": "fr"}, f"b3_{k}")
    ex.click_block(blk(lambda x: x.node.key == "composition"))
    ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
    ex.settle(8)
    ds = [d for d in ex.dialogs() if d.__class__.__name__ == "RunDialog"]
    texts = [x.text() for x in ds[0].findChildren(QLabel) if x.isVisible() and x.text()] if ds else []
    print("A3 Shift+F5 dialog:", bool(ds), "| texts:", [t[:120] for t in texts][:14])
    if ds and k == 1:
        ex.shot("r3_run_to_here_dialog", ds[0])
    ex.close_dialogs()
    # A5: the template's creation and the undo stack
    fz.start({"template": "classic_dn", "lang": "fr"}, f"b5_{k}")
    print("A5 after creation: undo entries", len(ex.s._undo), "| Annuler enabled", ex.w.a_undo.isEnabled(),
          "| text", ex.w.a_undo.text())
    fz.undo()
    fz.wait_ready()
    print("   after Ctrl+Z: workflows", wfs(), "| sessions", [s.name for s in ex.s.p.sessions], "| toasts", TOASTS[-1:])
    if k == 1:
        ex.shot("r3_template_undone")
    fz.redo()
    fz.wait_ready()
    print("   after Ctrl+Y:", wfs())
    # A8: right click on Empilement
    for tpl in ("full", "classic", "nights", "calibrated", "aligned"):
        fz.start({"template": tpl, "lang": "fr"}, f"b8_{tpl}_{k}")
        fz.activate()
        it = blk(lambda x: x.node.key in ("empilement", "stacking"))
        m = ex.right_click_block(it) if it is not None else None
        ex.settle(4)
        print("A8", tpl, "Empilement:", it is not None, "menu:", [x.get("text") for x in m] if m is not None else None)
    # A9: Supprimer workflow on a Fusion channel
    for tpl in ("lrgb", "hoo", "sho", "rgb"):
        fz.start({"template": tpl, "lang": "fr"}, f"b9_{tpl}_{k}")
        fz.activate()
        first = ex.s.p.workflows[0]
        hit = None
        for i in range(40):
            cands = [x for x in fz.items() if first.id in [q.pid for q in x.node.paths]
                     and x.node.key not in ("session", "sources")]
            if i >= len(cands):
                break
            it = cands[-1 - i]
            title = fz.title(it)
            m = ex.right_click_block(it)
            ex.settle(3)
            item = next((x.get("text") for x in m if "Supprimer workflow" in (x.get("text") or "")), None)
            if item:
                hit = title
                break
        if hit is None:
            print("A9", tpl, "no 'Supprimer workflow' in the menus of", first.name)
            continue
        n, w0 = len(TOASTS), wfs()
        fz.errs.clear()
        it = next(x for x in reversed(fz.items()) if fz.title(x) == hit and first.id in [q.pid for q in x.node.paths])
        ex.right_click_block(it, pick="Supprimer workflow")
        fz.wait_ready()
        w1 = wfs()
        try:
            ex.s.p.validate()
            ok = "valide"
        except Exception as e:  # noqa: BLE001
            ok = f"INVALIDE {e}"
        print("A9", tpl, f"on {hit} ({first.name}): toasts {TOASTS[n:]}; workflows {w0} -> {w1}; {ok}; undo n "
              f"{len(ex.s._undo)}; errs {[x.strip().splitlines()[-1] for x in fz.errs]}")
        if k == 1 and tpl == "lrgb":
            ex.shot("r3_delete_wf_fusion")
os._exit(0)
