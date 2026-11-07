"""usages E: an Image source (a 16-bit TIFF, its kind), two Gradients in one workflow (numbering), the workflow and the
session renamed, Show all, two colour nights as parallel flows; Save, the window closed, reopened in a new window;
then the same project reopened with the interface in English."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, TMP, U, _, button, compare, log, panel, start, wait  # noqa: E402
from PySide6.QtWidgets import QLineEdit, QPushButton  # noqa: E402

start(1200)
rep = sys.argv[1] if len(sys.argv) > 1 else "1"
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
u = U("fr")
ex = u.ex


def want(k):
    return only is None or k in only


def name_dialog(newname):
    def h(d):
        eds = d.findChildren(QLineEdit)
        if eds:
            eds[0].setText(newname)
        d.accept()
    return h


def tiff16(name):
    import numpy as np
    import tifffile
    d = TMP / "data" / "images"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    rng = np.random.default_rng(3)
    tifffile.imwrite(f, (3000 + rng.normal(0, 50, (120, 160, 3))).astype(np.uint16), photometric="rgb")
    return f


# ================================================================== I. Image source, TIFF; two Gradients; renames; Show all
if want("I"):
    tif = tiff16(f"pile16_{rep}.tif")
    root, _c = u.new_from_home(f"ImageTif_{rep}", "blank")
    u.place("image")
    ex.click_block(u.blk("image"))
    wait(ex, 0.3)
    b = button(panel(ex), _("Choose image…"), QPushButton) or button(panel(ex), _("Change image…"), QPushButton)
    ex.file_queue = [str(tif)]
    ex.click(b)
    wait(ex, 0.8)
    q = next(q for q in ex.s.paths if q.source)
    log("I source", q.source, u.toasts[-1:])
    u.place("astrometry", "gradient", "gradient", "color")
    titles = [it.title for it in ex.blocks()]
    log("I blocks", titles)
    # the second Gradient's setting
    g2 = u.title_blk("Gradient 2")
    ch = u.change_setting(g2) if g2 else None
    # rename the workflow by the block menu's Workflow submenu, the session absent here
    ex.dialog_handler = name_dialog("Ma pile TIFF")
    ex.right_click_block(u.blk("image"), _("Rename"))
    wait(ex, 0.6)
    log("I renamed", [q.name for q in ex.s.paths], "menu", json.dumps(ex.menus[-1:], ensure_ascii=False)[:1500],
        "modals", ex.modals[-2:], u.toasts[-1:])
    from astro_gpu.client.panel import CompactRow
    for tt in ("Gradient 1", "Gradient 2"):
        ex.click_block(u.title_blk(tt))
        wait(ex, 0.6)
        log("I panel rows", tt, [r.r["label"] for r in panel(ex).findChildren(CompactRow) if r.isVisible()],
            [b.text() for b in panel(ex).findChildren(QPushButton) if b.isVisible()][:8])
    # Show all on
    ex.click(ex.view.b_all)
    wait(ex, 0.4)
    ex.click_block(u.title_blk("Gradient 2") or u.blk("image"))
    wait(ex, 0.3)
    u.to_post(u.title_blk("Gradient 1") or u.blk("astrometrie"))
    log("I flows", [q.name for q in ex.s.paths], "toasts", u.toasts[-2:])
    ex.click_block(u.title_blk("Gradient 2") or u.blk("image"))
    wait(ex, 0.3)
    ex.shot(f"E_I{rep}_built")
    before, saved, after, diffs = u.roundtrip(root, "image TIFF", ["Image + TIFF 16 bits", "deux Gradient"])
    ex.shot(f"E_I{rep}_reopened")
    log("I titles after", [b[1] for b in after["blocks"]], after["flows"], after["selection"], after["show_all"])
    # the same project, the interface in English
    ex.w.change_language("en")
    wait(ex, 1.0)
    en = u.snapshot()
    log("I english", en["flows"], [b[1] for b in en["blocks"]], en["status"])
    ex.shot(f"E_I{rep}_english")
    ex.w.change_language("fr")
    wait(ex, 0.8)
    (ex.out / f"E_I{rep}.json").write_text(json.dumps({"saved": saved, "after": after, "diffs": diffs, "en": en},
                                                       ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex.w.close_project()

# ================================================================== F. two colour nights, parallel flows; session renamed
if want("F"):
    l1 = ex.raws(f"Nuits_{rep}/n1", 6)
    l2 = ex.raws(f"Nuits_{rep}/n2", 5, start=7000)
    root, _c = u.new_from_home(f"DeuxNuits_{rep}", "blank")
    u.place("session")
    u.add_frames(1, l1, 0)
    u.place("calibration", "registration", "stacking", "astrometry")
    z = ex.new_flow_zone()
    rec = ex.drag_palette("session", ex.ghost_center(z)) if z is not None else None
    wait(ex, 0.8)
    sids = [x.id for x in ex.s.p.sessions]
    log("F second flow", rec, sids, [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths], u.toasts[-1:])
    if len(sids) > 1:
        u.add_frames(sids[-1], l2, 0)
        u.place("calibration", "registration", "stacking", "gradient")
    ex.dialog_handler = name_dialog("Nuit du 12")
    ex.right_click_block(u.blk("session", sid=sids[-1]), _("Rename session"))
    wait(ex, 0.6)
    log("F flows", [(q.name, ex.s.p.workflow(q.pid).sessions) for q in ex.s.paths],
        [(x.id, x.name) for x in ex.s.p.sessions], [it.title for it in ex.blocks()])
    ex.shot(f"E_F{rep}_built")
    before, saved, after, diffs = u.roundtrip(root, "deux nuits", ["deux Sessions en flux parallèles"])
    ex.shot(f"E_F{rep}_reopened")
    log("F after", after["flows"], [b[1] for b in after["blocks"]])
    (ex.out / f"E_F{rep}.json").write_text(json.dumps({"saved": saved, "after": after, "diffs": diffs},
                                                       ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ex.w.close_project()

# ================================================================== L. a template made in French, reopened in English
if want("L"):
    root, _c = u.new_from_home(f"Langue_{rep}", "full")
    u.save()
    ex.w.close()
    ex.settle(4)
    ex.w = None
    from astro_gpu.client import app as appmod  # noqa: E402
    appmod.apply_language("en")
    u.reopen_new_window(root)
    en = u.snapshot()
    log("L english", en["flows"], [b[1] for b in en["blocks"]][:12], en["status"], en["title"])
    ex.shot(f"E_L{rep}_english")
    appmod.apply_language("fr")
    ex.w.close_project()
log("EXC", len(EXC), "\n".join(e[-1200:] for e in EXC))
if ex.w is not None:
    ex.w.close()
