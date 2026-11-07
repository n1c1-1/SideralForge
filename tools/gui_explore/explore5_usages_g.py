"""usages G: after reopening: Undo (menu and Ctrl+Z); names and numbering after a rename and a deletion, before and
after reopening; a darks folder and one light gone outside; the project's own folder renamed, opened from Home's
card."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore5_usages_common import EXC, U, _, log, panel, ptexts, start, wait  # noqa: E402
from PySide6.QtWidgets import QLineEdit, QWidget  # noqa: E402

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


def names():
    return {"sessions": [(x.id, x.name) for x in ex.s.p.sessions], "flows": [q.name for q in ex.s.paths],
            "blocks": sorted(it.title for it in ex.blocks() if it.node.key in ("session", "sources"))}


# ================================================================== U. undo after reopening
if want("U"):
    lights = ex.raws(f"AnnulU_{rep}/l", 5)
    root, _c = u.new_from_home(f"AnnulU_{rep}", "blank")
    u.place("session")
    u.add_frames(1, lights, 0)
    u.place("calibration", "registration", "stacking")
    u.save()
    u.reopen_new_window(root)
    a0 = (ex.w.a_undo.isEnabled(), ex.w.a_undo.toolTip(), ex.w.a_undo.text())
    st0 = ex.w.status.text()
    n = len(ex.popups)
    from explore3_sources_common import undo as undo_k  # noqa: E402
    ex.key(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Key_Z, ex.view.viewport(),
           __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.ControlModifier)
    wait(ex, 0.5)
    log("U after reopen: undo action", a0, "status before", st0, "after Ctrl+Z", ex.w.status.text(), u.toasts[-1:],
        ex.popups[n:], names())
    ex.w.close_project()

# ================================================================== N. names and numbering across a reopen
if want("N"):
    l1 = ex.raws(f"NomsN_{rep}/a", 4)
    l2 = ex.raws(f"NomsN_{rep}/b", 4, start=7000)
    l3 = ex.raws(f"NomsN_{rep}/c", 4, start=7500)
    root, _c = u.new_from_home(f"NomsN_{rep}", "blank")
    u.place("session")
    u.add_frames(1, l1, 0)
    u.place("calibration", "registration", "stacking")
    z = ex.new_flow_zone()
    ex.drag_palette("session", ex.ghost_center(z))
    wait(ex, 0.8)
    u.add_frames(2, l2, 0)
    u.place("calibration", "registration", "stacking")
    n1 = names()
    # session 2 removed (with its flow)
    ex.right_click_block(u.blk("session", sid=2), _("Delete"))
    wait(ex, 0.8)
    n2 = names()
    u.save()
    u.reopen_new_window(root)
    n3 = names()
    # a new parallel flow after the reopen
    z = ex.new_flow_zone()
    ex.drag_palette("session", ex.ghost_center(z))
    wait(ex, 0.8)
    n4 = names()
    log("N before", n1, "\n  removed", n2, "\n  reopened", n3, "\n  new flow", n4, u.toasts[-2:])
    # rename Session 1 "Session 3"-like: the next default name
    nxt = n4["sessions"][-1][1]
    ex.right_click_block(u.blk("session", sid=n4["sessions"][-1][0]), _("Delete"))
    wait(ex, 0.6)
    ex.dialog_handler = name_dialog(nxt)
    ex.right_click_block(u.blk("session", sid=1) or u.blk("sources"), _("Rename session"))
    wait(ex, 0.6)
    u.save()
    u.reopen_new_window(root)
    z = ex.new_flow_zone()
    ex.drag_palette("session", ex.ghost_center(z))
    wait(ex, 0.8)
    n5 = names()
    log("N session 1 renamed", repr(nxt), "then a new session after reopen", n5, u.toasts[-2:])
    ex.shot(f"G_N{rep}_names")
    ex.w.close_project()

# ================================================================== X. outside: a darks folder gone, one light gone
if want("X"):
    lights = ex.raws(f"DehorsX_{rep}/lights", 6)
    darks = ex.raws(f"DehorsX_{rep}/darks", 4, start=8000)
    root, _c = u.new_from_home(f"DehorsX_{rep}", "blank")
    u.place("session")
    u.add_frames(1, lights, 0)
    u.add_frames(1, darks, 1)
    u.place("calibration", "registration", "stacking")
    u.save()
    base = u.snapshot()
    ex.w.close()
    ex.settle(4)
    ex.w = None
    os.replace(darks, darks.with_name("darks_partis"))
    one = sorted(lights.iterdir())[0]
    os.replace(one, one.with_name(one.name + ".parti"))
    u.reopen_new_window(root)
    s1 = u.snapshot()
    ex.click_block(u.blk("session") or u.blk("sources"))
    wait(ex, 0.5)
    pt = ptexts(ex)
    blk = u.blk("session") or u.blk("sources")
    cal = u.blk("calibration")
    log("X status", s1["status"], "\n  run", s1["run"], "\n  session", blk.summary if blk else None,
        "\n  calibration", cal.summary if cal else None, cal.toolTip()[:300] if cal else None,
        "\n  panel", [t for t in pt if "·" in t or "Dark" in t or "dark" in t or "introuv" in t][:12])
    ex.shot(f"G_X{rep}_darks_gone")
    ex.w.close_project()
    os.replace(darks.with_name("darks_partis"), darks)

# ================================================================== H. the project's folder renamed, opened from Home's card
if want("H"):
    root, _c = u.new_from_home(f"DossierH_{rep}", "classic")
    u.save()
    ex.w.close_project()
    wait(ex, 0.5)
    new = root.with_name(root.name + "_deplace")
    os.replace(root, new)
    ex.w.show_home()
    wait(ex, 4.0)
    from astro_gpu.client.home import ProjectCard  # noqa: E402
    from PySide6.QtWidgets import QLabel  # noqa: E402
    log("H home labels", [x.text() for x in ex.w.findChildren(QLabel) if x.isVisible() and ("projet" in x.text()
                                                                                         or "Bench" in x.text())][:10],
        "ws", sorted(x.name for x in ex.ws.iterdir()))
    cards = [c for c in ex.w.findChildren(ProjectCard) if c.isVisible()]
    mine = [c for c in cards if c.info.name == root.name or root.name in (c.toolTip() or "")]
    log("H cards", [(c.info.name, c.title.text() if hasattr(c, "title") else "") for c in cards][:8], len(mine))
    ex.shot(f"G_H{rep}_home_before_click")
    n = len(ex.popups)
    if mine:
        ex.click(mine[0])
        wait(ex, 1.0)
    log("H click on the card", ex.popups[n:], "project", ex.w.project.root if ex.w.project else None,
        "status", ex.w.status.text())
    ex.shot(f"G_H{rep}_after_click")
    ex.w.show_home()
    wait(ex, 0.8)
    cards = [c for c in ex.w.findChildren(ProjectCard) if c.isVisible()]
    log("H cards after", [c.info.name for c in cards][:8])
log("EXC", len(EXC), "\n".join(e[-1200:] for e in EXC))
if ex.w is not None:
    ex.w.close()
