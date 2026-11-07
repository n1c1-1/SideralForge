"""run_status, phase A: what the run controls show for many graphs built through the GUI (a dump to read; no notes).

    python explore_run_observe.py [scenario ...] > dump.txt
"""
import faulthandler
import sys
import traceback

from harness import Explorer, stray_qt_errors
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QLabel, QPushButton

faulthandler.dump_traceback_later(900, exit=True)
stray_qt_errors()
LANG = "en" if "--en" in sys.argv else "fr"
ex = Explorer("run_status_dump", lang=LANG)
args = [a for a in sys.argv[1:] if not a.startswith("--")]


def P(*a):
    print(*a, flush=True)


def vis_texts(w):
    out = []
    for x in w.findChildren(QLabel) + w.findChildren(QAbstractButton):
        try:
            if x.isVisible() and x.text():
                out.append(x.text().replace("\n", " / "))
        except RuntimeError:
            pass
    return out


def top(tag):
    w, s = ex.w, ex.s
    P(f"=== {tag}")
    P("  paths:", [(q.pid, q.name, (q.source or {}).get("path", "") if q.source else None,
                    [n["block"] for n in (s.graph(q) or [])]) for q in s.paths])
    P("  blocks:", [(it.node.ref, it.title, it.summary if hasattr(it, "summary") else "", it.node.cut, it.node.grey,
                     getattr(it, "bp", None)) for it in sorted(ex.blocks(), key=lambda i: (i.pos().y(), i.pos().x()))])
    P("  run:", repr(w.bt_run.text()), "enabled", w.bt_run.isEnabled(), "tip", repr(w.bt_run.toolTip()))
    P("  status:", repr(w.status.text()), "| line", repr(getattr(w, "_line", "")))
    P("  combo:", [(w.device_cb.itemText(i), w.device_cb.itemData(i), w.device_cb.itemData(i, Qt.ToolTipRole))
                   for i in range(w.device_cb.count())], "cur", w.device_cb.currentText(), "en", w.device_cb.isEnabled())
    P("  menu: run_all", w.a_run.isEnabled(), "here", w.a_here.isEnabled(), "stop", w.a_stop.isEnabled(),
      "| sel", s.sel_step, "cur", s.current)


def plan(tag, here=False):
    try:
        d = ex.w.run.run_to_here() if here else ex.w.run.launch()
    except Exception as e:  # noqa: BLE001
        P(f"  !! plan[{tag}] EXC {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    ex.settle(6)
    if d is None:
        P(f"  plan[{tag}]: no window")
        return None
    P(f"  plan[{tag}] title {d.windowTitle()!r} jobs {[(j.stage, j.block, j.wf) for j in d.plan.jobs]}")
    for r in d.rows:
        P(f"     row {r.name.text()!r} | who {r.who.text()!r} | dur {r.dur.text()!r} | uptodate {r.row.uptodate}"
          f" | tip {r.toolTip()!r}")
    P(f"     summary {d.summary.text()!r} detail {d.detail.text()!r} pause {d.pause_note.text()!r} "
      f"more {d.pause_more.text()!r} warn {[x for x in vis_texts(d) if False]}")
    P(f"     buttons run {d.b_run.isVisible()}/{d.b_run.isEnabled()} cancel {d.b_cancel.isVisible()}:"
      f"{d.b_cancel.text()!r} review {d.review.isVisible()}")
    P(f"     texts {vis_texts(d)}")
    return d


def press_run(d, tag):
    """The window's Run: blocked on purpose; how the failure to start is said."""
    if d is None or not d.b_run.isVisible() or not d.b_run.isEnabled():
        P(f"  press_run[{tag}]: no Run to press")
        return
    ex.click(d.b_run)
    ex.settle(30)
    P(f"  press_run[{tag}] phase {d.phase() if not d.isHidden() or True else '?'} summary {d.summary.text()!r}")
    P(f"     recap {vis_texts(d.recap) if d.recap.isVisible() else '-'}")
    P(f"     texts {vis_texts(d)}")
    P(f"     status {ex.w.status.text()!r} runbtn {ex.w.bt_run.text()!r} running {ex.w.run.running}")
    P(f"     console {ex.w.logp.console.toPlainText()!r}")


def log(tag):
    lp = ex.w.logp
    lp.refresh()
    rows = [(lp.table.item(i, 0).text(), lp.table.item(i, 1).text()) for i in range(lp.table.rowCount())]
    P(f"  log[{tag}] {rows}")
    P(f"  console {lp.console.toPlainText()!r} placeholder {lp.console.placeholderText()!r}")


def inspect(item, tag):
    """Double-click a block: Inspect; what the centre says."""
    ex.double_click_block(item)
    ex.settle(15)
    w = ex.w
    t = w.tune
    P(f"  inspect[{tag}] mode {w.mode} sel {ex.s.sel_step} page {t.pages.currentIndex() if t else None}")
    if t is not None:
        P(f"     strip {[x for x in vis_texts(t.strip)]}")
        page = t.pages.currentWidget()
        txt = vis_texts(page)
        P(f"     page {txt[:60]}")
        for b in page.findChildren(QPushButton):
            if b.isVisible() and ("Run" in b.text() or "Exécuter" in b.text() or "Calculer" in b.text()):
                P(f"     button {b.text()!r} en {b.isEnabled()} tip {b.toolTip()!r}")
    P(f"     status {w.status.text()!r}")
    w.set_mode(0)
    ex.settle(6)


def every_block(tag, here=True, insp=True):
    keys = [(it.node.ref, it.node.paths[0].pid if it.node.paths else None) for it in ex.blocks()]
    for ref, pid in keys:
        it = next((x for x in ex.blocks() if x.node.ref == ref and
                   (x.node.paths[0].pid if x.node.paths else None) == pid), None)
        if it is None:
            continue
        ex.click_block(it)
        P(f"  -- block {ref} pid {pid}: sel {ex.s.sel_step} a_here {ex.w.a_here.isEnabled()} status "
          f"{ex.w.status.text()!r}")
        if here:
            d = plan(f"here {ref}", True)
            if d is not None:
                d.close()
                ex.settle()
        if insp:
            it = next((x for x in ex.blocks() if x.node.ref == ref and
                       (x.node.paths[0].pid if x.node.paths else None) == pid), None)
            if it is not None:
                inspect(it, ref)


def place(*blocks):
    for b in blocks:
        n = len(ex.popups)
        ex.double_click_card(b)
        ex.settle(12)
        P(f"  placed {b}: sel {ex.s.sel_step} toast {getattr(ex.view, '_toast_text', '')!r} popups {ex.popups[n:]}")


def menu_of(item):
    m = ex.right_click_block(item)
    return [(a.get("text"), a.get("enabled"), a.get("checked")) for a in m]


# ---------------------------------------------------------------------------------------------- scenarios
def sc_blank():
    ex.new_project("Vide")
    ex.settle(15)
    top("blank")
    log("blank")
    plan("blank all")
    ex.close_dialogs()
    ex.key(Qt.Key_F5, ex.w)
    P("  after F5 dialogs", [d.__class__.__name__ for d in ex.dialogs()], "status", repr(ex.w.status.text()))
    ex.close_dialogs()
    ex.key(Qt.Key_F5, ex.w, Qt.ShiftModifier)
    P("  after Shift+F5 dialogs", [d.__class__.__name__ for d in ex.dialogs()], "status", repr(ex.w.status.text()))
    ex.close_dialogs()


def sc_session_alone():
    ex.new_project("SessionSeule")
    ex.settle(10)
    place("session")
    top("session placed empty")
    every_block("session empty")
    log("session empty")
    ex.click(ex.w.bt_run)
    P("  clicked Run (disabled?) status", repr(ex.w.status.text()), [d.__class__.__name__ for d in ex.dialogs()])
    ex.key(Qt.Key_F5, ex.w)
    P("  F5 status", repr(ex.w.status.text()), [d.__class__.__name__ for d in ex.dialogs()])
    ex.close_dialogs()


def sc_session_lights_no_wf():
    root = ex.project("session")
    ex.open(root)
    ex.settle(15)
    top("session with lights, no workflow")
    every_block("lights no wf")
    plan("all")
    ex.close_dialogs()


def sc_stack_only():
    root = ex.project("session")
    ex.open(root)
    ex.settle(15)
    place("stacking")
    top("session->stacking")
    d = plan("all")
    press_run(d, "stack only")
    ex.close_dialogs()
    top("after failed start")
    log("after failed start")
    every_block("stack only")


def sc_full():
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    top("session + empty workflow")
    plan("here on empty", True)
    ex.close_dialogs()
    place("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "color", "denoise", "stars")
    top("full chain")
    plan("all")
    ex.close_dialogs()
    every_block("full")
    log("full")


def sc_repeated():
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "stacking", "gradient", "denoise", "gradient", "denoise")
    top("repeated")
    plan("all")
    ex.close_dialogs()
    every_block("repeated")
    for it in ex.blocks():
        if it.node.key.startswith(("fond", "debruitage")):
            P(f"  menu {it.node.key}: {menu_of(it)}")
    log("repeated")


def sc_branch():
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "stacking", "astrometry")
    ex.drag_palette_hover("gradient", ex.view.viewport().rect().center())
    gs = [(g.branch_key, g.pid, ex.ghost_center(g)) for g in ex.ghosts() if getattr(g, "branch_key", None)]
    P("  branch ghosts", [(a, b) for a, b, _c in gs])
    ex.drag_leave()
    if gs:
        target = next((g for g in gs if g[0] == "astrometrie"), gs[-1])
        r = ex.drag_palette("gradient", target[2])
        P("  branch drop", r)
        ex.settle(15)
    top("branch")
    plan("all")
    ex.close_dialogs()
    every_block("branch")
    log("branch")


def sc_two_flows():
    root = ex.project("two_flows")
    ex.open(root)
    ex.settle(15)
    top("two flows empty")
    d = plan("here on empty", True)
    ex.close_dialogs()
    for sid in (1, 2):
        it = next(x for x in ex.blocks() if x.node.key == "session" and x.node.sid == sid)
        ex.click_block(it)
        P("  selected session", sid, "sel", ex.s.sel_step, "cur", ex.s.current, "here", ex.w.a_here.isEnabled())
        place("stacking")
    top("two flows with stacking")
    d = plan("all")
    press_run(d, "two flows")
    ex.close_dialogs()
    every_block("two flows")


def sc_image(stretched=False, with_file=True):
    if with_file:
        root = ex.project("image")
        ex.open(root)
    else:
        ex.new_project("ImageVide")
        ex.settle(10)
        place("image")
    ex.settle(15)
    top(f"image file={with_file}")
    if stretched:
        it = ex.block("image")
        ex.click_block(it)
        b = next((x for x in ex.page.panel.findChildren(QPushButton) if x.text() in ("Stretched", "Étirée", "Étiré")),
                 None)
        P("  stretched button", b.text() if b else None)
        if b is not None:
            ex.click(b)
            ex.settle(10)
        top("image stretched")
    else:
        it = ex.block("image")
        ex.click_block(it)
        place("astrometry", "gradient", "color")
        top("image chain")
    plan("all")
    ex.close_dialogs()
    every_block("image")
    log("image")


def sc_mono_fusion():
    root = ex.project("mono")
    ex.open(root)
    ex.settle(15)
    top("mono")
    plan("all")
    ex.close_dialogs()
    place("fusion")
    top("mono+fusion")
    plan("all")
    ex.close_dialogs()
    every_block("mono fusion")
    log("mono")


def sc_legacy():
    root = ex.project("legacy")
    ex.open(root)
    ex.settle(15)
    top("legacy Full")
    d = plan("all")
    press_run(d, "legacy")
    ex.close_dialogs()
    every_block("legacy")


def sc_jpg():
    root = ex.project("jpg_mistake")
    ex.open(root)
    ex.settle(15)
    place("stacking")
    top("jpg mistake")
    d = plan("all")
    ex.close_dialogs()
    every_block("jpg", here=False)
    log("jpg")


def sc_two_sessions():
    root = ex.project("two_sessions")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "stacking")
    top("two sessions")
    plan("all")
    ex.close_dialogs()
    every_block("two sessions", here=True)


SC = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}
SC["image_stretched"] = lambda: sc_image(stretched=True)
SC["image_nofile"] = lambda: sc_image(with_file=False)
for name in args or list(SC):
    P(f"\n################ {name}")
    try:
        SC[name]()
    except Exception:  # noqa: BLE001
        P("!! SCENARIO EXC")
        traceback.print_exc(file=sys.stdout)
    P("  popups", ex.popups[-6:], "modals", ex.modals[-4:])
    ex.popups.clear()
    ex.close_dialogs()
ex.close()
P("END")
