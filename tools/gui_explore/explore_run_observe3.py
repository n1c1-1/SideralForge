"""run_status, phase A (3): review states (simulated "done" results, nothing computed), pauses on session and repeated
blocks, mixed data (a Session with lights and an Image without file), "no frames" texts, the English texts.

    python explore_run_observe3.py [scenario ...] > dump.txt
"""
import faulthandler
import sys
import traceback

from harness import Explorer, stray_qt_errors
from PySide6.QtWidgets import QAbstractButton, QLabel, QPushButton

faulthandler.dump_traceback_later(1200, exit=True)
stray_qt_errors()
LANG = "en" if "--en" in sys.argv else "fr"
ex = Explorer("run_status_dump", lang=LANG)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
EXC = []


def hook(t, v, tb):
    EXC.append("".join(traceback.format_exception(t, v, tb)))
    print("!! UNCAUGHT", "".join(traceback.format_exception(t, v, tb))[-1200:], flush=True)


sys.excepthook = hook
PAUSE = "Pause vérif" if LANG == "fr" else "Review pause"


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


def blk(ref, pid=None):
    return next((x for x in ex.blocks() if x.node.ref == ref and
                 (pid is None or (x.node.paths and x.node.paths[0].pid == pid))), None)


def place(*blocks):
    for b in blocks:
        ex.double_click_card(b)
        ex.settle(12)


def plan(tag, here=False):
    d = ex.w.run.run_to_here() if here else ex.w.run.launch()
    ex.settle(6)
    if d is None:
        P(f"  plan[{tag}]: none")
        return None
    P(f"  plan[{tag}] {d.windowTitle()!r}")
    for r in d.rows:
        P(f"     row {r.name.text()!r} who {r.who.text()!r} dur {r.dur.text()!r} up {r.row.uptodate} tip {r.toolTip()!r}")
    P(f"     texts {vis_texts(d)}")
    return d


def fake_done(stages_by_pid):
    """Simulated results: these (workflow, stage) pairs read as done (nothing is computed or written)."""
    from astro_gpu.client import model as mdl
    real = mdl.ProjectModel.state

    def state(self, p, stage):
        p2 = p or self.path
        if p2 is not None and stage in stages_by_pid.get(p2.pid, ()):
            return "done"
        return real(self, p, stage)
    mdl.ProjectModel.state = state
    ex.s._cache.clear()
    ex.s.pathsChanged.emit()
    ex.settle(10)
    return lambda: setattr(mdl.ProjectModel, "state", real)


def sc_review():
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "stacking", "gradient", "gradient")
    s = ex.s
    pid = s.paths[0].pid
    it = blk("calibration")
    ex.right_click_block(it, PAUSE)
    ex.settle(6)
    it = blk("fond")
    ex.right_click_block(it, PAUSE)
    ex.settle(6)
    undo = fake_done({pid: ("inventory", "calibrate")})
    it = blk("calibration")
    P("  calibration done (simulated): bp", it.bp, "state", getattr(it, "state", None), "tip", it.toolTip())
    m = ex.right_click_block(it)
    P("  menu", [(a.get("text"), a.get("enabled"), a.get("checked")) for a in m])
    d = plan("all, paused")
    if d is not None:
        P("   review visible", d.review.isVisible(), vis_texts(d.review))
        ex.shot("dump_review_dialog", d)
        # Open in Inspect
        if d.b_inspect.isVisible():
            ex.click(d.b_inspect)
            ex.settle(15)
            P("   open in inspect: mode", ex.w.mode, "sel", s.sel_step, "dialog visible", d.isVisible())
            P("   page", vis_texts(ex.w.tune.pages.currentWidget())[:25])
            ex.shot("dump_review_inspect")
            ex.w.set_mode(0)
            ex.settle(5)
    ex.close_dialogs()
    it = blk("calibration")
    ex.right_click_block(it, "Marquer" if LANG == "fr" else "Mark reviewed")
    ex.settle(8)
    it = blk("calibration")
    P("  after mark: bp", it.bp, "tip", it.toolTip(), "status", ex.w.status.text())
    P("  log", [(a, b) for a, b in ex.w.logp.rows()][:4])
    d = plan("all after review")
    ex.close_dialogs()
    # Up-to-date rows of repeated blocks: Gradient 1 and 2 simulated done
    undo()
    undo = fake_done({pid: ("inventory", "calibrate", "register", "stack", "gradient", "gradient~2")})
    P("  blocks", [(x.node.ref, x.title, getattr(x, "state", None)) for x in ex.blocks()])
    d = plan("all, repeated up to date")
    ex.close_dialogs()
    it = blk("fond~2")
    ex.click_block(it)
    d = plan("here fond~2 up to date", True)
    ex.close_dialogs()
    ex.double_click_block(blk("fond"))
    ex.settle(15)
    P("  inspect fond (done): head", vis_texts(ex.w.tune)[:20])
    ex.w.set_mode(0)
    undo()


def sc_pause_odd():
    root = ex.project("two_sessions")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "stacking", "denoise", "denoise", "color", "color")
    P("  keys", [(x.node.ref, x.title) for x in ex.blocks()])
    for ref in ("calibration@1", "calibration@2", "debruitage~2", "couleur~2", "couleur", "session@1"):
        it = blk(ref)
        if it is None:
            P("  no", ref)
            continue
        n = len(EXC)
        m = ex.right_click_block(it, PAUSE)
        ex.settle(8)
        it = blk(ref)
        P(f"  pause {ref}: has item {any(PAUSE.lower() in (a.get('text') or '').lower() for a in m)} exc {len(EXC) - n}"
          f" bp {getattr(it, 'bp', None)} wf {[w.breakpoints for w in ex.s.p.workflows]}")
    d = plan("all")
    ex.close_dialogs()


def sc_mixed():
    root = ex.project("session")
    ex.open(root)
    ex.settle(15)
    place("stacking")
    z = ex.new_flow_zone()
    ex.drag_palette("image", ex.ghost_center(z))
    ex.settle(15)
    P("  paths", [(q.pid, q.name, q.source) for q in ex.s.paths])
    P("  run", ex.w.bt_run.text(), ex.w.bt_run.isEnabled(), "status", ex.w.status.text())
    d = plan("all mixed")
    if d is not None and d.b_run.isEnabled():
        ex.click(d.b_run)
        ex.settle(25)
        P("  after Run:", vis_texts(d))
    ex.close_dialogs()
    it = blk("image")
    ex.click_block(it)
    P("  image selected: a_here", ex.w.a_here.isEnabled())
    d = plan("here image without file", True)
    ex.close_dialogs()


def sc_noframes():
    ex.new_project("SansBrutes")
    ex.settle(10)
    place("session", "stacking", "gradient")
    P("  paths", [(q.pid, q.name, ex.s.graph(q)) for q in ex.s.paths])
    P("  run", ex.w.bt_run.text(), ex.w.bt_run.isEnabled(), repr(ex.w.bt_run.toolTip()), "status", ex.w.status.text())
    for ref in ("empilement", "fond"):
        it = blk(ref)
        if it is None:
            continue
        ex.click_block(it)
        P(f"  {ref}: a_here", ex.w.a_here.isEnabled(), "panel", [t for t in vis_texts(ex.page.panel) if "xécut" in t or "Run" in t])
        plan(f"here {ref}", True)
        ex.close_dialogs()
        ex.double_click_block(blk(ref))
        ex.settle(12)
        page = ex.w.tune.pages.currentWidget()
        P(f"  inspect {ref}:", vis_texts(page)[:12])
        for b in page.findChildren(QPushButton):
            if b.isVisible() and ("xécuter" in b.text() or "Run" in b.text()):
                P("     button", b.text(), b.isEnabled(), repr(b.toolTip()))
        ex.shot(f"dump_noframes_{ref}")
        ex.w.set_mode(0)
        ex.settle(5)


def sc_english():
    """The same states in English: words left in French, or stage names."""
    root = ex.project("two_sessions")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "stacking", "gradient", "gradient", "stars")
    plan("EN all")
    ex.close_dialogs()
    ex.w.launch()
    ex.settle(25)
    d = ex.w.run.dialog
    P("  EN failed start", vis_texts(d) if d else None)
    ex.close_dialogs()
    P("  EN log", ex.w.logp.rows()[:6])
    P("  EN status", ex.w.status.text())
    ex.double_click_block(blk("empilement"))
    ex.settle(12)
    page = ex.w.tune.pages.currentWidget()
    P("  EN inspect", vis_texts(ex.w.tune.strip), vis_texts(page)[:10])
    for b in page.findChildren(QPushButton):
        if b.isVisible() and "Run" in b.text():
            P("     button", b.text(), repr(b.toolTip()))


SC = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}
for name in args or list(SC):
    P(f"\n################ {name}")
    try:
        SC[name]()
    except Exception:  # noqa: BLE001
        P("!! SCENARIO EXC")
        traceback.print_exc(file=sys.stdout)
    P("  popups", ex.popups[-6:], "uncaught", len(EXC))
    ex.popups.clear()
    ex.close_dialogs()
ex.close()
P("END")
