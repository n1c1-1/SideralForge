"""run_status, phase A (2): pauses for review, the device combo and Settings > Compute, the health check, the Log, the
mode buttons, what follows a start that failed. A dump to read; no notes.

    python explore_run_observe2.py [scenario ...] > dump.txt
"""
import faulthandler
import sys
import time
import traceback
from pathlib import Path

from harness import TMP, Explorer, stray_qt_errors
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QComboBox, QLabel, QPushButton

faulthandler.dump_traceback_later(1200, exit=True)
stray_qt_errors()
LANG = "en" if "--en" in sys.argv else "fr"
ex = Explorer("run_status_dump", lang=LANG)
args = [a for a in sys.argv[1:] if not a.startswith("--")]
EXC = []


def hook(t, v, tb):
    EXC.append("".join(traceback.format_exception(t, v, tb)))
    print("!! UNCAUGHT", "".join(traceback.format_exception(t, v, tb))[-1500:], flush=True)


sys.excepthook = hook


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


def toast():
    v = ex.view
    for name in ("_toast", "toast_item", "toast"):
        t = getattr(v, name, None)
        if t is not None and hasattr(t, "text"):
            try:
                return t.text() if callable(t.text) else t.text
            except Exception:  # noqa: BLE001
                pass
    return ""


def blk(ref, pid=None):
    return next((x for x in ex.blocks() if x.node.ref == ref and
                 (pid is None or (x.node.paths and x.node.paths[0].pid == pid))), None)


def place(*blocks):
    for b in blocks:
        ex.double_click_card(b)
        ex.settle(12)


def log_rows():
    lp = ex.w.logp
    lp.refresh()
    return [(lp.table.item(i, 0).text(), lp.table.item(i, 1).text()) for i in range(lp.table.rowCount())]


def plan(tag, here=False):
    d = ex.w.run.run_to_here() if here else ex.w.run.launch()
    ex.settle(6)
    if d is None:
        P(f"  plan[{tag}]: none")
        return None
    P(f"  plan[{tag}] {d.windowTitle()!r} rows {[r.name.text() for r in d.rows]} summary {d.summary.text()!r} "
      f"pause {d.pause_note.text()!r}/{d.pause_note.isVisible()} more {d.pause_more.text()!r} review "
      f"{d.review.isVisible()} run {d.b_run.isVisible()}/{d.b_run.isEnabled()}")
    P(f"     texts {vis_texts(d)}")
    return d


# ---------------------------------------------------------------------------------------------- scenarios
def sc_pause():
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "selection", "stacking", "astrometry", "gradient", "gradient", "denoise",
          "stars")
    P("  keys", [(it.node.ref, it.title) for it in ex.blocks()])
    for ref in ("calibration", "selection", "fond", "fond~2", "etoiles", "composition", "sources"):
        it = blk(ref)
        if it is None:
            P("  no block", ref)
            continue
        n0 = len(EXC)
        m = ex.right_click_block(it, "Pause vérif" if LANG == "fr" else "Review pause")
        ex.settle(10)
        P(f"  pause on {ref}: menu {[(a.get('text'), a.get('enabled'), a.get('checked')) for a in m]}")
        P(f"     exc {len(EXC) - n0} toast {toast()!r} status {ex.w.status.text()!r}")
        it = blk(ref)
        P(f"     bp {getattr(it, 'bp', None)} tip {it.toolTip()!r}" if it else "     gone")
        P(f"     wf breakpoints {[ (w.name, w.breakpoints) for w in ex.s.p.workflows]}")
    P("  log", log_rows()[:12])
    plan("all with pauses")
    ex.close_dialogs()
    it = blk("empilement")
    ex.click_block(it)
    plan("here stacking with pauses", True)
    ex.close_dialogs()
    it = blk("fond~2")
    if it is not None:
        ex.click_block(it)
        plan("here fond~2 with pauses", True)
        ex.close_dialogs()
    # take one off again, and the undo of a pause
    it = blk("calibration")
    m = ex.right_click_block(it, "Pause vérif" if LANG == "fr" else "Review pause")
    ex.settle(8)
    P("  second toggle calibration: checked before", [a.get("checked") for a in m if "ause" in (a.get("text") or "")],
      "toast", toast(), "bp", getattr(blk("calibration"), "bp", None))
    ex.undo()
    P("  after undo: bp", getattr(blk("calibration"), "bp", None), "status", ex.w.status.text())
    ex.redo()
    P("  after redo: bp", getattr(blk("calibration"), "bp", None), "status", ex.w.status.text())
    P("  log", log_rows()[:8])
    ex.shot("dump_pause_marks")


def sc_pause_shared():
    """A pause on a block two workflows share (a branch), and on a two-flow project."""
    root = ex.project("session_empty_wf")
    ex.open(root)
    ex.settle(15)
    place("calibration", "registration", "stacking", "astrometry")
    ex.drag_palette_hover("gradient", ex.view.viewport().rect().center())
    gs = [(g.branch_key, ex.ghost_center(g)) for g in ex.ghosts() if getattr(g, "branch_key", None)]
    ex.drag_leave()
    tgt = next((c for k, c in gs if k == "astrometrie"), None)
    if tgt is not None:
        ex.drag_palette("gradient", tgt)
        ex.settle(15)
    it = blk("empilement")
    m = ex.right_click_block(it, "Pause vérif" if LANG == "fr" else "Review pause")
    ex.settle(8)
    P("  shared pause menu", [(a.get('text'), a.get('checked')) for a in m], "toast", toast())
    P("  bp", getattr(blk("empilement"), "bp", None), blk("empilement").toolTip())
    plan("all shared pause")
    ex.close_dialogs()
    it = blk("fond")
    ex.click_block(it)
    plan("here gradient branch", True)
    ex.close_dialogs()
    P("  log", log_rows()[:5])


def sc_device():
    root = ex.project("session")
    ex.open(root)
    ex.settle(15)
    place("stacking")
    w = ex.w
    cb = w.device_cb

    def show(tag):
        P(f"  [{tag}] combo {cb.currentText()!r} data {cb.currentData()!r} style {cb.styleSheet()!r} pref "
          f"{__import__('astro_gpu.client.prefs', fromlist=['get']).get('device')!r} run {w.bt_run.text()!r} "
          f"status {w.status.text()!r} model.device {getattr(ex.s, 'device', None)!r}")
    show("start")
    for i in range(cb.count()):
        cb.setCurrentIndex(i)
        cb.activated.emit(i)
        ex.settle(10)
        show(f"pick {i} {cb.itemText(i)}")
        seen = {}

        def look(d, seen=seen):
            combos = d.findChildren(QComboBox)
            seen["combos"] = [([c.itemText(k) for k in range(c.count())], c.currentText()) for c in combos
                              if c.isVisible()]
            seen["texts"] = vis_texts(d)[:30]
            d.reject()
        ex.dialog_handler = look
        w.open_settings("compute")
        ex.settle(10)
        P("     settings compute:", seen.get("combos"))
        P("     settings texts:", seen.get("texts"))
    # Settings → Automatic: what the top combo shows
    from astro_gpu.client import prefs
    w.set_device("auto")
    ex.settle(6)
    show("settings auto")
    prefs.set("device", "auto")
    d = ex.w.run.launch()
    ex.settle(6)
    P("  plan on auto", d.summary.text() if d else None)
    ex.close_dialogs()
    w.set_device("cpu")
    ex.settle(6)
    d = ex.w.run.launch()
    ex.settle(6)
    P("  plan on cpu", d.summary.text() if d else None, [r.dur.text() for r in d.rows] if d else None)
    ex.close_dialogs()
    show("cpu")
    P("  log", log_rows()[:5])
    w.set_device("auto")


def sc_health():
    """The health check for frames of a different type (FITS lights and a JPEG), and for raws that cannot be read."""
    from astro_gpu.core import project2
    from astro_gpu.core.project import Target
    import numpy as np
    from PIL import Image
    # FITS lights + one JPEG
    name = "fits_jpg"
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    d = ex.fits_frames(f"{name}/lights", 8)
    j = TMP / "data" / name / "phone" / "IMG_0001.JPG"
    j.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((20, 30, 3), np.uint8)).save(j)
    p.sessions.append(project2.Session(1, "Session 1", lights=[str(d), str(j)]))
    project2.save(p)
    ex.open(root)
    ex.settle(20)
    place("stacking")
    it = blk("sources")
    P("  session block", it.title, "|", it.summary, "| tip", it.toolTip())
    ex.click_block(it)
    ex.settle(10)
    P("  panel", vis_texts(ex.page.panel)[:40])
    d = plan("all with a jpg")
    ex.close_dialogs()
    ex.double_click_block(blk("sources"))
    for _ in range(40):
        ex.settle(5)
        if ex.w.frames.data is not None:
            break
    P("  frames data", ex.w.frames.data is not None)
    P("  inspect page", vis_texts(ex.w.tune.pages.currentWidget()))
    ex.shot("dump_health_fits_jpg")
    ex.w.set_mode(0)
    ex.settle(5)
    # raws that cannot be read: the health check
    root = ex.project("jpg_mistake")
    ex.open(root)
    ex.settle(15)
    ex.double_click_block(blk("sources"))
    t0 = time.time()
    while time.time() - t0 < 20:
        ex.settle(5)
    P("  raws 20 s later: data", ex.w.frames.data is not None, vis_texts(ex.w.tune.pages.currentWidget())[:12])
    ex.w.set_mode(0)
    it = blk("sources")
    ex.click_block(it)
    ex.settle(10)
    P("  panel jpg", vis_texts(ex.page.panel)[:40])
    for b in ex.page.panel.findChildren(QPushButton):
        if b.isVisible() and ("Vérifier" in b.text() or "Review" in b.text() or "Revoir" in b.text()):
            P("  review button", b.text())
            ex.click(b)
            ex.settle(10)
            P("   -> mode", ex.w.mode, "sel", ex.s.sel_step, vis_texts(ex.w.tune.pages.currentWidget())[:15])
            ex.w.set_mode(0)
            break


def sc_log():
    root = ex.project("session")
    ex.open(root)
    ex.settle(15)
    w = ex.w
    P("  log button", w.b_log.text(), "checked", w.b_log.isChecked(), "visible logp", w.logp.isVisible())
    ex.click(w.b_log)
    ex.settle(5)
    P("  after click: checked", w.b_log.isChecked(), "logp", w.logp.isVisible(), "texts", vis_texts(w.logp))
    place("stacking", "astrometry", "gradient")
    ex.undo()
    ex.redo()
    w.save()
    ex.settle(5)
    P("  rows", log_rows())
    f = TMP / "exported_log.txt"
    ex.file_queue.append(str(f))
    ex.click(w.logp.b_export)
    ex.settle(5)
    P("  export asks", ex.file_asks[-1:], "file", f.exists(), f.read_text(encoding="utf-8")[:600] if f.exists() else "")
    # a failed start, then the log
    w.launch()
    ex.settle(30)
    P("  after start: rows", log_rows()[:4])
    P("  console", w.logp.console.toPlainText())
    P("  dialogs", [(d.__class__.__name__, d.windowTitle()) for d in ex.dialogs()])
    dd = w.run.dialog
    if dd is not None:
        P("  dialog texts", vis_texts(dd))
        if dd.b_log.isVisible():
            w.b_log.setChecked(False)
            ex.click(dd.b_log)
            ex.settle(5)
            P("  'show the log' -> log checked", w.b_log.isChecked(), "logp visible", w.logp.isVisible())
    ex.close_dialogs()
    ex.key(Qt.Key_L, w, Qt.ControlModifier)
    P("  ctrl+L ->", w.b_log.isChecked())
    # a second start right after
    w.launch()
    ex.settle(30)
    P("  second start console", w.logp.console.toPlainText()[-300:], "running", w.run.running)
    ex.shot("dump_log_after_fail")
    ex.close_dialogs()
    # language change with the log open
    from astro_gpu.client import app as appmod
    w.change_language("en")
    ex.settle(10)
    P("  EN rows", log_rows()[:6])
    P("  EN console", w.logp.console.toPlainText()[-300:])
    P("  EN status", w.status.text(), "| run", w.bt_run.text(), "| combo", w.device_cb.currentText(),
      [w.device_cb.itemText(i) for i in range(w.device_cb.count())])
    w.change_language("fr")
    ex.settle(10)
    P("  FR again rows", log_rows()[:6])


def sc_modes():
    ex.new_project("Modes")
    ex.settle(10)
    w = ex.w
    for i, b in enumerate(w.mode_btns):
        P(f"  mode button {i} {b.text()!r} enabled {b.isEnabled()} tip {b.toolTip()!r}")
    for i in (1, 2, 0):
        ex.click(w.mode_btns[i])
        ex.settle(8)
        page = w.modes.currentWidget()
        P(f"  blank: mode {i} -> w.mode {w.mode} sel {ex.s.sel_step} texts {vis_texts(page)[:20]}")
        ex.shot(f"dump_modes_blank_{i}")
    root = ex.project("image")
    ex.open(root)
    ex.settle(10)
    ex.double_click_block(blk("image"))
    ex.settle(10)
    t = w.tune
    btn = next((b for b in t.strip.findChildren(QPushButton) if b.text() in ("Images", "Frames")), None)
    P("  image wf strip 'Frames' button", btn.text() if btn else None)
    if btn is not None:
        ex.click(btn)
        ex.settle(10)
        P("   -> sel", ex.s.sel_step, "page", vis_texts(t.pages.currentWidget())[:15])
        ex.shot("dump_image_strip_frames")
    for k in (Qt.Key_F5,):
        ex.key(k, w)
        ex.settle(20)
        P("  F5 in Inspect: dialogs", [(d.__class__.__name__, d.windowTitle()) for d in ex.dialogs()],
          "status", w.status.text())
        ex.close_dialogs()
    ex.key(Qt.Key_Escape, t)
    P("  esc -> mode", w.mode)
    # Shift+F5 with Post selected
    w.set_mode(2)
    ex.settle(5)
    ex.key(Qt.Key_F5, w, Qt.ShiftModifier)
    P("  Shift+F5 in Post: a_here", w.a_here.isEnabled(), "dialogs", [d.windowTitle() for d in ex.dialogs()])
    ex.close_dialogs()
    w.set_mode(0)
    # the Run menu
    mb = w.menuBar()
    for a in mb.actions():
        m = a.menu()
        if m is not None and a.text() in ("Exécution", "Run", "&Exécution", "Exécuter"):
            P("  run menu", a.text(), [(x.text(), x.shortcut().toString(), x.isEnabled()) for x in m.actions()])
    P("  menus", [a.text() for a in mb.actions()])


def sc_failed_start():
    """What a start that failed leaves: the dialog, the status bar, the graph, a second Run."""
    root = ex.project("session")
    ex.open(root)
    ex.settle(15)
    place("stacking")
    w = ex.w
    ex.click(w.bt_run)
    ex.settle(30)
    d = w.run.dialog
    P("  after Run button: dialogs", [(x.__class__.__name__, x.windowTitle()) for x in ex.dialogs()])
    if d is not None:
        P("  dialog", d.windowTitle(), vis_texts(d))
        P("  rows tips", [r.toolTip() for r in d.rows])
        ex.shot("dump_failed_start_dialog", d)
    P("  status", w.status.text(), "run", w.bt_run.text(), w.bt_run.isEnabled(), "statuswidget",
      w._run_widget.isVisible() if w._run_widget else None)
    P("  graph block", [(it.node.ref, it.state if hasattr(it, 'state') else None) for it in ex.blocks()])
    st = Path(w.project.root) / "logs"
    P("  logs dir", sorted(x.name for x in st.iterdir()) if st.exists() else None)
    ex.close_dialogs()
    ex.key(Qt.Key_F5, w)
    ex.settle(30)
    P("  F5 again: dialogs", [(x.__class__.__name__, x.windowTitle()) for x in ex.dialogs()], "console",
      w.logp.console.toPlainText()[-400:])
    ex.close_dialogs()
    # Run to here after a failure, and press Run there
    ex.click_block(blk("empilement"))
    ex.key(Qt.Key_F5, w, Qt.ShiftModifier)
    ex.settle(10)
    d = w.run.dialog
    P("  Shift+F5: dialog", d.windowTitle() if d else None, vis_texts(d) if d else None)
    if d is not None and d.b_run.isVisible() and d.b_run.isEnabled():
        ex.click(d.b_run)
        ex.settle(30)
        P("  pressed Run there:", vis_texts(d))
        ex.shot("dump_failed_start_here", d)
    ex.close_dialogs()


SC = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}
for name in args or list(SC):
    P(f"\n################ {name}")
    try:
        SC[name]()
    except Exception:  # noqa: BLE001
        P("!! SCENARIO EXC")
        traceback.print_exc(file=sys.stdout)
    P("  popups", ex.popups[-6:], "modals", ex.modals[-4:], "uncaught", len(EXC))
    ex.popups.clear()
    ex.close_dialogs()
ex.close()
P("END")
