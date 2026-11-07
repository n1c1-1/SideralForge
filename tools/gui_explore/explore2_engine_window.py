"""Engine shapes, round 2: graphs built in the real window and computed for real on the CPU (tiny synthetic data).

Runs are allowed through the harness's opt-in (GUI_EXPLORE_CPU_RUNS=1, set here before the harness is imported): the
worker starts, forced on the CPU. Each scenario prints a dump to read; screenshots go to exploration2/engine_shapes.

    python explore2_engine_window.py <scenario> [--en]
"""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["GUI_EXPLORE_CPU_RUNS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import TMP, Explorer, stray_qt_errors  # noqa: E402
from explore_engine_common import bayer_frames, linear_fits, stretched_png  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget  # noqa: E402

from astro_gpu.core import project2  # noqa: E402
from astro_gpu.core import workflow as wfl  # noqa: E402
from astro_gpu.core.project import Target  # noqa: E402
from astro_gpu.core.project2 import Session  # noqa: E402

faulthandler.dump_traceback_later(600, exit=True)
stray_qt_errors()
LANG = "en" if "--en" in sys.argv else "fr"
ex = Explorer("engine_shapes", lang=LANG)
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SLOT_ERRORS: list[str] = []


def _hook(t, v, tb):
    SLOT_ERRORS.append("".join(traceback.format_exception(t, v, tb)))
    print("!! SLOT EXCEPTION", "".join(traceback.format_exception(t, v, tb))[-1800:], flush=True)


sys.excepthook = _hook


def P(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------------------------- projects
def frames_project(name: str, nights: int = 1, n: int = 6, calib: bool = False) -> Path:
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    p.stack_mode = "per_session"
    for i in range(1, nights + 1):
        d = TMP / "data" / name / f"n{i}"
        kw = {}
        if calib:
            kw = {"darks": [str(bayer_frames(d / "darks", n=3, kind="dark", seed=20 + i))],
                  "flats": [str(bayer_frames(d / "flats", n=3, kind="flat", seed=30 + i))]}
        p.sessions.append(Session(i, f"Session {i}", lights=[str(bayer_frames(d / "lights", n=n, seed=10 + i))], **kw))
    project2.save(p)
    w = p.add_workflow("Workflow", wfl.TEMPLATES["Classic stack"])
    p.set_blocks(w.id, [])
    project2.save(p)
    return root


def image_project(name: str, stretched: bool = False) -> Path:
    root = ex.ws / name
    p = project2.create(root, name, Target(name="M 31"))
    d = TMP / "data" / name
    f = stretched_png(d) if stretched else linear_fits(d)
    w = p.add_workflow("Image", wfl.TEMPLATES["Classic stack"])
    w.source = {"path": str(f), "stretched": stretched}
    p.set_blocks(w.id, [])
    project2.save(p)
    return root


# ---------------------------------------------------------------------------------------------- gestures
def place(*blocks):
    """Double-click on the palette's cards, the last block of the flow selected first (as a user does)."""
    for b in blocks:
        last = max((it for it in ex.blocks() if it.node.key not in ("final",)),
                   key=lambda it: (it.node.col, -it.node.y), default=None)
        if last is not None:
            ex.click_block(last)
        ex.double_click_card(b)
        ex.settle(10)
        P(f"  placed {b}: {[it.node.key for it in ex.blocks()]} toast {toast()!r}")


def cpu():
    cb = ex.w.device_cb
    i = cb.findData("cpu")
    cb.setCurrentIndex(i)
    cb.activated.emit(i)
    ex.settle(4)
    P("  device:", cb.currentText(), "| torch cuda:", _cuda())


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return "no torch"


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
    t = getattr(ex.view, "toast", None)
    try:
        return t.text() if t is not None and t.isVisible() else ""
    except Exception:  # noqa: BLE001
        return ""


def blocks_desc():
    return [f"{it.node.key}[{','.join(str(q.pid) for q in it.node.paths)}] '{it.title}' / '{it.summary}'"
            + (f" tip={it.toolTip()[:120]!r}" if it.toolTip() else "")
            for it in sorted(ex.blocks(), key=lambda b: (b.node.col, b.node.y))]


def blk(key, i=0):
    xs = [it for it in sorted(ex.blocks(), key=lambda b: (b.node.col, b.node.y)) if it.node.key == key]
    return xs[i] if len(xs) > i else None


def run_dialog():
    return ex.w.run.dialog


def dump_dialog(tag):
    d = run_dialog()
    if d is None:
        P(f"  [{tag}] no run dialog")
        return
    try:
        rows = [(r.name.text(), r.mark.state, r.dur.text(), r.toolTip().replace("\n", " / ")[:200]) for r in d.rows]
    except RuntimeError:
        rows = "gone"
    P(f"  [{tag}] dialog {d.windowTitle()!r} phase {d.phase()} visible {d.isVisible()}")
    P(f"     rows {rows}")
    P(f"     texts {vis_texts(d)}")


def click_run():
    """The Run button of the window (Run all, it starts at once)."""
    b = ex.w.bt_run
    P(f"  Run button: {b.text()!r} enabled {b.isEnabled()} tip {b.toolTip()!r}")
    if b.isEnabled():
        ex.click(b)
    return b.isEnabled()


def wait_run(timeout=150, stop_after: float | None = None) -> str:
    """Wait for the run to end (settling the window meanwhile); ``stop_after``: click Stop after that many seconds of
    running (the Stop of the run dialog)."""
    t0 = time.time()
    r = ex.w.run
    stopped = False
    seen_running = False
    while time.time() - t0 < timeout:
        ex.settle(3)
        if r.phase in ("running", "stopping"):
            seen_running = True
        if stop_after is not None and not stopped and r.phase == "running" and time.time() - t0 > stop_after:
            d = run_dialog()
            P(f"  stop clicked at {time.time() - t0:.1f}s; status line {ex.w.status.text()!r} run button "
              f"{ex.w.bt_run.text()!r}")
            ex.shot(f"{ARGS[0]}_running_before_stop", d if d is not None else None)
            ex.click(d.b_stop) if d is not None and d.b_stop.isVisible() else r.stop()
            stopped = True
        if r.phase == "idle" and r.track is not None and (r.track.finished or r.track.never_started):
            break
        time.sleep(0.05)
    t = r.track
    st = t.status if t is not None else "none"
    P(f"  run ended: {st} after {time.time() - t0:.1f}s (seen running {seen_running}); flash/status "
      f"{ex.w.status.text()!r}")
    for _ in range(10):
        ex.settle(3)
        time.sleep(0.05)
    return st


def store_dirs(root: Path):
    s = root / "store"
    if not s.exists():
        return {}
    return {d.name: sorted(x.name for x in d.iterdir()) for d in s.iterdir() if d.is_dir()}


def events(root: Path):
    f = root / "logs" / "events.jsonl"
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def stage_runs(root: Path, run_index: int = -1):
    """(stage, key, status, reused) of the stage_end events of one run (the last by default)."""
    evs = events(root)
    runs = [e.get("run") for e in evs if e.get("type") == "run_start"]
    if not runs:
        return []
    rid = runs[run_index]
    return [(e.get("stage"), e.get("key"), e.get("status"), bool(e.get("reused")), (e.get("error") or "")[:200])
            for e in evs if e.get("run") == rid and e.get("type") == "stage_end"]


def log_rows():
    lp = ex.w.logp
    lp.refresh()
    return [(lp.table.item(i, 0).text(), lp.table.item(i, 1).text()) for i in range(lp.table.rowCount())]


def panel_rows():
    from astro_gpu.client.panel import CompactRow
    pn = ex.page.panel
    return [r for r in pn.findChildren(CompactRow) if r.isVisibleTo(pn)]


def panel_texts():
    pn = ex.page.panel
    out = []
    for x in [pn] + pn.findChildren(QWidget):
        if not x.isVisibleTo(pn):
            continue
        if isinstance(x, (QLabel, QAbstractButton)) and x.text():
            out.append(x.text())
    return out


def set_setting(name: str, value) -> bool:
    """Type a value in the panel's editor of a setting and press Enter (the user's gesture)."""
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox
    row = next((r for r in panel_rows() if r.r.get("name") == name), None)
    if row is None:
        P(f"  setting {name}: not in the panel; there: {[r.r.get('name') for r in panel_rows()]}")
        return False
    e = row.editor
    P(f"  setting {name} ({row.r.get('stage')}): {row.r.get('value')} -> {value}")
    if isinstance(e, QComboBox):
        i = e.findData(value)
        e.setCurrentIndex(i)
    elif isinstance(e, (QSpinBox, QDoubleSpinBox)):
        e.setFocus()
        e.selectAll()
        QTest.keyClicks(e, str(value).replace(".", "," if LANG == "fr" else "."))
        QTest.keyClick(e, Qt.Key_Return)
    else:
        e.setFocus()
        e.selectAll()
        QTest.keyClicks(e, str(value))
        QTest.keyClick(e, Qt.Key_Return)
    ex.settle(10)
    return True


def show_state(tag):
    s = ex.s
    P(f"--- {tag}")
    P("  blocks:", blocks_desc())
    P("  status:", repr(ex.w.status.text()), "| run:", repr(ex.w.bt_run.text()), ex.w.bt_run.isEnabled(),
      repr(ex.w.bt_run.toolTip()))
    P("  toast:", repr(toast()), "| popups:", ex.popups[-3:], "| slot errors:", len(SLOT_ERRORS))


def go_post():
    ex.click(ex.w.mode_btns[2])
    ex.settle(10)
    t0 = time.time()
    while time.time() - t0 < 40:                       # Post renders in its worker
        ex.settle(5)
        time.sleep(0.1)
    page = ex.w.post_holder.page if ex.w.post_holder is not None else None
    P("  post page:", type(page).__name__, "texts:", vis_texts(page)[:40] if page is not None else None)
    img = getattr(getattr(getattr(page, "view", None), "src", None), "img1", None)
    P("  post image:", None if img is None else (img.shape, float(img.mean()), float(img.std())))
    return page


# ---------------------------------------------------------------------------------------------- scenarios
def sc_repeated_image():
    """Image, Gradient twice (double-click twice), CPU run from the window: run window rows, a folder for each, Post."""
    root = image_project("rep_img")
    ex.open(root)
    cpu()
    place("gradient", "gradient")
    show_state("placed")
    click_run()
    ex.settle(5)
    dump_dialog("start")
    st = wait_run()
    dump_dialog("end")
    ex.shot("rep_img_run_end", run_dialog())
    P("  store:", store_dirs(root))
    P("  stage_end:", stage_runs(root))
    show_state("after run")
    ex.shot("rep_img_graph_after")
    P("  log:", log_rows()[:12])
    ex.close_dialogs()
    page = go_post()
    ex.shot("rep_img_post")
    return st


def sc_gradient2_setting():
    """Gradient 2's setting changed in the panel (before a run, then after a run: Replace / Try as variant)."""
    root = image_project("g2set")
    ex.open(root)
    cpu()
    place("gradient", "gradient")
    g2 = blk("fond~2") or blk("fond", 1)
    P("  g2:", g2.node.key if g2 else None, g2.title if g2 else None)
    ex.click_block(g2)
    P("  panel rows:", [(r.r.get("stage"), r.r.get("name"), r.r.get("value")) for r in panel_rows()])
    P("  panel texts:", panel_texts()[:30])
    names = [r.r.get("name") for r in panel_rows() if r.r.get("type") in ("int", "float")]
    target = names[0] if names else None
    if target:
        row = next(r for r in panel_rows() if r.r.get("name") == target)
        v0 = row.r.get("value")
        nv = (int(v0) + 1) if row.r.get("type") == "int" else round(float(v0) * 1.5 + 0.1, 3)
        set_setting(target, nv)
        P("  params now:", {w.name: w.params for w in ex.s.p.workflows})
        show_state("after setting before run")
        ex.shot("g2set_before_run")
    click_run()
    st = wait_run()
    ex.close_dialogs()
    ex.click_block(blk("fond~2") or blk("fond", 1))
    if target:
        row = next(r for r in panel_rows() if r.r.get("name") == target)
        v0 = row.r.get("value")
        nv = (int(v0) + 1) if row.r.get("type") == "int" else round(float(v0) * 1.5 + 0.1, 3)
        set_setting(target, nv)
        P("  pending:", ex.s.pending_change)
        P("  panel texts after change on a computed block:", panel_texts()[:40])
        ex.shot("g2set_after_run_change")
        show_state("after setting after run")
        # Replace
        from PySide6.QtWidgets import QPushButton
        rep = next((b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and
                    b.text().startswith(("Replace", "Remplacer"))), None)
        P("  replace button:", rep.text() if rep else None)
        if rep is not None:
            ex.click(rep)
            ex.settle(10)
        P("  params now:", {w.name: w.params for w in ex.s.p.workflows})
        show_state("after Replace")
        ex.shot("g2set_after_replace")
        click_run()
        ex.settle(4)
        dump_dialog("second run start")
        st2 = wait_run()
        dump_dialog("second run end")
        P("  stage_end 2nd:", stage_runs(root))
    return st


def sc_outdated_frames():
    """Frames: Registration, Stacking, Gradient; run; change Stacking's setting -> what is Outdated, Run again."""
    root = frames_project("outd")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient")
    show_state("placed")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    st = wait_run(200)
    dump_dialog("end")
    ex.shot("outd_run_end", run_dialog())
    P("  stage_end:", stage_runs(root))
    ex.close_dialogs()
    show_state("after run")
    ex.shot("outd_after_run")
    stk = blk("empilement")
    ex.click_block(stk)
    rows = [(r.r.get("stage"), r.r.get("name"), r.r.get("type"), r.r.get("value"), r.r.get("choices"))
            for r in panel_rows()]
    P("  stacking panel rows:", rows)
    # a setting of the stacking: the first choice or number
    cand = next((r for r in panel_rows() if r.r.get("choices") and len(r.r["choices"]) > 1 and r.editor.isEnabled()),
                None) or next((r for r in panel_rows() if r.r.get("type") in ("int", "float") and r.editor.isEnabled()),
                              None)
    if cand is not None:
        if cand.r.get("choices"):
            other = next(v for v, _s in cand.r["choices"] if v != cand.r["value"])
        elif cand.r["type"] == "int":
            other = int(cand.r["value"]) + 1
        else:
            other = round(float(cand.r["value"]) * 1.2 + 0.1, 3)
        set_setting(cand.r["name"], other)
        P("  pending:", {k: v for k, v in (ex.s.pending_change or {}).items() if k != "pids"})
        P("  panel:", panel_texts()[:40])
        ex.shot("outd_change_proposed")
        from PySide6.QtWidgets import QPushButton
        rep = next((b for b in ex.page.panel.findChildren(QPushButton) if b.isVisible() and
                    b.text().startswith(("Replace", "Remplacer"))), None)
        if rep is not None:
            ex.click(rep)
            ex.settle(10)
        show_state("after change")
        ex.shot("outd_after_change")
        for it in ex.blocks():
            P(f"    {it.node.key}: state={getattr(it, 'state', None)!r} summary={it.summary!r}")
        click_run()
        ex.settle(3)
        dump_dialog("rerun start")
        wait_run(200)
        dump_dialog("rerun end")
        P("  stage_end rerun:", stage_runs(root))
    return st


def sc_pause_continue():
    """Frames: Registration, Stacking (review pause), Gradient; Run stops after Stacking; Continue finishes."""
    root = frames_project("pause")
    ex.open(root)
    cpu()
    place("registration", "stacking", "gradient")
    m = ex.right_click_block(blk("empilement"), "Pause vérif" if LANG == "fr" else "Review pause")
    P("  menu:", [(a.get("text"), a.get("enabled"), a.get("checked")) for a in m])
    show_state("pause set")
    P("  breakpoints:", [(w.name, w.breakpoints) for w in ex.s.p.workflows])
    click_run()
    ex.settle(3)
    dump_dialog("start")
    st = wait_run(200)
    dump_dialog("paused")
    ex.shot("pause_paused", run_dialog())
    show_state("paused")
    ex.shot("pause_graph_paused")
    P("  stage_end:", stage_runs(root))
    d = run_dialog()
    if d is not None and d.b_continue.isVisible():
        P("  continue:", d.b_continue.text(), d.b_continue.isEnabled())
        ex.click(d.b_continue)
        ex.settle(5)
        dump_dialog("continued")
        wait_run(200)
        dump_dialog("after continue")
        ex.shot("pause_after_continue", run_dialog())
        P("  stage_end after continue:", stage_runs(root))
    else:
        P("  NO CONTINUE BUTTON")
    show_state("end")
    P("  log:", log_rows()[:12])
    return st


def sc_stop_midway():
    """Frames: Calibration (darks, flats), Registration, Selection, Stacking, Gradient x3; Stop after 4 s; then Run again."""
    root = frames_project("stop", n=10, calib=True)
    ex.open(root)
    cpu()
    place("calibration", "registration", "selection", "stacking", "gradient", "gradient", "gradient")
    show_state("placed")
    click_run()
    st = wait_run(200, stop_after=4.0)
    dump_dialog("after stop")
    ex.shot("stop_after_stop", run_dialog())
    P("  stage_end:", stage_runs(root))
    P("  tmp left:", [str(x) for x in (root / "store").glob("*/*.tmp")] if (root / "store").exists() else [])
    ex.close_dialogs()
    show_state("after stop")
    ex.shot("stop_graph_after_stop")
    P("  log:", log_rows()[:8])
    click_run()
    st2 = wait_run(200)
    dump_dialog("rerun")
    P("  stage_end rerun:", stage_runs(root))
    show_state("after rerun")
    return st, st2


def sc_branch():
    """Image: Gradient, Gradient; a branch under the first Gradient (a Gradient dropped below it); run: shared part once."""
    root = image_project("branch")
    ex.open(root)
    cpu()
    place("gradient", "gradient")
    g1 = blk("fond")
    # the branch: drag the palette's Gradient under the first Gradient; the ghost that is a branch
    c = ex.drag_palette_hover("gradient", ex.point_of(g1))
    gh = [(g, getattr(g, "kind", None), getattr(g, "tip", None) or g.toolTip()) for g in ex.ghosts()]
    P("  ghosts:", [(k, t[:80] if t else t) for _g, k, t in gh])
    ex.drag_leave()
    branch = next((g for g, k, t in gh if "branch" in str(k).lower() or "branche" in str(t).lower()
                   or "branch" in str(t).lower()), None)
    if branch is None:
        g1r = g1.sceneBoundingRect()
        branch = min(ex.ghosts(), key=lambda g: abs(g.sceneBoundingRect().center().x() - g1r.center().x())
                     + abs(g.sceneBoundingRect().center().y() - g1r.bottom() - 40)) if ex.ghosts() else None
    if branch is not None:
        rec = ex.drag_palette("gradient", ex.ghost_center(branch))
        P("  drop:", rec)
    show_state("branched")
    P("  workflows:", [(w.name, w.blocks) for w in ex.s.p.workflows])
    ex.shot("branch_graph")
    click_run()
    ex.settle(3)
    dump_dialog("start")
    st = wait_run()
    dump_dialog("end")
    ex.shot("branch_run_end", run_dialog())
    P("  stage_end:", stage_runs(root))
    P("  store:", store_dirs(root))
    ex.close_dialogs()
    go_post()
    ex.shot("branch_post")
    return st


SCENARIOS = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("sc_")}

if __name__ == "__main__":
    name = ARGS[0]
    P(f"=== {name} ({LANG}) TMP {TMP}")
    try:
        r = SCENARIOS[name]()
        P("=== result", r)
    except Exception as e:  # noqa: BLE001
        P("!! SCENARIO EXCEPTION", type(e).__name__, e, traceback.format_exc()[-2500:])
    P("=== slot errors:", len(SLOT_ERRORS))
    for x in SLOT_ERRORS:
        P(x[-1500:])
    try:
        ex.close()
    except Exception:  # noqa: BLE001
        pass
    os._exit(0)
